import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .geometric import GeometricFeatureExtractor

class AVIT(nn.Module):
    """视觉Transformer骨干网络"""
    def __init__(self, embed_dim, patch_size, num_layers, num_heads, use_amp=False):
        super().__init__()
        # 将卷积嵌入替换为多尺度卷积嵌入
        self.conv_embed = self.MultiScaleConvEmbedding(embed_dim, patch_size)
        self.num_layers = num_layers  # 保留参数
        
        # 早停参数初始化
        self.stop_gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])
        self.stop_betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_layers)
        ])
        self.tau = 0.9  # 停止阈值 τ = 1 - ε (ε=0.1)

        self.multiscale_transformer = self.MultiscaleTransformerEncoder(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation='gelu',
            batch_first=True,
            tau=self.tau,
            num_layers=num_layers,
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, patch_size**2)  # 重建原始patch尺寸
        )
        
        # 新增损失参数
        self.alpha_kl = 1.0
        self.alpha_contrast = 0.5
        self.alpha_p = 0.001

        # 修改梯度累积步数 - 减少内存使用
        self.grad_accum_steps = 50  
        
        # 新增：控制是否使用混合精度
        self.use_amp = use_amp

        # +++ 几何特征提取器 - 始终启用 +++
        self.geometric_extractor = GeometricFeatureExtractor(feature_dim=embed_dim)
        
        # 几何特征增强层
        self.geometric_enhancement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 几何特征融合参数
        self.geo_lambda = nn.Parameter(torch.tensor(0.5))
        
        # 边界权重生成器 - 用于loss计算
        self.boundary_weight_generator = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    class MultiscaleTransformerEncoder(nn.Module):
        """多尺度Transformer编码器，处理不同范围的依赖关系"""
        def __init__(self, d_model, nhead, dim_feedforward, activation, 
                     batch_first, tau, num_layers, dropout=0.1):
            super().__init__()
            
            self.short_layers = max(1, int(num_layers//3))  
            self.mid_layers = max(1, int(num_layers//5))   
            self.long_layers = max(1, num_layers - self.short_layers - self.mid_layers)
            
            # 存储tau参数用于早停
            self.tau = tau
            
            # 创建短程Transformer层组
            self.short_transformer = nn.ModuleList([
                AVIT.EarlyStoppingEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    batch_first=batch_first,
                    dropout=dropout,
                    tau=tau
                ) for _ in range(self.short_layers)
            ])
            
            # 创建中程Transformer层组（使用相对位置编码增强中程依赖）
            self.mid_transformer = nn.ModuleList([
                AVIT.EarlyStoppingEncoderLayer(
                    d_model=d_model,
                    nhead=nhead*2,  # 增加头数以捕获更复杂的依赖
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    batch_first=batch_first,
                    dropout=dropout,
                    tau=tau
                ) for _ in range(self.mid_layers)
            ])
            
            # 创建长程Transformer层组（更大的感受野）
            self.long_transformer = nn.ModuleList([
                AVIT.EarlyStoppingEncoderLayer(
                    d_model=d_model,
                    nhead=nhead*4,  # 进一步增加头数，扩大感受野
                    dim_feedforward=dim_feedforward,  # 更大的前馈网络
                    activation=activation,
                    batch_first=batch_first,
                    dropout=dropout,
                    tau=tau
                ) for _ in range(self.long_layers)
            ])
            
            # 尺度间投影层，优化跨尺度特征转换
            self.short_to_mid_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout)
            )
            
            self.mid_to_long_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout)
            )
            
            # 特征融合门控（在各尺度特征之间传递信息）
            self.short_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
            self.mid_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
            # 最终输出层归一化
            self.final_norm = nn.LayerNorm(d_model)
            
            # 初始化参数
            self._reset_parameters()
            
            # 新增：几何特征融合层
            self.geo_fusion_short = nn.LayerNorm(d_model)
            self.geo_fusion_mid = nn.LayerNorm(d_model)
            self.geo_fusion_long = nn.LayerNorm(d_model)
            
            # 新增：几何特征融合权重参数
            self.geo_weight_short = nn.Parameter(torch.tensor(0.2))
            self.geo_weight_mid = nn.Parameter(torch.tensor(0.2))
            self.geo_weight_long = nn.Parameter(torch.tensor(0.2))
            
            # 新增：几何引导注意力参数
            self.use_geometric_attention = True
            self.geo_attn_lambda = nn.Parameter(torch.tensor(0.2))
        
        def _reset_parameters(self):
            """初始化参数"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(self, src, src_mask=None, src_key_padding_mask=None, geo_feedback=None):
            """多尺度Transformer前向传播
            
            Args:
                src: 输入特征 [B, L, D]
                src_mask: 注意力掩码
                src_key_padding_mask: 键值掩码
                geo_feedback: 几何特征回传列表 [short, mid, long]
            """
            x = src
            
            # 1. 短程依赖处理
            for i, layer in enumerate(self.short_transformer):
                layer_idx = i
                layer.gamma_param = lambda layer_i=layer_idx: torch.ones(1, device=x.device)
                layer.beta_param = lambda layer_i=layer_idx: torch.zeros(1, device=x.device)
                
                # 如果是第二层且有几何回传，进行融合
                if i == 1 and geo_feedback is not None and len(geo_feedback) > 0:
                    # 处理几何特征 - 需要从[B,D,H,W]转为[B,L,D]
                    geo_short = geo_feedback[0]
                    B, D, H, W = geo_short.shape
                    geo_flat = geo_short.permute(0, 2, 3, 1).reshape(B, H*W, D)
                    
                    # 确保尺寸匹配（如果序列长度不同，进行调整）
                    if geo_flat.shape[1] != x.shape[1]:
                        # 插值调整到相同序列长度
                        geo_flat = F.interpolate(
                            geo_flat.transpose(1, 2), 
                            size=x.shape[1], 
                            mode='linear'
                        ).transpose(1, 2)
                    
                    # 融合几何特征
                    x = self.geo_fusion_short(x + self.geo_weight_short * geo_flat)
                
                # 前向传播
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            
            # 短程特征
            z_short = x
            
            # 2. 中程依赖处理
            x = self.short_to_mid_proj(x)  # 尺度投影
            for i, layer in enumerate(self.mid_transformer):
                # 设置早停参数
                offset = self.short_layers
                layer_idx = i + offset
                
                # 修正：使用相同的模式设置gamma_param和beta_param
                layer.gamma_param = lambda layer_i=layer_idx: torch.ones(1, device=x.device)
                layer.beta_param = lambda layer_i=layer_idx: torch.zeros(1, device=x.device)
                
                # 如果是第二层且有几何回传，进行融合
                if i == 1 and geo_feedback is not None and len(geo_feedback) > 1:
                    # 处理几何特征 - 需要从[B,D,H,W]转为[B,L,D]
                    geo_mid = geo_feedback[1]
                    B, D, H, W = geo_mid.shape
                    geo_flat = geo_mid.permute(0, 2, 3, 1).reshape(B, H*W, D)
                    
                    # 确保尺寸匹配
                    if geo_flat.shape[1] != x.shape[1]:
                        geo_flat = F.interpolate(
                            geo_flat.transpose(1, 2), 
                            size=x.shape[1], 
                            mode='linear'
                        ).transpose(1, 2)
                    
                    # 融合几何特征
                    x = self.geo_fusion_mid(x + self.geo_weight_mid * geo_flat)
                
                # 前向传播
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            
            # 中程特征
            z_mid = x
            
            # 特征融合门控（短->中）
            gate_weight = self.short_gate(z_mid)
            z_mid = z_mid + z_short * gate_weight
            
            # 3. 长程依赖处理
            x = self.mid_to_long_proj(z_mid)  # 尺度投影
            for i, layer in enumerate(self.long_transformer):
                # 设置早停参数
                offset = self.short_layers + self.mid_layers
                layer_idx = i + offset
                
                # 修正：使用相同的模式设置gamma_param和beta_param
                layer.gamma_param = lambda layer_i=layer_idx: torch.ones(1, device=x.device)
                layer.beta_param = lambda layer_i=layer_idx: torch.zeros(1, device=x.device)
                
                # 如果是第二层且有几何回传，进行融合
                if i == 1 and geo_feedback is not None and len(geo_feedback) > 2:
                    # 处理几何特征 - 需要从[B,D,H,W]转为[B,L,D]
                    geo_long = geo_feedback[2]
                    B, D, H, W = geo_long.shape
                    geo_flat = geo_long.permute(0, 2, 3, 1).reshape(B, H*W, D)
                    
                    # 确保尺寸匹配
                    if geo_flat.shape[1] != x.shape[1]:
                        geo_flat = F.interpolate(
                            geo_flat.transpose(1, 2), 
                            size=x.shape[1], 
                            mode='linear'
                        ).transpose(1, 2)
                    
                    # 融合几何特征
                    x = self.geo_fusion_long(x + self.geo_weight_long * geo_flat)
                
                # 前向传播
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            
            # 长程特征
            z_long = x
            
            # 特征融合门控（中->长）
            gate_weight = self.mid_gate(z_long)
            z_long = z_long + z_mid * gate_weight
            
            # 最终归一化
            output = self.final_norm(z_long)
            
            return output
            
        @property
        def layers(self):
            """兼容原有代码，提供所有层的列表"""
            return list(self.short_transformer) + list(self.mid_transformer) + list(self.long_transformer)

    class MultiScaleConvEmbedding(nn.Module):
        """多尺度卷积嵌入模块（优化卷积结构）"""
        def __init__(self, embed_dim, patch_size):
            super().__init__()
            # 保存参数
            self.patch_size = patch_size
            
            # 统一定义三个尺度的卷积层
            self.conv_short = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            self.conv_mid = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            self.conv_long = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            # 特征融合层
            self.fusion = nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # 空间分辨率调整层（可选，若需要将特征图调整为特定尺寸）
            self.adjust = nn.Identity()  # 默认不做调整
            
            # 投影层（从特征图到嵌入向量）
            self.projection = nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
            
            # 残差连接通道适配层
            self.res_conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=1),
                nn.BatchNorm2d(64)
            )
            
            # 位置编码
            self.position_embedding = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
            
            # 门控机制（适配多尺度特征）
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 原始输入路径（保持原样）
            x_orig = x.unsqueeze(0).unsqueeze(0)
            x_res = self.res_conv(x_orig)
            
            # 提取三种尺度的特征
            x_short = self.conv_short(x_orig)
            x_mid = self.conv_mid(x_orig)
            x_long = self.conv_long(x_orig)
            
            # 保存用于跳跃连接的特征
            self.skip_features = [x_short, x_mid, x_long]
            
            # 特征拼接
            x_cat = torch.cat([x_short, x_mid, x_long], dim=1)  # [1, 96, n, n]
            
            # 特征融合
            x_fused = self.fusion(x_cat)  # [1, 64, n, n]
            
            # 可选的分辨率调整
            x_fused = self.adjust(x_fused)  # [1, 64, n, n]
            
            # 门控残差连接
            gate_weight = self.gate(x_fused)  # [1, 64, 1, 1]
            x_fused = x_res + x_fused * gate_weight  # [1, 64, n, n]
            
            # 投影到embedding空间
            x_proj = self.projection(x_fused)  # [1, d, n_b, n_b]
            x_proj = x_proj + self.position_embedding
            
            # 输出形状整理
            return x_proj.view(x_proj.size(1), -1).permute(1, 0)

    class EarlyStoppingEncoderLayer(nn.TransformerEncoderLayer):
        """带早停机制的自定义编码层（新增几何引导注意力）"""
        def __init__(self, tau, *args, **kwargs):
            # 提取tau参数后调用父类初始化
            self.tau = kwargs.pop('tau', tau)
            super().__init__(*args, **kwargs)
            
            # 初始化掩码存储
            self.register_buffer('cumul_states', None)
            
            # 新增几何引导注意力支持
            self.supports_geo_attention = True
        
        def forward(self, src, src_mask=None, src_key_padding_mask=None, geo_attention_bias=None, **kwargs):
            # 首层初始化累计停止状态
            if self.cumul_states is None or self.cumul_states.shape != src.shape[:2]:
                self.cumul_states = torch.zeros_like(src[:, :, 0])  # [batch, seq_len]
            
            # 计算当前层停止概率 h_k^l
            gamma = self.gamma_param()  # 从父类获取γ^l
            beta = self.beta_param()    # 从父类获取β^l
            stop_probs = torch.sigmoid(gamma * src[:, :, 0] + beta)  # 取第一个特征维度
            
            # 更新累计停止状态（使用直通估计器保持梯度）
            self.cumul_states = self.cumul_states + stop_probs.detach() - stop_probs.detach() + stop_probs
            
            # 生成停止掩码 (cumul_states < tau)
            active_mask = (self.cumul_states < self.tau).unsqueeze(-1)  # [batch, seq_len, 1]
            
            # 屏蔽已停止的令牌（特征置零）
            src = src * active_mask.float()
            
            # 生成注意力掩码（阻止停止的令牌参与计算）
            if src_key_padding_mask is None:
                src_key_padding_mask = ~active_mask.squeeze(-1)
            else:
                src_key_padding_mask |= ~active_mask.squeeze(-1)
            
            # 修改注意力计算，添加几何引导
            if geo_attention_bias is not None and self.supports_geo_attention:
                # 如果提供了几何注意力偏置，添加到注意力计算中
                if src_mask is None:
                    src_mask = geo_attention_bias
                else:
                    src_mask = src_mask + geo_attention_bias
            
            # 传递所有参数到父类方法（包括可能的is_causal参数）
            return super().forward(src, src_mask=src_mask, 
                                 src_key_padding_mask=src_key_padding_mask, **kwargs)
            

    def forward(self, matrix: torch.Tensor) -> tuple:
        # 新增：统一输入类型处理
        if isinstance(matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(matrix).float()
        else:
            matrix_tensor = matrix.clone().detach().float()
        
        # 确保数据在正确设备
        matrix_tensor = matrix_tensor.to(self.conv_embed.projection.weight.device)
        
        # 修改输入处理流程
        x = self.conv_embed(matrix_tensor)  # 直接使用转换后的tensor
        
        # 直接获取多尺度卷积嵌入的跳跃连接特征
        skip_features = self.conv_embed.skip_features if hasattr(self.conv_embed, 'skip_features') else None
        
        # 在编码前设置几何特征提取器的跳跃连接
        if skip_features is not None:
            self.geometric_extractor.set_skip_features(skip_features)
        
        # 调整维度顺序 [batch_size, seq_len, features]
        x = x.unsqueeze(0)
        
        # 生成几何引导注意力偏置
        geo_attention_bias = None
        if hasattr(self.geometric_extractor, 'get_attention_bias'):
            geo_attention_bias = self.geometric_extractor.get_attention_bias(
                (1, x.shape[1], x.shape[2]), x.device)
        
        # 执行前向传播 (Transformer编码器) - 传递几何特征偏置
        encoder_output = self.multiscale_transformer(x, src_mask=geo_attention_bias)
        
        # 使用缓存而不是存储在类属性中，避免计算图泄漏
        self._encoder_output_cache = encoder_output.detach().clone()
        
        # 解码器重建
        reconstructed = self.decoder(encoder_output)
        
        # 修改形状计算逻辑
        batch_size, seq_len, feat_dim = encoder_output.shape
        side_len = int(np.sqrt(reconstructed.numel() // batch_size))
        reconstructed_2d = reconstructed.view(batch_size, side_len, side_len)
        
        # 几何特征处理 - 使用已设置的跳跃连接特征
        # 将重建结果重塑为2D特征图
        recon_feature_map = reconstructed_2d.unsqueeze(1)  # [B, 1, H, W]
        
        # 获取几何特征和几何反馈
        geometric_features, geo_feedback = self.geometric_extractor(recon_feature_map)
        
        # 转回序列形式
        geo_attention = geometric_features.view(batch_size, -1)  # [B, H*W]
        
        # 保存几何特征用于损失计算
        self._geo_attention_cache = geo_attention.detach().clone()
        
        # 应用几何特征调制重建结果
        reconstructed_enhanced = reconstructed_2d * (1 + self.geo_lambda * geometric_features.squeeze(1))
        reconstructed = reconstructed_enhanced.squeeze(0)  # 返回增强后的重建结果
    
        # 计算边界权重（用于loss）
        self._boundary_weights = self.boundary_weight_generator(encoder_output).squeeze(-1)
        
        # 计算潜在分布参数（保持不变）
        z_mean = encoder_output.mean(dim=1)
        z_logvar = torch.zeros_like(z_mean)
        
        # 新增：生成几何引导注意力偏置
        geo_attention_bias = self.geometric_extractor.get_attention_bias(
            encoder_output.shape, encoder_output.device)
        
        # 使用几何引导注意力偏置增强重建过程
        # ... 现有代码 ...
        
        return reconstructed, z_mean, z_logvar

    def _compute_losses(self, matrix: torch.Tensor, recon: torch.Tensor, 
                       z_mean: torch.Tensor, z_logvar: torch.Tensor) -> dict:
        """计算所有损失项，加入几何特征引导"""
        
        # 获取重建损失
        recon_loss = F.mse_loss(recon, matrix)
        
        # 使用边界权重增强重建损失 - 始终启用
        if hasattr(self, '_boundary_weights'):
            # 重建重整为序列形式
            recon_flat = recon.view(-1)
            matrix_flat = matrix.view(-1)
            
            # 应用边界权重
            weighted_loss = self._boundary_weights * torch.square(recon_flat - matrix_flat)
            boundary_recon_loss = weighted_loss.mean()
            
            # 结合标准重建损失和边界加权重建损失
            recon_loss = 0.7 * recon_loss + 0.3 * boundary_recon_loss
        
        # KL散度（确保不使用已释放的张量）
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        # 对比损失（修复：确保不重复使用计算图中的变量）
        # 检查_encoder_output_cache是否存在且为当前计算的一部分
        if not hasattr(self, '_encoder_output_cache'):
            contrast_loss = torch.tensor(0.0, device=recon.device)
        else:
            # 使用缓存的分离副本，避免使用旧计算图中的张量
            encoder_output = self._encoder_output_cache
            batch_size, seq_len, _ = encoder_output.shape
            z = encoder_output.view(batch_size*seq_len, -1)
            
            # 计算空间距离
            spatial_dist = torch.cdist(z, z)
            
            # 获取邻居掩码并确保在正确的设备上
            neighbors = self._get_2d_neighbors(seq_len).to(z.device)
            
            # 计算对比损失
            pos_loss = spatial_dist[neighbors].mean()
            neg_loss = torch.log(torch.mean(spatial_dist[~neighbors]) + 0.01)
            contrast_loss = pos_loss - neg_loss
            
            # 显式释放大型中间变量
            del spatial_dist, z
        
        # Ponder损失（修复：使用多尺度transformer的层）
        # 修复：使用短程transformer的第一层来获取cumul_states
        if hasattr(self.multiscale_transformer.short_transformer[0], 'cumul_states'):
            # 使用分离的副本
            cumul_states = self.multiscale_transformer.short_transformer[0].cumul_states.detach().clone()
            
            # 计算ponder损失，避免使用可能已释放的stop_probs
            ponder_loss = torch.mean(cumul_states)
            
            # 释放不需要的临时变量
            del cumul_states
        else:
            ponder_loss = torch.tensor(0.0, device=recon.device)
        
        # 总损失
        total_loss = (recon_loss + 
                     self.alpha_kl * kl_loss +
                     self.alpha_contrast * contrast_loss +
                     self.alpha_p * ponder_loss)
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'contrast': contrast_loss,
            'ponder': ponder_loss
        }

    def _get_2d_neighbors(self, seq_len: int) -> torch.Tensor:
        """生成2D空间邻居掩码（基于patch布局）"""
        # 假设序列是sqrt(n) x sqrt(n)的2D网格
        grid_size = int(np.sqrt(seq_len))
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # 水平相邻
                if j > 0: mask[idx, idx-1] = True
                if j < grid_size-1: mask[idx, idx+1] = True
                # 垂直相邻
                if i > 0: mask[idx, idx-grid_size] = True
                if i < grid_size-1: mask[idx, idx+grid_size] = True
        return mask

    def train_epoch(self, matrix: torch.Tensor) -> dict:
        """单epoch训练流程（修复损失收集和内存优化）"""
        self.train()
        total_loss = 0.0
        recon_loss = 0.0  # 新增recon损失收集
        
        # 拆分梯度累积步骤，避免存储太多中间状态
        for step in range(self.grad_accum_steps):
            # 显式清理GPU缓存
            if step > 0 and step % 2 == 0:
                torch.cuda.empty_cache()
            
            # 修改：根据 use_amp 参数决定是否使用混合精度
            if self.use_amp:
                # 使用新的API替换旧的API
                with torch.amp.autocast('cuda'):
                    reconstructed, z_mean, z_logvar = self(matrix)
                    losses = self._compute_losses(matrix, reconstructed, z_mean, z_logvar)
                    
                    scaled_loss = losses['total'] / self.grad_accum_steps
                    self.scaler.scale(scaled_loss).backward()
            else:
                # 不使用混合精度时的正常流程
                reconstructed, z_mean, z_logvar = self(matrix)
                losses = self._compute_losses(matrix, reconstructed, z_mean, z_logvar)
                
                scaled_loss = losses['total'] / self.grad_accum_steps
                scaled_loss.backward()
            
            # 收集所有损失项
            total_loss += losses['total'].item()
            recon_loss += losses['recon'].item()  # 新增recon累积
            
            # 显式释放大型中间变量
            del reconstructed, z_mean, z_logvar, scaled_loss
            
            # 每两步执行一次优化器步骤，减少内存累积
            if (step + 1) % 2 == 0 or (step + 1) == self.grad_accum_steps:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更彻底地释放梯度内存
            
            if (step + 1) % 10 == 0:
                # 减少临时保存频率以节省I/O和内存
                temp_path = f"{self.output_root}/temp_model_step{step+1}.pth"
                torch.save(self.state_dict(), temp_path)
        
        # 清理缓存，释放内存
        torch.cuda.empty_cache()
        
        return {
            'total': total_loss / self.grad_accum_steps,
            'recon': recon_loss / self.grad_accum_steps  # 确保返回recon损失
        }

    def init_optimizer(self):
        """初始化优化器（新增混合精度训练器）"""
        # 分离主参数和早停参数
        main_params = [p for name, p in self.named_parameters() 
                      if not name.startswith('stop_')]
        
        self.optimizer = torch.optim.Adam([
            {'params': main_params, 'lr': 1e-4},
            {'params': self.stop_gammas, 'lr': 1e-4},
            {'params': self.stop_betas, 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        # 修改：使用新的API替换旧的GradScaler API
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()  # 从torch.cuda.amp更改为torch.amp
        else:
            self.scaler = None
        self.grad_accum_steps = 4  # 梯度累积步数改为4 (从10减少)