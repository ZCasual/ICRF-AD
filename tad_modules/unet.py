import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionGuidedConv(nn.Module):
    """注意力引导卷积层：集中处理重要区域"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # 注意力评估分支
        self.attention_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 生成注意力图
        attention_map = self.attention_branch(x)
        
        # 应用卷积
        features = self.conv(x)
        features = self.bn(features)
        
        # 注意力加权
        weighted_features = features * (1.0 + 0.5 * attention_map)
        output = self.activation(weighted_features)
        
        return output, attention_map

class EntropyBasedEarlyStopping(nn.Module):
    """基于信息熵的早停机制"""
    def __init__(self, threshold=0.9, channel_dim=16):
        super().__init__()
        self.threshold = threshold
        self.channel_dim = channel_dim
        
        # 添加一个占位参数，确保module有参数可遍历
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        # 熵评估层将在前向传播中动态创建
        self.entropy_estimator = None
        
    def _build_entropy_estimator(self, channels, device=None):
        """动态构建熵估计器，适应输入通道数"""
        # 如果没有提供设备，使用输入张量的设备或默认值
        if device is None:
            # 尝试从dummy_param获取设备
            try:
                device = self.dummy_param.device
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        return nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 4, kernel_size=1),  # 固定输出4个通道用于熵计算
            nn.Softmax(dim=1)  # 用于计算熵
        ).to(device)
        
    def _compute_entropy(self, probs):
        """计算香农熵 H(X) = -sum(p*log(p))"""
        # 避免数值问题
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=1, keepdim=True)
        return entropy
        
    def forward(self, x, prev_entropy=None, prev_mask=None):
        """
        Args:
            x: 输入特征 [B,C,H,W]
            prev_entropy: 之前层的累积熵 [B,1,H,W]
            prev_mask: 之前层的活跃掩码 [B,1,H,W]
        """
        batch_size, channels, height, width = x.shape
        
        # 动态创建熵估计器以匹配输入通道
        if self.entropy_estimator is None or self.entropy_estimator[0].in_channels != channels:
            self.entropy_estimator = self._build_entropy_estimator(channels, device=x.device)
        
        # 初始化状态（如果是第一次调用）
        if prev_entropy is None:
            cumul_entropy = torch.zeros((batch_size, 1, height, width), device=x.device)
        else:
            # 确保尺寸匹配
            if prev_entropy.shape[2:] != x.shape[2:]:
                cumul_entropy = F.interpolate(prev_entropy, size=(height, width), mode='nearest')
            else:
                cumul_entropy = prev_entropy
            
        if prev_mask is None:
            active_mask = torch.ones((batch_size, 1, height, width), device=x.device, dtype=torch.bool)
        else:
            # 确保尺寸匹配
            if prev_mask.shape[2:] != x.shape[2:]:
                active_mask = F.interpolate(prev_mask.float(), size=(height, width), mode='nearest').bool()
            else:
                active_mask = prev_mask
        
        # 计算当前层熵
        probs = self.entropy_estimator(x)
        current_entropy = self._compute_entropy(probs)
        
        # 更新累积熵
        cumul_entropy = cumul_entropy + current_entropy
        
        # 更新活跃掩码 - 停止熵超过阈值的位置
        new_active_mask = (cumul_entropy < self.threshold) & active_mask
        
        # 应用掩码进行选择性计算
        if not torch.all(new_active_mask == active_mask):
            # 有区域被停止，应用掩码
            masked_x = x * active_mask.float()
            # 保留未停止区域的特征，冻结已停止区域（detach防止梯度流）
            x = x * new_active_mask.float() + masked_x.detach() * (~new_active_mask).float()
        
        return x, new_active_mask, cumul_entropy

class BoundaryEnhancedSkipConnection(nn.Module):
    """边界感知的跳跃连接增强器"""
    def __init__(self, channels, reduction=4, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
        # 边界检测分支 - 确保输入通道数正确
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征融合 - 接收两组特征的通道总数
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, skip_features, main_features):
        """
        Args:
            skip_features: 跳跃连接特征 [B,C,H,W]
            main_features: 主路径特征 [B,C,H,W]
        """
        # 检测边界
        boundary_map = self.boundary_detector(skip_features)
        
        # 增强跳跃连接特征
        enhanced_skip = skip_features * (1.0 + self.alpha * boundary_map)
        
        # 调整主特征形状以匹配跳跃连接（如果需要）
        if main_features.shape[2:] != skip_features.shape[2:]:
            main_features = F.interpolate(
                main_features, 
                size=skip_features.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # 确保通道数匹配后再拼接
        if main_features.shape[1] != skip_features.shape[1]:
            # 如果通道数不匹配，则通过1x1卷积调整
            channel_adapter = nn.Conv2d(
                main_features.shape[1], skip_features.shape[1], kernel_size=1
            ).to(device=skip_features.device)
            main_features = channel_adapter(main_features)
            
        # 融合特征
        combined = torch.cat([enhanced_skip, main_features], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features, boundary_map

class MultiscaleAttention(nn.Module):
    """轻量级多尺度自注意力，用于TAD尺度评估"""
    def __init__(self, in_channels=16):
        super().__init__()
        # 添加一个假参数以避免没有parameters()的问题
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=1),
                nn.InstanceNorm2d(8),
                nn.ReLU()
            ) for _ in range(4)  # 4个不同尺度
        ])
        
        # 尺度评估网络
        self.scale_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8*4, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # TAD尺寸预测器 - 预测min_size, avg_size, max_size
        self.tad_size_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8*4, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),  # 3个尺寸参数
            nn.Sigmoid()  # 归一化到0-1范围
        )
        
    def forward(self, x):
        """
        输入: [B,C,H,W] 特征图
        输出: 多尺度汇总特征, 各尺度权重, TAD尺寸参数
        """
        B, C, H, W = x.shape
        
        # 保护机制，确保输入通道数匹配
        if C != self.scale_heads[0][0].in_channels:
            # 创建通道适配器
            adapter = nn.Conv2d(C, self.scale_heads[0][0].in_channels, kernel_size=1).to(x.device)
            x = adapter(x)
            C = self.scale_heads[0][0].in_channels
        
        # 动态计算尺度大小，确保至少为4x4
        min_size = min(H, W)
        scale_sizes = [max(int(min_size * s), 4) for s in [0.1, 0.25, 0.5, 0.75]]
        
        # 多尺度特征处理
        scale_features = []
        for i, head in enumerate(self.scale_heads):
            # 下采样到当前尺度
            scale_size = scale_sizes[i]
            scale_feat = F.adaptive_avg_pool2d(x, (scale_size, scale_size))
            
            # 特征处理
            scale_feat = head(scale_feat)
            
            # 上采样回原始尺寸
            scale_feat = F.interpolate(scale_feat, size=(H, W), 
                                    mode='bilinear', align_corners=False)
            scale_features.append(scale_feat)
            
        # 拼接多尺度特征
        multi_scale = torch.cat(scale_features, dim=1)  # [B,32,H,W]
        
        # 评估各尺度权重
        scale_weights = self.scale_evaluator(multi_scale)  # [B,4,1,1]
        
        # 生成加权融合结果
        weighted_feat = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            weighted_feat += feat * scale_weights[:, i:i+1]
        
        # 预测TAD尺寸参数并缩放到合理范围
        # [0,1]范围的输出缩放到实际像素大小
        raw_size_params = self.tad_size_predictor(multi_scale)  # [B,3,1,1]
        # 将归一化尺寸转换为实际像素尺寸 (最小5，最大为图像大小的40%)
        min_tad_size = 5 + raw_size_params[:, 0:1] * 10  # 最小TAD尺寸：5-15
        avg_tad_size = 15 + raw_size_params[:, 1:2] * min(H, W) * 0.3  # 平均TAD尺寸：合理中等大小
        max_tad_size = avg_tad_size + raw_size_params[:, 2:3] * min(H, W) * 0.1  # 最大TAD尺寸
        
        # 拼接尺寸参数
        tad_size_params = torch.cat([min_tad_size, avg_tad_size, max_tad_size], dim=1)  # [B,3,1,1]
            
        return weighted_feat, scale_weights, tad_size_params

class SimplifiedUNet(nn.Module):
    """添加多尺度反思机制的U-Net，修复维度问题"""
    def __init__(self, input_channels=1, output_channels=1, adv_mode=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 编码器改为使用注意力引导卷积
        # 第一层编码
        self.enc1 = nn.ModuleList([
            AttentionGuidedConv(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            AttentionGuidedConv(16, 16, kernel_size=3, padding=1)
        ])
        
        # 第二层编码
        self.enc2 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            AttentionGuidedConv(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            AttentionGuidedConv(32, 32, kernel_size=3, padding=1)
        ])
        
        # 第三层编码
        self.enc3 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            AttentionGuidedConv(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionGuidedConv(64, 64, kernel_size=3, padding=1)
        ])
        
        # 解码器 (上采样路径)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 边界增强跳跃连接
        self.skip_conn2 = BoundaryEnhancedSkipConnection(32, reduction=4)
        self.skip_conn1 = BoundaryEnhancedSkipConnection(16, reduction=2)
        
        # 早停机制 - 动态适应输入通道数
        self.early_stopping = EntropyBasedEarlyStopping(threshold=0.85)
        
        # 多尺度反思机制
        self.multiscale_reflection = MultiscaleAttention(in_channels=16)
        
        # 修正TAD尺度选择网络
        self.tad_scale_selector = nn.Sequential(
            nn.Conv2d(16+8, 16, kernel_size=3, padding=1),  # 16(原特征) + 8(多尺度特征)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # TAD边界增强层
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 新增对抗训练专用层
        self.adv_mode = adv_mode
        if adv_mode:
            self.feature_perturb = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.InstanceNorm2d(16),
                nn.ReLU()
            )
        
        # 最终输出调整
        self.final_sigmoid = nn.Sigmoid()
        
        # 保存注意力图和掩码用于可视化和调试
        self.attention_maps = {}
        self.boundary_maps = {}
        self.active_masks = None
        self.cumul_entropy = None
        
    def forward(self, x):
        """前向传播，确保所有维度正确匹配"""
        B, C, H, W = x.shape
        self.attention_maps = {}
        self.boundary_maps = {}
        
        # 初始化早停状态
        active_mask = torch.ones((B, 1, H, W), device=x.device, dtype=torch.bool)
        cumul_entropy = torch.zeros((B, 1, H, W), device=x.device)
        
        # ===== 编码器路径 =====
        # 第一层编码 - 注意力引导卷积
        attention_maps_1 = []
        
        # 第一个Conv层
        enc1, att_map = self.enc1[0](x)
        attention_maps_1.append(att_map)
        
        # 批归一化和激活
        enc1 = self.enc1[1](enc1)
        enc1 = self.enc1[2](enc1)
        
        # 第二个Conv层
        enc1, att_map = self.enc1[3](enc1)
        attention_maps_1.append(att_map)
        
        self.attention_maps['enc1'] = attention_maps_1
        
        # 应用早停机制
        enc1, active_mask, cumul_entropy = self.early_stopping(enc1, cumul_entropy, active_mask)
        
        # 第二层编码
        # 池化操作
        x_pool = self.enc2[0](enc1)
        
        # 注意力引导卷积
        attention_maps_2 = []
        
        # 第一个Conv层
        enc2, att_map = self.enc2[1](x_pool)
        attention_maps_2.append(att_map)
        
        # 批归一化和激活
        enc2 = self.enc2[2](enc2)
        enc2 = self.enc2[3](enc2)
        
        # 第二个Conv层
        enc2, att_map = self.enc2[4](enc2)
        attention_maps_2.append(att_map)
        
        self.attention_maps['enc2'] = attention_maps_2
        
        # 继续应用早停机制 - 调整尺寸匹配
        _, enc2_h, enc2_w = enc2.shape[1:]
        enc2_entropy = F.interpolate(cumul_entropy, size=(enc2_h, enc2_w), mode='nearest')
        enc2_mask = F.interpolate(active_mask.float(), size=(enc2_h, enc2_w), mode='nearest').bool()
        enc2, active_mask_2, cumul_entropy_2 = self.early_stopping(enc2, enc2_entropy, enc2_mask)
        
        # 第三层编码
        # 池化操作
        x_pool = self.enc3[0](enc2)
        
        # 注意力引导卷积
        attention_maps_3 = []
        
        # 第一个Conv层
        enc3, att_map = self.enc3[1](x_pool)
        attention_maps_3.append(att_map)
        
        # 批归一化和激活
        enc3 = self.enc3[2](enc3)
        enc3 = self.enc3[3](enc3)
        
        # 第二个Conv层
        enc3, att_map = self.enc3[4](enc3)
        attention_maps_3.append(att_map)
        
        self.attention_maps['enc3'] = attention_maps_3
        
        # ===== 解码器路径与边界增强跳跃连接 =====
        # 上采样enc3到enc2尺寸
        dec2_up = F.interpolate(enc3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        
        # 应用边界增强跳跃连接
        enhanced_skip2, boundary_map2 = self.skip_conn2(enc2, dec2_up)
        self.boundary_maps['skip2'] = boundary_map2
        
        # 继续解码器处理 - 修复可能的通道数不匹配
        # 确保拼接前通道数匹配预期
        expected_cat_channels = self.dec2[0].in_channels
        current_cat_channels = dec2_up.shape[1] + enhanced_skip2.shape[1]
        
        if current_cat_channels != expected_cat_channels:
            # 使用1x1卷积调整通道数
            channel_adapter = nn.Conv2d(
                current_cat_channels, expected_cat_channels, kernel_size=1
            ).to(device=dec2_up.device)
            dec2_cat = channel_adapter(torch.cat([dec2_up, enhanced_skip2], dim=1))
        else:
            dec2_cat = torch.cat([dec2_up, enhanced_skip2], dim=1)
            
        dec2 = self.dec2(dec2_cat)
        
        # 上采样dec2到enc1尺寸
        dec1_up = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        
        # 应用边界增强跳跃连接
        enhanced_skip1, boundary_map1 = self.skip_conn1(enc1, dec1_up)
        self.boundary_maps['skip1'] = boundary_map1
        
        # 继续解码器处理
        expected_cat_channels = self.dec1[0].in_channels
        current_cat_channels = dec1_up.shape[1] + enhanced_skip1.shape[1]
        
        if current_cat_channels != expected_cat_channels:
            # 使用1x1卷积调整通道数
            channel_adapter = nn.Conv2d(
                current_cat_channels, expected_cat_channels, kernel_size=1
            ).to(device=dec1_up.device)
            dec1_cat = channel_adapter(torch.cat([dec1_up, enhanced_skip1], dim=1))
        else:
            dec1_cat = torch.cat([dec1_up, enhanced_skip1], dim=1)
            
        dec1 = self.dec1(dec1_cat)
        
        # ----- 多尺度反思机制 -----
        # 确保输入通道数匹配
        multiscale_feat, scale_weights, tad_size_params = self.multiscale_reflection(dec1)
        
        # 融合多尺度特征与原始特征 - 确保通道数匹配
        enhanced_feat = torch.cat([dec1, multiscale_feat], dim=1)  # [B,24,H,W]
        tad_feat = self.tad_scale_selector(enhanced_feat)  # [B,16,H,W]
        
        # 边界增强和分割输出
        edge_map = self.edge_enhancement(tad_feat)  # [B,1,H,W]
        final_out = self.final(tad_feat)  # [B,1,H,W]
        
        # TAD大小调整 - 简化逻辑，减少运行时错误风险
        # 获取TAD尺寸参数
        min_size = tad_size_params[:, 0:1]  # [B,1,1,1]
        avg_size = tad_size_params[:, 1:2]  # [B,1,1,1]
        max_size = tad_size_params[:, 2:3]  # [B,1,1,1]
        
        # 融合边界信息
        enhanced_out = final_out * 0.7 + edge_map * 0.3
        enhanced_out = torch.clamp(enhanced_out, 0.0, 1.0)  # 确保输出在[0,1]范围内
        
        # 保存尺度信息供外部使用
        self.scale_info = {
            'weights': scale_weights,
            'optimal_idx': torch.argmax(scale_weights, dim=1),
            'tad_sizes': tad_size_params
        }
        
        # 保存早停信息用于可视化
        self.active_masks = active_mask
        self.cumul_entropy = cumul_entropy
        
        return enhanced_out
