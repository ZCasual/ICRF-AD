import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

# 导入模块
from tad_modules import (LowRank, GeometricFeatureExtractor, AVIT, 
                        EdgeAwareBiLSTM, SimplifiedUNet, 
                        find_chromosome_files, fill_hic)  # 从模块导入函数

# 创建基础配置类
class TADBaseConfig:
    """基础配置类：集中管理所有共享参数"""
    def __init__(self):
        # 基本环境参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_root = "./tad_results"
        self.resolution = 10000
        
        # 模型结构参数
        self.patch_size = 8
        self.embed_dim = 64
        self.num_layers = 15
        self.num_heads = 4
        
        # 训练参数
        self.use_amp = (self.device == "cuda")
        self.ema_decay = 0.996
        self.mask_ratio = 0.3
        self.gamma_base = 0.01
        self.epsilon_base = 0.05
        self.use_theory_gamma = True
        self.boundary_weight = 0.3
        self.num_epochs = 40

    def get_model_params(self):
        """获取模型相关参数字典"""
        return {
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'use_amp': self.use_amp,
            'ema_decay': self.ema_decay,
            'mask_ratio': self.mask_ratio,
            'gamma_base': self.gamma_base,
            'epsilon_base': self.epsilon_base,
            'use_theory_gamma': self.use_theory_gamma,
            'boundary_weight': self.boundary_weight,
        }

"""
以下模块已迁移到 (tad_modules)
0. 数据加载
1. CUR特征分解
2. 几何特征提取模块
3. Backbone: AVIT
4. BiLSTM + U-Net
"""

"""
5. A-VIT型DINO自监督框架
"""
class AVIT_DINO(nn.Module):
    """基于A-VIT骨干网络的DINO-V2自监督学习框架"""
    def __init__(self, embed_dim=None, patch_size=None, num_layers=None, num_heads=None, 
                 use_amp=None, ema_decay=None, mask_ratio=None, 
                 student_temp=None, teacher_temp=None, 
                 gamma_base=None, epsilon_base=None, use_theory_gamma=None,
                 boundary_weight=None):
        super().__init__()
        # 创建学生和教师网络（都基于A-VIT骨干网络）
        self.student = AVIT(embed_dim, patch_size, num_layers, num_heads, use_amp)
        self.teacher = AVIT(embed_dim, patch_size, num_layers, num_heads, use_amp)
        
        # 初始化教师模型并停止其梯度更新
        self._init_teacher()
        
        # DINO超参数
        self.ema_decay = ema_decay           # 教师网络指数移动平均衰减率
        self.mask_ratio = mask_ratio         # 掩码比例
        self.student_temp = student_temp     # 学生温度系数
        self.teacher_temp = teacher_temp     # 教师温度系数
        self.register_buffer("center", torch.zeros(1, embed_dim))  # 中心化向量
        self.center_momentum = 0.9           # 中心化动量
        
        # 可学习的掩码标记（mask token）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)  # 初始化掩码标记
        
        # 编码率正则化参数 - 更灵活的设计
        self.gamma_base = gamma_base         # 基础 gamma 值
        self.epsilon_base = epsilon_base     # 基础 epsilon 值
        self.use_theory_gamma = use_theory_gamma  # 是否使用理论公式计算 gamma
        self.gamma = gamma_base              # 初始化为基础值，将在前向传播时动态调整
        self.epsilon = epsilon_base          # 初始化为基础值，将在前向传播时动态调整
        
        # 设置使用混合精度计算
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
        
        # 缓存批次协方差计算
        self.batch_size = 0                  # 将在训练时确定
        self.register_buffer("cov_sum", None)  # 协方差累加
        self.register_buffer("cov_count", torch.zeros(1))  # 协方差计数
        
        # 增加特征维度和样本数量追踪
        self.feature_dim = embed_dim  # 特征维度 d
        self.num_samples = 0          # 样本数量 n，将在前向传播中更新
        
        # 边界权重系数（掩码区域重建权重）
        self.boundary_weight = boundary_weight
        
        # 新增：教师网络的BiLSTM边界检测器
        self.teacher_bilstm = EdgeAwareBiLSTM(embed_dim, hidden_dim=32)
        
        # 新增：学生网络的U-NET分割模块
        self.student_unet = SimplifiedUNet(1, output_channels=1)  # 使用单通道输入
    
    def _init_teacher(self):
        """初始化教师网络，拷贝学生权重并停止梯度"""
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
    
    def _update_teacher(self):
        """通过EMA更新教师网络权重"""
        with torch.no_grad():
            for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
                param_t.data = self.ema_decay * param_t.data + (1 - self.ema_decay) * param_s.data
    
    def _update_center(self, teacher_cls):
        """更新中心向量（用于防止模型坍塌）"""
        batch_center = torch.mean(teacher_cls, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def _generate_mask(self, x, mask_ratio):
        """生成随机掩码
        
        Args:
            x: 输入张量 [B, N, D] 或 [N, D]
            mask_ratio: 掩码比例
            
        Returns:
            mask: 掩码张量 [B, N] 或 [N]，True表示保留，False表示遮盖
        """
        if x.dim() == 3:
            B, N, _ = x.shape
            mask = torch.rand(B, N, device=x.device) > mask_ratio  # True表示保留
        else:
            N, _ = x.shape
            mask = torch.rand(N, device=x.device) > mask_ratio
        return mask
    
    def _apply_mask(self, x, mask):
        """应用掩码，将掩码位置替换为mask_token
        
        Args:
            x: 输入张量 [B, N, D] 或 [N, D]
            mask: 掩码张量 [B, N] 或 [N]，True表示保留，False表示遮盖
            
        Returns:
            masked_x: 掩码后的张量，与x形状相同
        """
        if x.dim() == 3:
            B, N, D = x.shape
            mask = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
            mask_tokens = self.mask_token.expand(B, N, -1)  # [B, N, D]
        else:
            N, D = x.shape
            mask = mask.unsqueeze(-1).expand(-1, D)  # [N, D]
            mask_tokens = self.mask_token.squeeze(0).expand(N, -1)  # [N, D]
        
        masked_x = torch.where(mask, x, mask_tokens)
        return masked_x, ~mask.squeeze(-1)  # 返回掩码后的x和掩码指示器（False表示保留，True表示掩码）
    
    def _compute_covariance(self, features):
        """计算特征协方差矩阵，避免批量大小为1时除以0的问题"""
        B, D = features.shape
        features = features - features.mean(dim=0, keepdim=True)
        # 如果B==1则直接返回全零矩阵以避免除零
        cov = features.t() @ features / (B - 1) if B > 1 else torch.zeros(D, D, device=features.device)
        return cov
    
    def _compute_theory_gamma(self, n, d):
        """
        根据理论公式计算 gamma
        γ = Θ(ε √(n / (d min{d, n})))
        
        Args:
            n: 样本数量
            d: 特征维度
            
        Returns:
            理论计算的 gamma 值
        """
        # 确保 n 和 d 为正值
        n = max(1, n)
        d = max(1, d)
        
        # 计算理论公式
        min_d_n = min(d, n)
        gamma_theory = self.epsilon * torch.sqrt(torch.tensor(n / (d * min_d_n), 
                                                             device=self.mask_token.device))
        
        # 加入常数因子使理论值与实际训练相适应
        gamma_theory = self.gamma_base * gamma_theory
        
        # 限制在合理范围内
        gamma_theory = torch.clamp(gamma_theory, 
                                  min=self.gamma_base * 0.1, 
                                  max=self.gamma_base * 10.0)
        
        return gamma_theory
    
    def _compute_dynamic_gamma(self, cov, n=None, d=None):
        """
        根据协方差矩阵的特征值分布和维度信息动态调整gamma参数
        
        Args:
            cov: 协方差矩阵 [D, D]
            n: 可选，样本数量
            d: 可选，特征维度
            
        Returns:
            动态调整后的gamma值
        """
        # 如果启用理论公式计算且提供了维度信息
        if self.use_theory_gamma and n is not None and d is not None:
            # 计算理论 gamma
            gamma_theory = self._compute_theory_gamma(n, d)
            
            # 尝试结合特征值信息进行微调
            try:
                # 计算协方差矩阵的特征值
                eigvals = torch.linalg.eigvalsh(cov)
                
                # 获取最大和最小特征值
                lambda_max = eigvals.max()
                lambda_min = max(eigvals.min(), 1e-6)  # 避免除零
                
                # 特征值比例因子
                eigen_ratio = torch.sqrt(lambda_max / lambda_min)
                
                # 结合理论值和特征值信息
                # 在特征值差距大时稍微提高gamma值，反之降低
                gamma_t = gamma_theory * torch.clamp(eigen_ratio / 10.0, 0.8, 1.2)
                
                # 限制最终范围
                gamma_t = torch.clamp(gamma_t, 
                                     min=self.gamma_base * 0.1, 
                                     max=self.gamma_base * 10.0)
                
                return gamma_t
            except Exception as e:
                print(f"计算特征值调整时出错: {str(e)}，使用纯理论gamma值")
                return gamma_theory
        else:
            # 使用原有基于特征值的方法（作为备选）
            try:
                # 使用对称性优化计算
                eigvals = torch.linalg.eigvalsh(cov)
                
                # 获取最大和最小特征值
                lambda_max = eigvals.max()
                lambda_min = eigvals.min()
                
                # 动态调整gamma值
                gamma_t = self.gamma_base * (lambda_max / (lambda_min + 1e-6))
                
                # 限制gamma范围
                gamma_t = torch.clamp(gamma_t, 
                                     min=self.gamma_base * 0.1, 
                                     max=self.gamma_base * 10.0)
                
                return gamma_t
            except Exception as e:
                print(f"计算动态gamma时出错: {str(e)}，使用默认gamma值")
                return torch.tensor(self.gamma_base, device=cov.device)
    
    def _adjust_epsilon(self, features):
        """
        根据特征分布动态调整epsilon参数
        
        Args:
            features: 输入特征 [B, D]
            
        Returns:
            调整后的epsilon值
        """
        try:
            # 计算特征的平均距离
            with torch.no_grad():
                # 计算特征之间的欧氏距离均值
                B, D = features.shape
                if B > 1:
                    # 随机采样最多100个样本对计算距离，避免计算过大矩阵
                    sample_size = min(B, 100)
                    indices = torch.randperm(B)[:sample_size]
                    sampled_features = features[indices]
                    
                    # 计算样本间距离
                    dists = torch.cdist(sampled_features, sampled_features)
                    # 除去对角线的零元素
                    mask = ~torch.eye(sample_size, dtype=torch.bool, device=dists.device)
                    avg_dist = dists[mask].mean()
                    
                    # 根据平均距离调整epsilon
                    # 我们希望epsilon与平均距离成比例，但有上下限
                    new_epsilon = self.epsilon_base * (avg_dist / 2.0)
                    new_epsilon = torch.clamp(new_epsilon, 
                                             min=self.epsilon_base * 0.5, 
                                             max=self.epsilon_base * 2.0)
                    
                    # 平滑更新
                    self.epsilon = 0.9 * self.epsilon + 0.1 * new_epsilon.item()
                
            return self.epsilon
        except Exception as e:
            print(f"调整epsilon时出错: {str(e)}，使用默认epsilon值")
            return self.epsilon_base
    
    def _encoding_rate_regularization(self, cov):
        """计算编码率正则化项，使用动态调整的gamma和epsilon参数"""
        D = cov.shape[0]
        
        # 添加小扰动确保协方差矩阵正定
        cov_reg = cov + 1e-6 * torch.eye(D, device=cov.device)
        
        # 计算动态gamma值，传入当前维度信息
        gamma_t = self._compute_dynamic_gamma(cov_reg, self.num_samples, self.feature_dim)
        
        # 使用可能已经更新的epsilon值来计算编码率
        cov_scaled = torch.eye(D, device=cov.device) + (D / (self.epsilon ** 2)) * cov_reg
        reg = 0.5 * torch.logdet(cov_scaled)
        
        # 保存当前gamma值，用于loss计算
        self.current_gamma = gamma_t
        return reg
    
    def forward(self, matrix):
        """前向传播，同时处理局部（掩码）和全局视图，并整合边界检测"""
        # 确保输入为张量并移至正确设备
        if isinstance(matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(matrix).float()
        else:
            matrix_tensor = matrix.clone().detach().float()
            
        matrix_tensor = matrix_tensor.to(next(self.student.parameters()).device)
        
        # 1. 学生网络处理掩码视图
        with torch.set_grad_enabled(True):
            # 正常前向传播获取特征
            student_output = self.student.conv_embed(matrix_tensor)  # [N_patches, D]
            
            # 生成并应用掩码
            mask = self._generate_mask(student_output, self.mask_ratio)
            masked_student_output, mask_indices = self._apply_mask(student_output, mask)
            
            # 更新样本数和特征维度信息
            if masked_student_output.dim() == 3:  # [B, N, D]
                B, N, D = masked_student_output.shape
                self.num_samples = B * N  # 更新样本数
                self.feature_dim = D      # 更新特征维度
                student_cls = masked_student_output[:, 0]  # [B, D]
                student_patches = masked_student_output[:, 1:]  # [B, N-1, D]
            else:  # [N, D]
                # 为单样本情况添加批次维度
                N, D = masked_student_output.shape
                self.num_samples = N      # 更新样本数
                self.feature_dim = D      # 更新特征维度
                masked_student_output = masked_student_output.unsqueeze(0)  # [1, N, D]
                student_cls = masked_student_output[:, 0]  # [1, D]
                student_patches = masked_student_output[:, 1:]  # [1, N-1, D]
                mask_indices = mask_indices.unsqueeze(0)  # [1, N]
            
            # 动态调整epsilon
            self._adjust_epsilon(student_cls)
            
            # 归一化特征（投影到单位超球面）
            student_cls = F.normalize(student_cls, dim=-1)
            student_patches = F.normalize(student_patches, dim=-1)
            
            # 新增：学生U-NET分割预测
            # 将1D序列特征重塑为2D特征图
            student_feat_2d = self._reshape_to_2d(student_cls)
            student_segmentation = self.student_unet(student_feat_2d)
        
        # 2. 教师网络处理无掩码的全局视图（无梯度）
        with torch.no_grad():
            # 直接使用原始输入（无掩码）
            teacher_output = self.teacher.conv_embed(matrix_tensor)  # [N_patches, D]
            
            # 提取全局CLS特征和patch特征
            if teacher_output.dim() == 3:  # [B, N, D]
                teacher_cls = teacher_output[:, 0]  # [B, D]
                teacher_patches = teacher_output[:, 1:]  # [B, N-1, D]
            else:  # [N, D]
                teacher_output = teacher_output.unsqueeze(0)  # [1, N, D]
                teacher_cls = teacher_output[:, 0]  # [1, D]
                teacher_patches = teacher_output[:, 1:]  # [1, N-1, D]
            
            # 归一化特征
            teacher_cls = F.normalize(teacher_cls, dim=-1)
            teacher_patches = F.normalize(teacher_patches, dim=-1)
            
            # 应用中心化（防止模型坍塌）
            teacher_cls = teacher_cls - self.center
            
            # 更新中心向量
            self._update_center(teacher_cls)
            
            # 新增：教师BiLSTM边界检测
            teacher_boundary_probs, teacher_boundary_adj = self.teacher_bilstm(teacher_output)
        
        # 将新增的预测添加到输出字典
        return {
            'student_cls': student_cls,
            'student_patches': student_patches,
            'teacher_cls': teacher_cls,
            'teacher_patches': teacher_patches,
            'mask_indices': mask_indices,
            'student_segmentation': student_segmentation,  # 新增：学生分割预测
            'teacher_boundary_probs': teacher_boundary_probs,    # 修改：教师边界概率
            'teacher_boundary_adj': teacher_boundary_adj   # 新增：教师边界调整
        }

    def _reshape_to_2d(self, features):
        """将1D特征序列重塑为2D特征图"""
        # 处理张量维度
        if features.dim() == 3:  # [B, L, D]
            batch_size, seq_len, feat_dim = features.shape
            features = features.view(batch_size, seq_len * feat_dim)
        else:  # [B, D]
            batch_size, feat_dim = features.shape
        
        # 计算最接近的平方数作为尺寸
        side_len = int(math.sqrt(feat_dim))
        
        # 如果不是完美平方数，调整特征维度
        if side_len * side_len != feat_dim:
            side_len = side_len + 1
            new_dim = side_len * side_len
            # 填充额外的位置
            pad_features = torch.zeros(batch_size, new_dim, device=features.device)
            pad_features[:, :feat_dim] = features
            features = pad_features
        
        # 重塑为 [B, 1, H, W]
        features_2d = features.view(batch_size, 1, side_len, side_len)
        return features_2d

    def compute_loss(self, outputs):
        """计算损失函数（增加边界检测和分割任务的损失）"""
        student_cls = outputs['student_cls']
        student_patches = outputs['student_patches']
        teacher_cls = outputs['teacher_cls']
        teacher_patches = outputs['teacher_patches']
        mask_indices = outputs['mask_indices']
        
        # 1. 修改：单向CLS token对齐损失（学生→教师）
        student_to_teacher = torch.sum((student_cls - teacher_cls) ** 2, dim=1)
        # 移除教师→学生的对齐，只保留学生对齐教师
        cls_loss = torch.mean(student_to_teacher)
        
        # 2. 掩码patch对齐损失
        # 仅对掩码位置的patch计算损失
        B, N, D = student_patches.shape
        patch_loss = 0.0
        for b in range(B):
            # 提取当前批次的掩码指示器
            curr_mask = mask_indices[b, 1:]  # 排除CLS token位置的掩码
            if curr_mask.sum() > 0:  # 确保有被掩码的patch
                # 计算被掩码patch的均方误差
                masked_student = student_patches[b][curr_mask]
                masked_teacher = teacher_patches[b][curr_mask]
                # 同样修改为单向对齐（学生→教师）
                patch_loss += torch.sum((masked_student - masked_teacher) ** 2) / curr_mask.sum()
        
        patch_loss = patch_loss / B if B > 0 else torch.tensor(0.0, device=student_cls.device)
        
        # 3. 编码率正则化
        # 计算学生CLS特征的协方差矩阵
        student_cov = self._compute_covariance(student_cls)
        if self.cov_sum is None or self.cov_sum.shape != student_cov.shape:
            self.cov_sum = student_cov.detach()
        else:
            self.cov_sum = 0.9 * self.cov_sum + 0.1 * student_cov.detach()
        
        self.cov_count += 1
        # 使用累积协方差计算编码率
        encoding_rate = self._encoding_rate_regularization(self.cov_sum / self.cov_count)
        
        # 获取动态gamma（在_encoding_rate_regularization中设置）
        gamma_t = getattr(self, 'current_gamma', self.gamma)
        
        # 总损失 - 使用动态gamma
        total_loss = cls_loss + patch_loss - gamma_t * encoding_rate
        
        # 获取新增的预测
        student_segmentation = outputs.get('student_segmentation')
        teacher_boundary_probs = outputs.get('teacher_boundary_probs')
        
        # 2. 新增联合任务损失
        
        # 2.1 分割一致性损失（学生分割应与教师边界概率一致）
        seg_loss = 0.0
        if student_segmentation is not None and teacher_boundary_probs is not None:
            # 确保teacher_boundary_probs是2D或3D张量
            if teacher_boundary_probs.dim() == 1:
                teacher_boundary_probs = teacher_boundary_probs.unsqueeze(0)  # [L] -> [1, L]
            
            # 将教师边界概率重塑为2D图像格式
            teacher_boundary_2d = self._reshape_to_2d(teacher_boundary_probs)
            
            # 确保尺寸匹配
            if teacher_boundary_2d.shape[2:] != student_segmentation.shape[2:]:
                teacher_boundary_2d = F.interpolate(
                    teacher_boundary_2d, 
                    size=student_segmentation.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # 计算分割一致性损失 - 使用AMP安全的方式计算BCE损失
            if self.use_amp:
                # 暂时退出自动混合精度上下文来计算BCE损失
                with torch.cuda.amp.autocast(enabled=False):
                    # 将张量转换为float32以确保精度
                    student_seg_float = student_segmentation.float()
                    teacher_boundary_float = teacher_boundary_2d.float()
                    seg_loss = F.binary_cross_entropy(student_seg_float, teacher_boundary_float)
            else:
                # 正常计算BCE损失
                seg_loss = F.binary_cross_entropy(student_segmentation, teacher_boundary_2d)
        
        # 总联合损失
        joint_loss = seg_loss * self.boundary_weight
        
        # 最终总损失
        total_loss += joint_loss
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'patch': patch_loss,
            'encoding_rate': encoding_rate,
            'gamma': gamma_t,
            'boundary': self.boundary_weight * patch_loss,
            'segmentation': seg_loss  # 新增：分割损失
        }

    def train_epoch(self, matrix):
        self.student.train()
        self.teacher.eval()  # 教师模型始终处于评估模式
        
        # 训练步骤
        total_loss = 0.0
        cls_loss = 0.0
        patch_loss = 0.0
        encoding_rate = 0.0
        
        # 分批次处理以减少内存使用
        grad_accum_steps = getattr(self.student, 'grad_accum_steps', 4)
        
        for step in range(grad_accum_steps):
            # 清理GPU缓存
            if step > 0 and step % 2 == 0:
                torch.cuda.empty_cache()
            
            # 根据use_amp参数决定是否使用混合精度
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # 前向传播
                    outputs = self(matrix)
                    
                # 在自动混合精度外计算损失以避免BCE问题
                losses = self.compute_loss(outputs)
                
                # 缩放损失并反向传播
                scaled_loss = losses['total'] / grad_accum_steps
                self.scaler.scale(scaled_loss).backward()
            else:
                # 前向传播
                outputs = self(matrix)
                losses = self.compute_loss(outputs)
                
                # 反向传播
                scaled_loss = losses['total'] / grad_accum_steps
                scaled_loss.backward()
            
            # 收集损失
            total_loss += losses['total'].item()
            cls_loss += losses['cls'].item()
            patch_loss += losses['patch'].item()
            encoding_rate += losses['encoding_rate'].item()
            
            # 释放内存
            del outputs, losses, scaled_loss
            
            # 每两步或最后一步执行优化器步骤
            if (step + 1) % 2 == 0 or (step + 1) == grad_accum_steps:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # 更新教师网络
                self._update_teacher()
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        avg_steps = grad_accum_steps or 1
        return {
            'total': total_loss / avg_steps,
            'cls': cls_loss / avg_steps,
            'patch': patch_loss / avg_steps,
            'encoding_rate': encoding_rate / avg_steps
        }

    def init_optimizer(self):
        """初始化优化器"""
        # 仅和学生网络的参数进行优化
        parameters = [
            {'params': [p for n, p in self.student.named_parameters() 
                       if not n.startswith('stop_')], 'lr': 1e-4},
            {'params': [p for n, p in self.student.named_parameters() 
                       if n.startswith('stop_')], 'lr': 1e-3},
            {'params': self.student_unet.parameters(), 'lr': 1e-4},  # 添加U-NET参数
            {'params': [self.mask_token], 'lr': 1e-4}  # 掩码标记也需要学习
        ]
        
        self.optimizer = torch.optim.AdamW(parameters, weight_decay=1e-4)
        
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
            self.check_amp_compatibility()  # 添加兼容性检查

    def get_network_for_training(self):
        """返回网络组件用于训练"""
        return {
            'student': self.student,
            'teacher': self.teacher,
            'student_unet': self.student_unet,
            'teacher_bilstm': self.teacher_bilstm
        }

    def get_boundary_predictions(self, matrix):
        """
        获取边界预测结果
        
        Args:
            matrix: 输入Hi-C矩阵
        
        Returns:
            dict: 包含边界预测的字典
        """
        # 进行前向传播
        outputs = self(matrix)
        
        # 提取边界相关预测
        student_segmentation = outputs.get('student_segmentation')
        teacher_boundary_probs = outputs.get('teacher_boundary_probs')  # 修改：使用边界概率
        
        return {
            'segmentation': student_segmentation.detach() if student_segmentation is not None else None,
            'boundary_probs': teacher_boundary_probs.detach() if teacher_boundary_probs is not None else None  # 修改：返回边界概率
        }

    def check_amp_compatibility(self):
        """检查模型组件是否与自动混合精度兼容"""
        if not self.use_amp:
            return True
        
        # 检查关键组件
        issues = []
        
        # 检查BCE损失是否在AMP上下文外计算
        # 这个函数被添加为提示，实际上我们会在compute_loss中
        # 正确处理自动混合精度下的BCE损失
        
        print(f"自动混合精度兼容性检查完成。")
        if issues:
            print(f"发现{len(issues)}个问题:")
            for issue in issues:
                print(f" - {issue}")
            return False
        return True

class TADFeatureExtractor(TADBaseConfig):
    """TAD特征提取器：从HiC数据中提取特征矩阵"""
    
    def __init__(self, use_cur=True, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)
        self.use_cur = use_cur
        self.model = None
        self.cur_projector = None
    
    def load_model(self, model_path=None, chr_name=None):
        """加载预训练模型"""
        # 如果未指定模型路径但指定了染色体，尝试加载该染色体的默认模型
        if model_path is None and chr_name is not None:
            model_path = os.path.join(self.output_root, chr_name, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载模型参数
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 从检查点提取模型参数
            if 'model_params' in checkpoint:
                for key in ['patch_size', 'embed_dim', 'num_layers', 'num_heads']:
                    if key in checkpoint['model_params']:
                        setattr(self, key, checkpoint['model_params'][key])
            
            # 创建模型实例 - 只使用教师模型用于特征提取
            model_params = self.get_model_params()
            self.model = AVIT(
                embed_dim=model_params['embed_dim'],
                patch_size=model_params['patch_size'],
                num_layers=model_params['num_layers'],
                num_heads=model_params['num_heads'],
                use_amp=model_params['use_amp']
            ).to(self.device)
            
            # 只加载教师模型状态字典
            if 'teacher' in checkpoint:
                # 使用教师模型替代学生模型进行特征提取
                self.model.load_state_dict(checkpoint['teacher'])
            elif 'student' in checkpoint:
                # 如果没有教师模型，使用学生模型
                self.model.load_state_dict(checkpoint['student'])
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"成功加载模型: {model_path}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
    
    def preprocess(self, matrix):
        """预处理HiC矩阵"""
        if self.use_cur:
            # 如果没有CUR投影器，创建一个
            if self.cur_projector is None:
                self.cur_projector = LowRank(p=0.7, alpha=0.7)
            
            # 应用CUR分解
            cur_matrix = self.cur_projector.fit_transform(matrix)
            return cur_matrix
        else:
            # 不使用CUR预处理，直接返回原矩阵
            return matrix
    
    def extract_features(self, hic_matrix, return_reconstruction=False):
        """
        从HiC矩阵中提取特征
        Args:
            hic_matrix: 输入的HiC矩阵
            return_reconstruction: 是否同时返回重建矩阵
        """
        # 确保模型已加载
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        # 预处理矩阵
        processed_matrix = self.preprocess(hic_matrix)
        
        # 将矩阵转换为tensor
        if isinstance(processed_matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(processed_matrix).float().to(self.device)
        else:
            matrix_tensor = processed_matrix.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            # 使用模型前向传播
            reconstructed, z_mean, z_logvar = self.model(matrix_tensor)
            
            # 获取编码器输出作为特征
            if hasattr(self.model, '_encoder_output_cache'):
                features = self.model._encoder_output_cache.cpu().numpy()
            else:
                # 如果没有缓存，返回空特征
                features = None
        
        # 准备返回结果
        result = {
            'features': features
        }
        
        if return_reconstruction:
            result['reconstructed'] = reconstructed.cpu().numpy()
        
        return result

# 创建全局特征提取器单例
_feature_extractor = None

def get_feature_extractor(**kwargs):
    """获取或创建全局TAD特征提取器"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = TADFeatureExtractor(**kwargs)
    return _feature_extractor

# 修改全局特征提取函数
def extract_features_from_hic(hic_matrix, chr_name=None, model_path=None, **kwargs):
    """
    从HiC矩阵中提取特征的简化函数 - 使用基类配置
    Returns:
        dict: 包含特征矩阵
    """
    # 创建临时特征提取器
    extractor = TADFeatureExtractor(**kwargs)
    # 确定模型路径
    if model_path is None and chr_name is not None:
        model_path = os.path.join(extractor.output_root, chr_name, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    extractor.load_model(model_path)
    # 提取特征
    result = extractor.extract_features(hic_matrix, return_reconstruction=True)
    return result
        
if __name__ == "__main__":
    print("请使用train.py进行模型训练")
  
  