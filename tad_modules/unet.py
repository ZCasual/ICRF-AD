import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .bayesian_layers import BayesianConvBlock
from .self_reflection import SelfReflectionModule, UncertaintyEstimator, DifferentiableCanny
from .internal_reward import TADBoundaryReward, BoundaryRefiner

class BayesianUNet(nn.Module):
    """结合贝叶斯层和自我反思机制的U-Net，增加了内部边界优化"""
    def __init__(self, input_channels=1, output_channels=1, adv_mode=False, mc_samples=5,
                 use_internal_optimization=True, optimization_iterations=2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.mc_samples = mc_samples
        self.adv_mode = adv_mode
        self.use_internal_optimization = use_internal_optimization
        
        # 初始化Canny边缘检测器
        self.canny_detector = DifferentiableCanny(
            low_threshold=0.1, 
            high_threshold=0.3,
            sigma=1.2
        )
        
        # 边缘融合权重（可学习）
        self.edge_weight = nn.Parameter(torch.tensor(0.4))
        self.canny_weight = nn.Parameter(torch.tensor(0.3))
        
        # 编码器 (下采样路径)
        self.enc1 = nn.Sequential(
            BayesianConvBlock(input_channels, 16),
            BayesianConvBlock(16, 16)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BayesianConvBlock(16, 32),
            BayesianConvBlock(32, 32)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BayesianConvBlock(32, 64),
            BayesianConvBlock(64, 64)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BayesianConvBlock(64, 128),
            BayesianConvBlock(128, 128)
        )
        
        # 自我反思模块 - 在各跳跃连接处添加
        self.reflect1 = SelfReflectionModule(16)
        self.reflect2 = SelfReflectionModule(32)
        self.reflect3 = SelfReflectionModule(64)
        
        # 解码器 (上采样路径) - 修复通道数匹配问题
        self.dec4 = nn.Sequential(
            BayesianConvBlock(128 + 64, 64),  # 128(enc4) + 64(enc3)
            BayesianConvBlock(64, 64)
        )
        
        self.dec3 = nn.Sequential(
            BayesianConvBlock(64 + 32, 32),  # 64(dec4) + 32(enc2)
            BayesianConvBlock(32, 32)
        )
        
        self.dec2 = nn.Sequential(
            BayesianConvBlock(32 + 16, 16),  # 32(dec3) + 16(enc1)
            BayesianConvBlock(16, 16)
        )
        
        # 这里是问题所在，应该只接收16通道输入，而不是32
        self.dec1 = nn.Sequential(
            BayesianConvBlock(16, 16),  # 修正：不再连接enc1，只使用dec2上采样结果
            BayesianConvBlock(16, 16)
        )
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(dropout_rate=0.2)
        
        # 边界增强层
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 新增：TAD边界优化模块
        if use_internal_optimization:
            # 创建激励函数模块
            self.boundary_reward = TADBoundaryReward(
                alpha=1.0, beta=2.0, gamma=1.5, window_size=5
            )
            
            # 创建边界精化模块
            self.boundary_refiner = BoundaryRefiner(
                reward_module=self.boundary_reward,
                iterations=optimization_iterations,
                learning_rate=0.01
            )
            
            # 边界增强模块
            self.boundary_enhancer = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # 对抗训练专用层
        if adv_mode:
            self.feature_perturb = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.InstanceNorm2d(16),
                nn.ReLU()
            )
    
    def forward(self, x, return_uncertainty=False, n_samples=0):
        """前向传播，支持不确定性估计"""
        # 如果没有指定采样数量，使用默认值
        if n_samples <= 0:
            n_samples = self.mc_samples if return_uncertainty else 1
        
        # 首先进行Canny边缘检测
        with torch.set_grad_enabled(True):  # 确保梯度计算启用
            # 确保输入需要梯度
            if not x.requires_grad:
                x.requires_grad_(True)
            
            # 调用Canny检测器
            canny_edges, _ = self.canny_detector(x)
        
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # 应用自我反思模块优化特征
        if hasattr(self, 'reflect3'):
            enc3_refined, unc3 = self.reflect3(enc3)
        else:
            enc3_refined = enc3
            unc3 = torch.zeros(enc3.shape[0], 1, enc3.shape[2], enc3.shape[3], device=enc3.device)
        
        # 解码器路径（上采样）
        dec4_up = F.interpolate(enc4, size=enc3_refined.shape[2:], mode='bilinear', align_corners=False)
        dec4_cat = torch.cat([dec4_up, enc3_refined], dim=1)
        dec4 = self.dec4(dec4_cat)
        
        dec3_up = F.interpolate(dec4, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec3_cat = torch.cat([dec3_up, enc2], dim=1)
        dec3 = self.dec3(dec3_cat)
        
        dec2_up = F.interpolate(dec3, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec2_cat = torch.cat([dec2_up, enc1], dim=1)
        dec2 = self.dec2(dec2_cat)
        
        # 最终解码层 - 不再拼接额外特征
        dec1 = self.dec1(F.interpolate(dec2, size=x.shape[2:], mode='bilinear', align_corners=False))
        
        # TAD边界增强
        edge_map = self.edge_enhancement(dec1)
        
        # 边缘融合 - 结合预测边缘和Canny边缘
        # 确保canny_edges维度与edge_map一致
        if canny_edges.shape != edge_map.shape:
            canny_edges = F.interpolate(canny_edges, size=edge_map.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        # 修复：使用clone()保留梯度流
        enhanced_edge = torch.sigmoid(
            self.edge_weight * edge_map + self.canny_weight * canny_edges
        ).clone()
        
        # 最终分割输出
        final_out = self.final(dec1)
        
        # 如果使用内部边界优化
        if self.use_internal_optimization and hasattr(self, 'boundary_refiner'):
            # 提取特征用于边界精化
            features = dec1  # 使用解码器最后层特征
            
            # 确保输入需要梯度
            final_out.requires_grad_(True)
            
            # 明确使用需要梯度的上下文
            with torch.set_grad_enabled(True):
                # 精化边界
                refined_out, reward_history = self.boundary_refiner(final_out, x, features)
            
            # 边界增强的最终输出 - 修改权重比例
            if hasattr(self, 'boundary_enhancer'):
                refined_features = self.boundary_enhancer(dec1)
                output = refined_out * 0.6 + enhanced_edge * 0.4  # 增加边缘权重
            else:
                output = refined_out * 0.6 + enhanced_edge * 0.4  # 增加边缘权重
        else:
            # 无内部优化时的输出 - 同样增加边缘权重
            output = final_out * 0.6 + enhanced_edge * 0.4
        
        # 不确定性估计（不论训练模式）
        uncertainty = None
        if return_uncertainty or (hasattr(self, 'uncertainty_estimator') and self.training and self.adv_mode):
            # 对于不确定性估计，我们可以使用no_grad，因为它不需要影响主网络的梯度流
            with torch.no_grad():
                mean_pred, uncertainty = self.uncertainty_estimator(dec1.detach(), n_samples)
            
            # 仅在训练时使用不确定性调整输出，并确保梯度流
            if self.training:
                # 不确定性引导的输出调整 - 保持计算图连接
                uncertainty_weight = torch.sigmoid(uncertainty * 5)
                # 创建新张量以保持梯度流
                uncertainty_term = enhanced_edge * (uncertainty_weight * 0.3)
                # 使用操作组合确保梯度正确传播
                output = output * (1 - uncertainty_weight * 0.3) + uncertainty_term
        
        # 确保输出类型一致
        if output.dtype != x.dtype:
            output = output.type_as(x)
        
        # 关键修复：确保在return_uncertainty=True时始终返回不确定性
        if return_uncertainty:
            # 如果uncertainty为None，创建零张量
            if uncertainty is None:
                uncertainty = torch.zeros_like(output)
            return output, uncertainty
        else:
            return output
    
    def get_kl_divergence(self):
        """获取网络中所有贝叶斯层的KL散度"""
        kl_div = 0.0
        
        # 收集编码器的KL散度
        for layer in self.enc1:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        for layer in self.enc2:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
                
        for layer in self.enc3:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        for layer in self.enc4:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        # 收集解码器的KL散度
        for layer in self.dec4:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        for layer in self.dec3:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        for layer in self.dec2:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
        
        for layer in self.dec1:
            if hasattr(layer, 'kl_divergence'):
                kl_div += layer.kl_divergence
                
        return kl_div

class SimplifiedUNet(nn.Module):
    """简化版U-Net用于分割任务"""
    def __init__(self, input_channels=1, output_channels=1, adv_mode=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 编码器 (下采样路径)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
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
    
    def forward(self, x):
        """前向传播"""
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        # 解码器路径与跳跃连接
        dec2_up = F.interpolate(enc3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2_cat = torch.cat([dec2_up, enc2], dim=1)
        dec2 = self.dec2(dec2_cat)
        
        dec1_up = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1_cat = torch.cat([dec1_up, enc1], dim=1)
        dec1 = self.dec1(dec1_cat)
        
        # TAD边界增强
        edge_map = self.edge_enhancement(dec1)
        
        # 最终分割输出
        final_out = self.final(dec1)
        
        # 增强边界的分割结果
        enhanced_out = final_out * 0.7 + edge_map * 0.3
        enhanced_out = torch.clamp(enhanced_out, 0.0, 1.0)
        
        # 确保输出类型与输入一致（支持混合精度训练）
        if enhanced_out.dtype != x.dtype:
            enhanced_out = enhanced_out.to(x.dtype)
        
        # 在解码路径添加扰动
        if self.adv_mode and hasattr(self, 'feature_perturb'):
            dec1 = self.feature_perturb(dec1)
        
        return enhanced_out
