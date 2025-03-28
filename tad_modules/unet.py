import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .bayesian_layers import BayesianConvBlock
from .self_reflection import SelfReflectionModule, UncertaintyEstimator

class BayesianUNet(nn.Module):
    """结合贝叶斯层和自我反思机制的U-Net"""
    def __init__(self, input_channels=1, output_channels=1, adv_mode=False, mc_samples=5):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.mc_samples = mc_samples
        
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
        
        # 自我反思模块 - 在各跳跃连接处添加
        self.reflect1 = SelfReflectionModule(16)
        self.reflect2 = SelfReflectionModule(32)
        self.reflect3 = SelfReflectionModule(64)
        
        # 解码器 (上采样路径)
        self.dec2 = nn.Sequential(
            BayesianConvBlock(96, 32),
            BayesianConvBlock(32, 32)
        )
        
        self.dec1 = nn.Sequential(
            BayesianConvBlock(48, 16),
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
        
        # 对抗训练专用层
        self.adv_mode = adv_mode
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
            n_samples = 1 if not return_uncertainty else self.mc_samples
        
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        # 自我反思 - 编码器特征
        enc3_refined, unc3 = self.reflect3(enc3)
        
        # 解码器路径与跳跃连接
        dec2_up = F.interpolate(enc3_refined, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        
        # 自我反思 - 第二层跳跃连接
        enc2_refined, unc2 = self.reflect2(enc2, prev_prediction=unc3)
        dec2_cat = torch.cat([dec2_up, enc2_refined], dim=1)
        dec2 = self.dec2(dec2_cat)
        
        dec1_up = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        
        # 自我反思 - 第一层跳跃连接
        enc1_refined, unc1 = self.reflect1(enc1, prev_prediction=unc2)
        dec1_cat = torch.cat([dec1_up, enc1_refined], dim=1)
        dec1 = self.dec1(dec1_cat)
        
        # TAD边界增强
        edge_map = self.edge_enhancement(dec1)
        
        if return_uncertainty:
            # 使用不确定性估计器 - 多次采样
            final_out, uncertainty = self.uncertainty_estimator(dec1, n_samples)
            final_out = self.final(final_out)
            
            # 增强边界的分割结果
            enhanced_out = final_out * 0.7 + edge_map * 0.3
            enhanced_out = torch.clamp(enhanced_out, 0.0, 1.0)
            
            # 在解码路径添加扰动
            if self.adv_mode and hasattr(self, 'feature_perturb'):
                dec1 = self.feature_perturb(dec1)
            
            # 返回预测结果和不确定性
            return enhanced_out, uncertainty
        else:
            # 标准模式 - 单次预测
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
        
        # 收集解码器的KL散度
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
