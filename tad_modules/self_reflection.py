import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfReflectionModule(nn.Module):
    """自我反思模块，评估预测质量并提供反馈"""
    def __init__(self, feature_channels, reduction=8):
        super().__init__()
        self.feature_channels = feature_channels
        
        # 特征质量评估 - 使用通道注意力机制
        self.quality_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, feature_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // reduction, feature_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 不确定性估计 - 空间注意力
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 自适应特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # 记录历史预测用于自我评估
        self.register_buffer('prediction_history', None)
        self.history_len = 3
        
    def forward(self, features, skip_features=None, prev_prediction=None):
        batch_size, channels, height, width = features.shape
        
        # 特征质量评估 - 确定哪些通道重要
        quality_weights = self.quality_estimator(features)  # [B, C, 1, 1]
        
        # 不确定性估计 - 确定哪些空间区域不确定
        uncertainty_map = self.uncertainty_estimator(features)  # [B, 1, H, W]
        
        # 更新预测历史
        if prev_prediction is not None:
            # 确保prev_prediction的维度正确 [B, H, W] 或 [B, 1, H, W]
            if prev_prediction.dim() == 4 and prev_prediction.shape[1] == 1:
                # 如果是 [B, 1, H, W]，去掉通道维度
                prev_prediction = prev_prediction.squeeze(1)
            elif prev_prediction.dim() != 3:
                raise ValueError(f"预期prev_prediction维度为3或4(通道=1)，实际为: {prev_prediction.dim()}")
            
            # 确保空间尺寸与当前特征匹配
            if prev_prediction.shape[1:] != (height, width):
                prev_prediction = F.interpolate(
                    prev_prediction.unsqueeze(1), 
                    size=(height, width),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # 重新初始化历史，确保空间维度匹配
            if self.prediction_history is None or self.prediction_history.shape[2:] != (height, width):
                # 初始化预测历史 [B, history_len, H, W]
                self.prediction_history = torch.zeros(
                    (batch_size, self.history_len, height, width), 
                    device=features.device
                )
            
            # 确保批次大小匹配
            if self.prediction_history.shape[0] != batch_size:
                # 如果批次大小发生变化，重新初始化历史
                self.prediction_history = torch.zeros(
                    (batch_size, self.history_len, height, width),
                    device=features.device
                )
            
            # 移动历史记录并添加新预测
            # 取出除了第一个时间步外的所有历史 [B, history_len-1, H, W]
            if self.prediction_history.shape[1] > 1:
                shifted_history = self.prediction_history[:, 1:]
                # 添加新预测，确保维度匹配 [B, 1, H, W]
                new_prediction = prev_prediction.unsqueeze(1)
                # 拼接历史与新预测
                self.prediction_history = torch.cat([shifted_history, new_prediction], dim=1)
            else:
                # 历史长度为1或0的情况
                self.prediction_history[:, 0] = prev_prediction
            
            # 计算预测稳定性
            if self.prediction_history.shape[1] >= 2:
                prediction_variance = torch.var(self.prediction_history, dim=1, keepdim=True)
                # 结合方差和不确定性
                uncertainty_map = 0.7 * uncertainty_map + 0.3 * prediction_variance
        
        # 应用通道注意力
        refined_features = features * quality_weights
        
        # 应用不确定性引导的特征增强
        enhanced_features = self.feature_enhancer(refined_features)
        
        # 特征融合 (如果有跳跃连接)
        if skip_features is not None:
            # 确保维度匹配
            if skip_features.shape[2:] != enhanced_features.shape[2:]:
                skip_features = F.interpolate(
                    skip_features, 
                    size=enhanced_features.shape[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 基于不确定性的自适应融合
            fusion_weight = 1.0 - uncertainty_map  # 不确定区域减少跳跃连接贡献
            combined_features = enhanced_features + skip_features * fusion_weight
        else:
            combined_features = enhanced_features
        
        return combined_features, uncertainty_map

class UncertaintyEstimator(nn.Module):
    """不确定性估计模块，用于多次蒙特卡洛采样"""
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, decoder_output, num_samples=5):
        """进行多次前向传播"""
        # 存储多次采样结果
        predictions = []
        
        # 记录原始训练状态
        training_state = self.training
        
        # 设置为训练模式以启用dropout
        self.train(True)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # dropout会在每次前向传播中随机
                x = self.dropout(decoder_output)
                predictions.append(x)
        
        # 恢复原始状态
        self.train(training_state)
        
        # 计算预测均值和不确定性
        predictions = torch.stack(predictions, dim=0)  # [samples, B, C, H, W]
        mean_prediction = torch.mean(predictions, dim=0)  # [B, C, H, W]
        uncertainty = torch.var(predictions, dim=0).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        return mean_prediction, uncertainty 