import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DifferentiableCanny(nn.Module):
    """可微分Canny边缘检测器
    
    相比传统Canny算法，此版本所有操作均为可微分，
    支持反向传播以便在神经网络中端到端训练。
    """
    def __init__(self, low_threshold=0.1, high_threshold=0.3, kernel_size=3, sigma=1.0):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # 创建高斯平滑卷积核
        self.kernel_size = kernel_size
        sigma = sigma if sigma else (kernel_size - 1) / 6
        
        # 生成高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2
        
        # 计算高斯核
        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # 将高斯核转换为适用于nn.Conv2d的形式
        self.gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # Sobel卷积核
        self.sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册卷积核为缓冲区，不作为模型参数
        self.register_buffer('gaussian', self.gaussian_kernel)
        self.register_buffer('sobel_x', self.sobel_kernel_x)
        self.register_buffer('sobel_y', self.sobel_kernel_y)
        
    def forward(self, x):
        """前向传播实现可微分Canny检测"""
        # 确保输入是4D张量 [B,C,H,W]
        input_dim = x.dim()
        if input_dim == 3:
            x = x.unsqueeze(0)  # 添加批次维度
        
        # 如果输入是多通道，转为单通道
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # 步骤1: 高斯平滑
        smoothed = F.conv2d(x, self.gaussian, padding=self.kernel_size//2)
        
        # 步骤2: 计算梯度
        grad_x = F.conv2d(smoothed, self.sobel_x, padding=1)
        grad_y = F.conv2d(smoothed, self.sobel_y, padding=1)
        
        # 步骤3: 计算梯度幅度和方向
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        grad_direction = torch.atan2(grad_y, grad_x)
        
        # 步骤4: 非最大抑制 (可微分版本)
        # 将方向量化为0, 45, 90, 135度
        directions = torch.round(grad_direction * (4 / math.pi)) % 4
        
        # 使用可微分的近似代替硬阈值的非最大抑制
        nms_result = grad_magnitude.clone()
        
        # 通过内插和卷积近似非最大抑制
        for i in range(4):  # 4个方向: 0, 45, 90, 135度
            mask = (directions == i).float()
            
            # 获取每个方向上的偏移
            if i == 0:  # 0度 - 水平
                offset_pos = F.pad(grad_magnitude[:, :, :, 1:], (0, 1, 0, 0))
                offset_neg = F.pad(grad_magnitude[:, :, :, :-1], (1, 0, 0, 0))
            elif i == 1:  # 45度 - 对角线
                offset_pos = F.pad(grad_magnitude[:, :, 1:, 1:], (0, 1, 0, 1))
                offset_neg = F.pad(grad_magnitude[:, :, :-1, :-1], (1, 0, 1, 0))
            elif i == 2:  # 90度 - 垂直
                offset_pos = F.pad(grad_magnitude[:, :, 1:, :], (0, 0, 0, 1))
                offset_neg = F.pad(grad_magnitude[:, :, :-1, :], (0, 0, 1, 0))
            else:  # 135度 - 对角线
                offset_pos = F.pad(grad_magnitude[:, :, 1:, :-1], (1, 0, 0, 1))
                offset_neg = F.pad(grad_magnitude[:, :, :-1, 1:], (0, 1, 1, 0))
            
            # 柔和的非最大抑制
            is_max = (grad_magnitude > offset_pos).float() * (grad_magnitude > offset_neg).float()
            nms_result = nms_result * (1 - mask + mask * is_max)
        
        # 步骤5: 滞后阈值处理
        # 使用Sigmoid函数代替硬阈值
        low_mask = torch.sigmoid((nms_result - self.low_threshold) * 10)
        high_mask = torch.sigmoid((nms_result - self.high_threshold) * 10)
        
        # 返回原始维度
        if input_dim == 3:
            nms_result = nms_result.squeeze(0)
            low_mask = low_mask.squeeze(0)
            high_mask = high_mask.squeeze(0)
        
        return nms_result * high_mask, nms_result * low_mask

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
        
        # 预测历史跟踪
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
            
            # 基于不确定性的自适应特征融合
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

class BiFPNBlock(nn.Module):
    """双向特征金字塔网络块 - 优化多尺度特征融合"""
    def __init__(self, feature_size, eps=1e-4):
        super(BiFPNBlock, self).__init__()
        self.epsilon = eps
        
        # 自适应权重（可学习）
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        # 特征处理卷积
        self.conv_up = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, groups=feature_size),
            nn.BatchNorm2d(feature_size),
            nn.Conv2d(feature_size, feature_size, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_down = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, groups=feature_size),
            nn.BatchNorm2d(feature_size),
            nn.Conv2d(feature_size, feature_size, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p3, p4, p5):
        """
        多尺度特征融合
        p3, p4, p5: 不同尺度的特征 (小到大)
        """
        # 权重归一化
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        
        # 自顶向下路径 (从高层特征到低层特征)
        p5_td = p5
        
        # P4自顶向下: 结合P5上采样和P4
        p4_in = w1[0] * p4 + w1[1] * F.interpolate(p5_td, size=p4.shape[2:], 
                                                mode='bilinear', align_corners=False)
        p4_td = self.conv_up(p4_in)
        
        # P3自顶向下: 结合P4上采样和P3
        p3_in = w1[0] * p3 + w1[1] * F.interpolate(p4_td, size=p3.shape[2:], 
                                               mode='bilinear', align_corners=False)
        p3_td = self.conv_up(p3_in)
        
        # 自底向上路径 (从低层特征到高层特征)
        p3_out = p3_td
        
        # P4自底向上: 结合P3下采样, P4原始输入, P4自顶向下结果
        p4_out_size = p4.shape[2:]
        p3_down = F.adaptive_max_pool2d(p3_out, output_size=p4_out_size)
        p4_out = w2[0] * p4 + w2[1] * p4_td + w2[2] * p3_down
        p4_out = self.conv_down(p4_out)
        
        # P5自底向上: 结合P4下采样, P5
        p5_out_size = p5.shape[2:]
        p4_down = F.adaptive_max_pool2d(p4_out, output_size=p5_out_size)
        p5_out = w2[0] * p5 + w2[1] * p5_td + w2[2] * p4_down
        p5_out = self.conv_down(p5_out)
        
        return p3_out, p4_out, p5_out 