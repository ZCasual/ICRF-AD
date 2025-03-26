import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimplifiedUNet(nn.Module):
    """简化版U-Net用于分割任务"""
    def __init__(self, input_channels=1, output_channels=1):
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
        
        return enhanced_out
