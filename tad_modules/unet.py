import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiscaleAttention(nn.Module):
    """轻量级多尺度自注意力，用于TAD尺度评估"""
    def __init__(self, in_channels=16):
        super().__init__()
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
        
    def forward(self, x):
        """
        输入: [B,C,H,W] 特征图
        输出: 
            - 多尺度汇总特征
            - 各尺度权重
        """
        B, C, H, W = x.shape
        
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
        multi_scale = torch.cat(scale_features, dim=1)
        
        # 评估各尺度权重
        scale_weights = self.scale_evaluator(multi_scale)  # [B,4,1,1]
        
        # 生成加权融合结果
        weighted_feat = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            weighted_feat += feat * scale_weights[:, i:i+1]
            
        return weighted_feat, scale_weights

class SimplifiedUNet(nn.Module):
    """添加多尺度反思机制的U-Net"""
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
        
        # 添加多尺度反思模块
        self.multiscale_reflection = MultiscaleAttention(in_channels=16)
        
        # 修正TAD尺度选择网络的输入通道数 (16+8=24)
        self.tad_scale_selector = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),  # 修改这里，输入通道从16+8=24
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),  # 保持输出通道与原边界增强层一致
            nn.ReLU(inplace=True)
        )
        
        # 最终输出调整
        self.final_sigmoid = nn.Sigmoid()
    
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
        
        # ----- 多尺度反思机制 -----
        # 1. 生成多尺度特征和权重
        multiscale_feat, scale_weights = self.multiscale_reflection(dec1)  # dec1[B,16,H,W], multiscale_feat[B,8,H,W]
        
        # 2. 融合多尺度特征与原始特征
        enhanced_feat = torch.cat([dec1, multiscale_feat], dim=1)  # [B,24,H,W]
        tad_feat = self.tad_scale_selector(enhanced_feat)  # [B,16,H,W]
        
        # 3. 边界增强和分割输出
        edge_map = self.edge_enhancement(tad_feat)  # [B,1,H,W]
        final_out = self.final(tad_feat)  # [B,1,H,W]
        
        # 尺度自适应的输出调整
        enhanced_out = final_out * 0.7 + edge_map * 0.3
        enhanced_out = torch.clamp(enhanced_out, 0.0, 1.0)  # 确保输出在[0,1]范围内
        
        # 保存尺度信息供外部使用
        self.scale_info = {
            'weights': scale_weights,
            'optimal_idx': torch.argmax(scale_weights, dim=1)
        }
        
        return enhanced_out
