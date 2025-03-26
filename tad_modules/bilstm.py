import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EdgeAwareBiLSTM(nn.Module):
    """BiLSTM边界判别器，输出TAD边界概率"""
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 降维投影 - 减少计算量
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # 双向LSTM - 序列边缘分析
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # 边界概率判别器 - 输出边界概率
        self.boundary_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 使用Sigmoid确保输出为概率值
        )
    
    def forward(self, features, regions=None, hic_matrix=None):
        """
        分析特征并输出边界概率
        
        Args:
            features: 特征张量 [B, C, H, W] 或 [B, L, D]
            regions: 可选的区域列表 [(start, end, type), ...]
            hic_matrix: 可选的Hi-C矩阵
            
        Returns:
            boundary_probs: 边界概率 [B, L]
            boundary_adj: 边界调整建议 [B, L]
        """
        # 处理特征形状
        if features.dim() == 4:  # [B, C, H, W]
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            features_flat = features  # 已经是 [B, L, D] 格式
        
        # 降维
        features_proj = self.projection(features_flat)  # [B, L, hidden_dim]
        
        # 通过BiLSTM处理序列
        lstm_out, _ = self.bilstm(features_proj)  # [B, L, hidden_dim*2]
        
        # 计算所有位置的边界概率
        batch_size, seq_len, _ = lstm_out.shape
        boundary_probs = torch.zeros(batch_size, seq_len, device=features.device)
        
        for i in range(seq_len):
            boundary_probs[:, i] = self.boundary_classifier(lstm_out[:, i]).squeeze(-1)
        
        # 为序列起始和结束位置增强边界概率信号
        boundary_probs[:, 0] = boundary_probs[:, 0] * 1.2  # 增强左边界
        boundary_probs[:, -1] = boundary_probs[:, -1] * 1.2  # 增强右边界
        
        # 约束概率范围在[0,1]之间，同时确保精度正确
        boundary_probs = torch.clamp(boundary_probs, 0.0, 1.0)
        
        # 生成边界调整建议 (基于概率梯度)
        boundary_adj = torch.zeros_like(boundary_probs)
        for b in range(batch_size):
            for i in range(1, seq_len-1):
                # 根据概率梯度确定调整方向
                left_grad = boundary_probs[b, i] - boundary_probs[b, i-1]
                right_grad = boundary_probs[b, i] - boundary_probs[b, i+1]
                
                if left_grad < 0 and abs(left_grad) > abs(right_grad):
                    boundary_adj[b, i] = -1  # 向左调整
                elif right_grad < 0 and abs(right_grad) > abs(left_grad):
                    boundary_adj[b, i] = 1   # 向右调整
        
        # 返回前确保张量类型与输入一致（支持混合精度训练）
        if features.dtype != boundary_probs.dtype:
            boundary_probs = boundary_probs.to(features.dtype)
            boundary_adj = boundary_adj.to(features.dtype)
        
        return boundary_probs, boundary_adj