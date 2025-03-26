import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EdgeAwareBiLSTM(nn.Module):
    """BiLSTM边界判别器，输出TAD边界概率"""
    def __init__(self, input_dim=64, hidden_dim=32, with_classifier=True):
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
        
        # 新增真实性分类器
        self.real_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        ) if with_classifier else None
    
    def forward(self, features, regions=None, hic_matrix=None, is_sequence=False):
        """
        分析特征并输出边界概率
        
        Args:
            features: 特征张量 [B, C, H, W] 或 [B, L, D]
            regions: 可选的区域列表 [(start, end, type), ...]
            hic_matrix: 可选的Hi-C矩阵
            is_sequence: 标识输入是否为生成器产生的序列
            
        Returns:
            boundary_probs: 边界概率 [B, L]
            boundary_adj: 边界调整建议 [B, L]
            real_prob: 真实性概率
        """
        # 首先确保输入数据连续性
        if not features.is_contiguous():
            features = features.contiguous()
        
        # 简化处理逻辑，避免频繁维度转换
        if is_sequence:  # 序列输入 [B,1,L]
            # 直接转换为LSTM期望的格式 [B,L,D]
            B, C, L = features.shape
            features_flat = features.transpose(1, 2)  # [B,L,C]
        elif features.dim() == 4:  # 4D输入 [B,C,H,W]
            B, C, H, W = features.shape
            features_flat = features.view(B, C, H*W).transpose(1, 2)  # [B,H*W,C]
        elif features.dim() == 3 and not is_sequence:  # 3D非序列输入 [B,H,W]
            B, H, W = features.shape
            features_flat = features.reshape(B, H*W, 1)  # [B,H*W,1]
        elif features.dim() == 2:  # 2D输入 [H,W]
            features = features.unsqueeze(0)  # [1,H,W]
            H, W = features.shape[1:]
            features_flat = features.reshape(1, H*W, 1)  # [1,H*W,1]
        else:
            raise ValueError(f"不支持的输入维度: {features.shape}")
        
        # 确保LSTM输入连续
        features_flat = features_flat.contiguous()
        
        # 投影层处理
        projected = self.projection(features_flat)  # [B,L,hidden_dim]
        
        # LSTM处理 (明确检查输入连续性)
        lstm_out, _ = self.bilstm(projected)  # [B,L,hidden_dim*2]
        
        # 计算所有位置的边界概率
        batch_size, seq_len, _ = lstm_out.shape
        boundary_probs_chunk = torch.zeros(batch_size, seq_len, device=projected.device)
        
        for i in range(seq_len):
            boundary_probs_chunk[:, i] = self.boundary_classifier(lstm_out[:, i]).squeeze(-1)
        
        # 为序列起始和结束位置增强边界概率信号
        boundary_probs_chunk[:, 0] = boundary_probs_chunk[:, 0] * 1.2  # 增强左边界
        boundary_probs_chunk[:, -1] = boundary_probs_chunk[:, -1] * 1.2  # 增强右边界
        
        # 约束概率范围在[0,1]之间，同时确保精度正确
        boundary_probs_chunk = torch.clamp(boundary_probs_chunk, 0.0, 1.0)
        
        # 生成边界调整建议 (基于概率梯度)
        boundary_adj_chunk = torch.zeros_like(boundary_probs_chunk)
        for b in range(batch_size):
            for i in range(1, seq_len-1):
                # 根据概率梯度确定调整方向
                left_grad = boundary_probs_chunk[b, i] - boundary_probs_chunk[b, i-1]
                right_grad = boundary_probs_chunk[b, i] - boundary_probs_chunk[b, i+1]
                
                if left_grad < 0 and abs(left_grad) > abs(right_grad):
                    boundary_adj_chunk[b, i] = -1  # 向左调整
                elif right_grad < 0 and abs(right_grad) > abs(left_grad):
                    boundary_adj_chunk[b, i] = 1   # 向右调整
        
        # 返回前确保张量类型与输入一致（支持混合精度训练）
        if projected.dtype != boundary_probs_chunk.dtype:
            boundary_probs_chunk = boundary_probs_chunk.to(projected.dtype)
            boundary_adj_chunk = boundary_adj_chunk.to(projected.dtype)
        
        # 新增真实性判断
        real_prob = self.real_classifier(lstm_out.mean(dim=1)) if self.real_classifier else None
        
        boundary_probs = boundary_probs_chunk
        boundary_adj = boundary_adj_chunk
        real_prob = real_prob
        
        return boundary_probs, boundary_adj, real_prob