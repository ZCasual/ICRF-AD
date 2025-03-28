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
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 可选：真实性分类器 - 判断序列是否真实
        self.real_classifier = None
        if with_classifier:
            self.real_classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x, is_sequence=False):
        """前向传播函数
        Args:
            x: 输入张量，可以是:
               - 序列形式 [B, C, L] - is_sequence=True
               - 或图像形式 [B, C, H, W] - is_sequence=False
            is_sequence: 指示输入是否已经是序列格式
            
        Returns:
            tuple: (边界概率, 边界调整建议, 真实性评分)
        """
        # 确保输入是张量
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是张量，得到: {type(x)}")
            
        # 获取批次大小
        batch_size = x.shape[0]
        
        # 1. 将输入处理为序列
        if not is_sequence:
            # 处理2D输入 [B, C, H, W]
            if x.dim() == 4:
                B, C, H, W = x.shape
                # 贝叶斯模型输入可能有多个通道 (包括不确定性)
                
                # 将2D图像转为1D序列 - 注意这里的修改
                if C > 1:
                    # 只使用第一个通道（分割结果）
                    x_seq = x[:, 0].view(B, 1, H*W)  # [B, 1, H*W]
                else:
                    x_seq = x.view(B, C, H*W)  # [B, C, H*W]
                    
                seq_len = H*W
            else:
                raise ValueError(f"当is_sequence=False时，预期维度为4，得到{x.dim()}")
        else:
            # 输入已经是序列 [B, C, L]
            if x.dim() != 3:
                raise ValueError(f"当is_sequence=True时，预期维度为3，得到{x.dim()}")
            B, C, seq_len = x.shape
            x_seq = x
        
        # 2. 调整维度顺序为 [B, L, C]
        x_seq = x_seq.permute(0, 2, 1)  # [B, L, C]
        
        # 3. 线性投影 - 将不同维度的输入统一转为hidden_dim
        # 检查输入通道数是否与投影层匹配
        if x_seq.shape[2] != self.input_dim:
            # 如果通道数不匹配，创建新的投影层
            self.projection = nn.Linear(x_seq.shape[2], self.hidden_dim).to(x_seq.device)
        
        projected = self.projection(x_seq)  # [B, L, hidden_dim]
        
        # 4. BiLSTM处理
        lstm_out, _ = self.bilstm(projected)  # [B, L, hidden_dim*2]
        
        # 5. 边界概率预测
        boundary_probs_chunk = self.boundary_classifier(lstm_out).squeeze(-1)  # [B, L]
        
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
        
        return boundary_probs, boundary_adj, real_prob