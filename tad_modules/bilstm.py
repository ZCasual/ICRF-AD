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
        
        # 新增结构约束评估网络 - 结合多种评估标准
        self.structure_evaluator = nn.Sequential(
            nn.Linear(4, 16),  # 输入4种结构评分
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, regions=None, hic_matrix=None, is_sequence=False):
        """
        分析特征并输出边界概率
        
        Args:
            features: 特征张量 [B, C, H, W] 或 [B, L, D]
            regions: 可选的区域列表 [(start, end, type), ...]
            hic_matrix: 可选的Hi-C矩阵，用于计算结构约束
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
            # 直接转换为LSTM期望的格式 [B,L,C]
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
        
        # ----- 新增：结构约束评估 -----
        if hic_matrix is not None:
            # 应用结构约束，增强边界概率
            structure_scores = self._compute_structure_constraints(
                boundary_probs_chunk, hic_matrix, features
            )
            
            # 融合结构约束评分到边界概率中
            for b in range(batch_size):
                # 增强高结构约束分数的位置
                for i in range(seq_len):
                    # 使用结构分数加权边界概率
                    alpha = 0.7  # 边界检测器权重
                    beta = 0.3   # 结构约束评分权重
                    boundary_probs_chunk[b, i] = alpha * boundary_probs_chunk[b, i] + beta * structure_scores[b, i]
        
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
    
    def _compute_structure_constraints(self, boundary_probs, hic_matrix, features):
        """
        计算结构约束评分
        
        Args:
            boundary_probs: 边界概率 [B,L]
            hic_matrix: Hi-C接触矩阵 [B,H,W] 或 [B,1,H,W]
            features: 特征张量 [B,C,H,W]
            
        Returns:
            structure_scores: 结构约束评分 [B,L]
        """
        # 确保维度一致
        if hic_matrix.dim() == 4 and hic_matrix.shape[1] == 1:
            hic_matrix = hic_matrix.squeeze(1)  # [B,1,H,W] -> [B,H,W]
        
        batch_size, seq_len = boundary_probs.shape
        device = boundary_probs.device
        
        # 预先定义评分tensor
        di_scores = torch.zeros_like(boundary_probs)       # 内部一致性评分
        edge_scores = torch.zeros_like(boundary_probs)     # 边缘显著性评分
        change_scores = torch.zeros_like(boundary_probs)   # 变点检测评分
        length_scores = torch.zeros_like(boundary_probs)   # TAD长度评分
        
        # 计算最可能的边界位置
        potential_boundaries = []
        for b in range(batch_size):
            # 找出概率超过阈值的位置作为潜在边界
            idx = torch.where(boundary_probs[b] > 0.3)[0].cpu().numpy().tolist()
            if len(idx) == 0:  # 如果没有明显边界，使用概率最高的几个点
                values, indices = torch.topk(boundary_probs[b], min(5, seq_len))
                idx = indices.cpu().numpy().tolist()
            potential_boundaries.append(idx)
        
        # 逐个批次处理
        for b in range(batch_size):
            # 获取当前批次的矩阵
            matrix = hic_matrix[b]  # [H,W]
            
            # 获取潜在边界位置
            boundaries = potential_boundaries[b]
            boundaries = [0] + sorted(boundaries) + [seq_len-1]  # 添加起始和结束位置
            
            # 计算每个潜在TAD区域的结构约束评分
            for k in range(len(boundaries)-1):
                start, end = boundaries[k], boundaries[k+1]
                
                # 跳过长度为0或1的区域
                if end - start <= 1:
                    continue
                
                # 1. 计算内部一致性分数 (DI)
                di_score = self._compute_directionality_index(
                    matrix, start, end, window_size=min(10, (end-start)//2)
                )
                
                # 2. 计算边缘显著性分数
                edge_score = self._compute_edge_significance(
                    features[b], start, end
                )
                
                # 3. 计算变点检测分数
                change_score = self._compute_changepoint_detection(
                    matrix, start, end
                )
                
                # 4. 计算长度适宜性分数 - 奖励合适大小的TAD
                # 假设TAD的理想长度范围在10-50之间
                tad_len = end - start
                if 10 <= tad_len <= 50:
                    length_score = 1.0  # 理想长度
                elif tad_len < 10:
                    length_score = 0.3 + (tad_len / 10) * 0.7  # 线性惩罚过小TAD
                else:  # tad_len > 50
                    length_score = 1.0 - min(0.7, (tad_len - 50) / 100)  # 线性惩罚过大TAD
                
                # 将分数应用到对应位置 - 主要在边界处加强
                di_scores[b, start] = max(di_scores[b, start], di_score)
                di_scores[b, end] = max(di_scores[b, end], di_score)
                
                edge_scores[b, start] = max(edge_scores[b, start], edge_score)
                edge_scores[b, end] = max(edge_scores[b, end], edge_score)
                
                change_scores[b, start] = max(change_scores[b, start], change_score)
                change_scores[b, end] = max(change_scores[b, end], change_score)
                
                length_scores[b, start] = max(length_scores[b, start], length_score)
                length_scores[b, end] = max(length_scores[b, end], length_score)
            
            # 对于每个位置，使用评估网络计算综合结构约束评分
            combined_scores = torch.zeros_like(boundary_probs[b])
            for i in range(seq_len):
                # 将四种评分组合成一个输入向量
                scores_vector = torch.tensor([
                    di_scores[b, i],
                    edge_scores[b, i],
                    change_scores[b, i],
                    length_scores[b, i]
                ], device=device).unsqueeze(0)  # [1,4]
                
                # 使用评估网络得到最终评分
                combined_scores[i] = self.structure_evaluator(scores_vector).squeeze()
        
        # 归一化评分
        structure_scores = torch.zeros_like(boundary_probs)
        for b in range(batch_size):
            # 如果有有效分数，进行min-max归一化
            if torch.max(combined_scores) > torch.min(combined_scores):
                normalized = (combined_scores - torch.min(combined_scores)) / \
                            (torch.max(combined_scores) - torch.min(combined_scores) + 1e-8)
                structure_scores[b] = normalized
            else:
                # 没有差异时使用原始边界概率
                structure_scores[b] = boundary_probs[b]
        
        return structure_scores
    
    def _compute_directionality_index(self, matrix, start, end, window_size=5):
        """
        计算内部一致性 (Directionality Index)
        
        Args:
            matrix: 接触矩阵 [H,W]
            start, end: TAD区域的起始和结束位置
            window_size: 窗口大小
            
        Returns:
            di_score: 内部一致性评分 (0-1)
        """
        # 参数检查
        if end <= start:
            return 0.0
        
        # 提取子矩阵
        submatrix = matrix[start:end, start:end]
        
        # 计算每个位置的DI
        di_values = []
        
        for i in range(end - start):
            # 确定上下游窗口范围
            upstream_start = max(0, i - window_size)
            upstream_end = i
            
            downstream_start = i + 1
            downstream_end = min(end - start, i + window_size + 1)
            
            # 如果位置不足以形成有效窗口，则跳过
            if upstream_start >= upstream_end or downstream_start >= downstream_end:
                continue
            
            # 提取上下游区域
            upstream = submatrix[i, upstream_start:upstream_end]
            downstream = submatrix[i, downstream_start:downstream_end]
            
            # 计算DI值
            A = torch.sum(downstream)
            B = torch.sum(upstream)
            epsilon = 1e-8  # 防止除零
            
            di = (A - B) / (A + B + epsilon)
            di_values.append(di.abs().item())  # 使用绝对值，因为我们关心变化强度而非方向
        
        # 取DI均值作为评分 (0-1范围)
        if len(di_values) > 0:
            return min(1.0, sum(di_values) / len(di_values))
        else:
            return 0.0
    
    def _compute_edge_significance(self, feature, start, end):
        """
        计算边缘显著性
        
        Args:
            feature: 特征张量 [C,H,W]
            start, end: TAD区域的起始和结束位置
            
        Returns:
            edge_score: 边缘显著性评分 (0-1)
        """
        # 确保有效区间
        if end <= start + 1:
            return 0.0
        
        # 计算边界位置的特征梯度
        if start > 0 and end < feature.shape[-1] - 1:
            # 使用中心差分计算梯度
            start_grad = torch.norm(feature[:, start+1] - feature[:, start-1]) / 2
            end_grad = torch.norm(feature[:, end+1] - feature[:, end-1]) / 2
        elif start > 0:
            # 只能计算起始位置的梯度
            start_grad = torch.norm(feature[:, start] - feature[:, start-1])
            end_grad = torch.tensor(0.0, device=feature.device)
        elif end < feature.shape[-1] - 1:
            # 只能计算结束位置的梯度
            start_grad = torch.tensor(0.0, device=feature.device)
            end_grad = torch.norm(feature[:, end+1] - feature[:, end])
        else:
            # 无法计算梯度
            return 0.0
        
        # 计算区域内部平均梯度作为参考
        internal_grads = []
        for i in range(start+1, end):
            internal_grads.append(torch.norm(feature[:, i] - feature[:, i-1]))
        
        if len(internal_grads) > 0:
            avg_internal_grad = torch.stack(internal_grads).mean()
        else:
            avg_internal_grad = torch.tensor(1e-8, device=feature.device)
        
        # 边缘显著性 = 边界梯度与内部梯度的比值
        significance = (start_grad + end_grad) / (2 * avg_internal_grad + 1e-8)
        
        # 归一化到0-1范围
        return min(1.0, significance.item())
    
    def _compute_changepoint_detection(self, matrix, start, end):
        """
        计算变点检测分数
        
        Args:
            matrix: 接触矩阵 [H,W]
            start, end: TAD区域的起始和结束位置
            
        Returns:
            change_score: 变点检测评分 (0-1)
        """
        # 确保有效区间
        if end <= start + 3:  # 至少需要4个点才能形成两个有意义的区域
            return 0.0
        
        # 提取子矩阵
        submatrix = matrix[start:end, start:end]
        
        # 计算最佳分割点
        best_score = 0.0
        mid_point = (start + end) // 2  # 默认中点
        
        # 尝试不同的分割点
        for split in range(start + 1, end - 1):
            # 相对于子矩阵的分割位置
            split_idx = split - start
            
            # 计算分割前后区域的均值和标准差
            region1 = submatrix[:split_idx, :split_idx]
            region2 = submatrix[split_idx:, split_idx:]
            
            if region1.numel() > 0 and region2.numel() > 0:
                mean1 = torch.mean(region1)
                mean2 = torch.mean(region2)
                
                std1 = torch.std(region1) if region1.numel() > 1 else torch.tensor(1e-8, device=matrix.device)
                std2 = torch.std(region2) if region2.numel() > 1 else torch.tensor(1e-8, device=matrix.device)
                
                # 计算均值差异相对于方差的比例
                epsilon = 1e-8  # 防止除零
                score = torch.abs(mean1 - mean2) / (torch.sqrt(std1**2 + std2**2) + epsilon)
                
                if score > best_score:
                    best_score = score
                    mid_point = split
        
        # 归一化到0-1范围
        change_score = min(1.0, best_score.item())
        
        return change_score