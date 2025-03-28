import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TADBoundaryReward(nn.Module):
    """TAD边界内部激励函数
    
    基于三元组合评估TAD边界质量:
    1. 内部一致性(IC): 确保TAD内部连接模式统一
    2. 边缘显著性(ES): 强化真实边界的梯度特征
    3. 变点检测(CP): 精确定位统计特性变化点
    """
    
    def __init__(self, alpha=1.0, beta=2.0, gamma=1.5, window_size=5):
        """初始化TAD边界激励模块
        
        Args:
            alpha: 内部一致性权重
            beta: 边缘显著性权重
            gamma: 变点检测权重
            window_size: 方向性指数计算窗口大小
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        
        # 梯度计算的卷积核
        self.grad_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
        self.grad_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
    
    def compute_internal_consistency(self, segmentation, hic_matrix):
        """计算TAD内部一致性分数
        
        采用方向性指数(DI)计算每个位置的内部一致性
        
        Args:
            segmentation: TAD分割图，形状[B, 1, H, W]
            hic_matrix: 原始Hi-C矩阵，形状[B, 1, H, W]
            
        Returns:
            consistency_score: 内部一致性得分，值越低越好
        """
        batch_size, _, h, w = segmentation.shape
        
        # 获取离散的TAD标签
        # 使用阈值0.5二值化，然后应用连通成分标记
        tad_labels = (segmentation > 0.5).float()
        
        # 计算方向性指数的效率版本
        # 1. 使用卷积计算上游和下游窗口的接触总和
        # 这比循环快得多，特别是对于大批量/大矩阵
        window = self.window_size
        
        # 填充以处理边界
        padded_hic = F.pad(hic_matrix, (window, window, window, window), mode='constant', value=0)
        
        # 为每个位置计算上游和下游的总和
        upstream_sum = torch.zeros((batch_size, 1, h, w), device=segmentation.device)
        downstream_sum = torch.zeros((batch_size, 1, h, w), device=segmentation.device)
        
        # 使用高效的平移和加法而不是循环
        for i in range(window):
            # 下游接触
            downstream_sum += torch.roll(padded_hic[:, :, window:-window, window:-window], 
                                         shifts=(0, i+1), dims=(2, 3))
            # 上游接触
            upstream_sum += torch.roll(padded_hic[:, :, window:-window, window:-window], 
                                       shifts=(0, -(i+1)), dims=(2, 3))
        
        # 计算方向性指数 (DI)
        total = upstream_sum + downstream_sum + 1e-8  # 避免除零
        di_score = (downstream_sum - upstream_sum) / total
        
        # 计算每个TAD区域的一致性分数
        # TAD内部应该有一致的DI模式
        tad_consistency = torch.var(di_score * tad_labels, dim=(2, 3))
        
        return tad_consistency.mean()
    
    def compute_edge_significance(self, segmentation):
        """计算边缘显著性分数
        
        使用图像梯度计算边界的显著性，较大的梯度表示更清晰的边界
        
        Args:
            segmentation: TAD分割图，形状[B, 1, H, W]
            
        Returns:
            edge_score: 边缘显著性得分，值越高越好
        """
        # 计算水平和垂直梯度
        grad_x = F.conv2d(segmentation, self.grad_x, padding=1)
        grad_y = F.conv2d(segmentation, self.grad_y, padding=1)
        
        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 提取前10%的最大值作为边界指示器
        batch_size = segmentation.shape[0]
        topk = max(1, int(0.1 * segmentation.shape[2] * segmentation.shape[3]))
        
        edge_scores = []
        for i in range(batch_size):
            # 获取每个样本的梯度图
            sample_grad = grad_magnitude[i].flatten()
            # 取top-k梯度值
            top_values, _ = torch.topk(sample_grad, topk)
            # 使用最小值作为阈值，识别所有显著的边界
            threshold = top_values[-1]
            # 计算显著边界的平均强度
            significant_edges = sample_grad[sample_grad >= threshold]
            edge_scores.append(significant_edges.mean())
        
        # 边缘显著性得分：越高越好
        edge_score = torch.stack(edge_scores).mean()
        return edge_score
    
    def compute_change_point_detection(self, segmentation, features):
        """计算变点检测分数
        
        识别特征统计属性发生显著变化的位置
        
        Args:
            segmentation: TAD分割图，形状[B, 1, H, W]
            features: 特征图，形状[B, C, H, W]
            
        Returns:
            cp_score: 变点检测得分，值越高越好
        """
        batch_size, _, h, w = segmentation.shape
        
        # 计算分割图的边缘
        # 使用简单的差分操作
        edge_h = torch.abs(segmentation[:, :, 1:, :] - segmentation[:, :, :-1, :])
        edge_v = torch.abs(segmentation[:, :, :, 1:] - segmentation[:, :, :, :-1])
        
        # 填充使尺寸一致
        edge_h = F.pad(edge_h, (0, 0, 0, 1), "constant", 0)
        edge_v = F.pad(edge_v, (0, 1, 0, 0), "constant", 0)
        
        # 边缘图 = 水平边缘 OR 垂直边缘
        edge_map = torch.maximum(edge_h, edge_v)
        
        # 对每个边界点，计算两侧特征的统计差异
        window = self.window_size
        cp_scores = []
        
        # 计算显著的边界点
        _, edge_indices = torch.topk(edge_map.flatten(2), k=max(1, h//10), dim=2)
        
        for b in range(batch_size):
            sample_scores = []
            # 获取当前样本的特征
            sample_features = features[b]  # [C, H, W]
            
            # 对每个显著边界点计算统计差异
            for idx in edge_indices[b, 0]:
                # 转换为2D索引
                i, j = idx // w, idx % w
                
                # 确保有足够的上下文窗口
                if i < window or i >= h - window or j < window or j >= w - window:
                    continue
                
                # 提取边界两侧的特征
                before_features = sample_features[:, i-window:i, j-window:j]
                after_features = sample_features[:, i:i+window, j:j+window]
                
                # 计算统计差异
                before_mean = before_features.mean(dim=(1, 2))
                after_mean = after_features.mean(dim=(1, 2))
                before_std = before_features.std(dim=(1, 2)) + 1e-8
                after_std = after_features.std(dim=(1, 2)) + 1e-8
                
                # 计算归一化的均值差异
                mean_diff = torch.abs(after_mean - before_mean) / (before_std + after_std)
                sample_scores.append(mean_diff.mean())
            
            # 如果找到了有效的边界点
            if sample_scores:
                cp_scores.append(torch.stack(sample_scores).mean())
            else:
                cp_scores.append(torch.tensor(0.0, device=segmentation.device))
        
        # 变点检测得分：越高越好
        cp_score = torch.stack(cp_scores).mean()
        return cp_score
    
    def forward(self, segmentation, hic_matrix, features):
        """计算综合激励分数
        
        Args:
            segmentation: TAD分割图，形状[B, 1, H, W]
            hic_matrix: 原始Hi-C矩阵，形状[B, 1, H, W]
            features: 特征图，形状[B, C, H, W]
            
        Returns:
            reward: 综合激励分数 (负值，因为作为损失最小化)
            components: 各组件分数的字典
        """
        # 计算三元组件
        ic_score = self.compute_internal_consistency(segmentation, hic_matrix)
        es_score = self.compute_edge_significance(segmentation)
        cp_score = self.compute_change_point_detection(segmentation, features)
        
        # 按照公式组合分数: IC_loss - β·ES + γ·CP
        # 注意ES和CP是越高越好，所以取负号使其符合最小化目标
        reward = self.alpha * ic_score - self.beta * es_score + self.gamma * cp_score
        
        # 返回综合激励分数和各组件
        components = {
            'internal_consistency': ic_score.item(),
            'edge_significance': es_score.item(),
            'change_point': cp_score.item(),
            'total_reward': reward.item()
        }
        
        return reward, components

class BoundaryRefiner(nn.Module):
    """TAD边界精化模块
    
    基于内部激励函数的迭代边界优化器
    """
    
    def __init__(self, reward_module, iterations=2, learning_rate=0.01, 
                 early_stop_threshold=0.01):
        """初始化边界精化模块
        
        Args:
            reward_module: TAD边界激励函数模块
            iterations: 最大迭代次数
            learning_rate: 边界优化学习率
            early_stop_threshold: 提前停止阈值
        """
        super().__init__()
        self.reward = reward_module
        self.iterations = iterations
        self.lr = learning_rate
        self.threshold = early_stop_threshold
        
    def forward(self, initial_segmentation, hic_matrix, features):
        """迭代优化TAD边界 - 内存优化版本
        
        使用梯度累积而非存储中间状态，更适合大分辨率图像
        """
        # 使用分离的计算流，避免影响主反向传播
        with torch.no_grad():
            # 初始分割结果
            current_seg = initial_segmentation.clone()
            
            # 循环优化，避免存储所有中间状态
            reward_history = []
            
            for i in range(self.iterations):
                # 设置当前分割需要梯度
                current_seg.requires_grad_(True)
                
                # 前向传播计算激励分数
                reward, components = self.reward(current_seg, hic_matrix, features)
                reward_history.append(components)
                
                # 如果不是最后一轮，计算梯度并更新
                if i < self.iterations - 1:
                    # 计算梯度
                    grads = torch.autograd.grad(reward, current_seg, 
                                                create_graph=False, retain_graph=False)[0]
                    
                    # 分离当前状态并更新
                    current_seg = current_seg.detach()
                    current_seg = current_seg - self.lr * grads
                    
                    # 确保值域在[0,1]
                    current_seg = torch.clamp(current_seg, 0.0, 1.0)
                    
        return current_seg, reward_history 