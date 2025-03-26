import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeometricFeatureExtractor(nn.Module):
    """几何特征提取器：检测Hi-C矩阵中的边缘和对角线结构"""
    def __init__(self, edge_threshold=0.1, angle_threshold=15, min_votes=3, feature_dim=64):
        super().__init__()
        self.edge_threshold = edge_threshold  # Canny边缘检测阈值
        self.angle_threshold = angle_threshold  # Hough变换角度阈值(度)
        self.min_votes = min_votes  # Hough变换最小投票数
        self.feature_dim = feature_dim
        
        # 边缘检测卷积核 - 水平Sobel
        self.horizontal_kernel = nn.Parameter(
            torch.tensor([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        
        # 边缘检测卷积核 - 垂直Sobel
        self.vertical_kernel = nn.Parameter(
            torch.tensor([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        
        # 对角线检测核 - 主对角线方向
        self.diag1_kernel = nn.Parameter(
            torch.tensor([[-1, -1, 0], 
                         [-1, 0, 1], 
                         [0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        
        # 对角线检测核 - 副对角线方向
        self.diag2_kernel = nn.Parameter(
            torch.tensor([[0, 1, 1], 
                         [-1, 0, 1], 
                         [-1, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        
        # TAD结构特征卷积核 - 捕获边界
        self.tad_kernel = nn.Parameter(
            torch.tensor([[1, -1, -1], 
                         [1, 2, -1], 
                         [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        
        # 1. 跳跃连接处理器 - 增强版，添加更多批归一化和激活函数
        self.skip_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 24, 1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 16, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(16),  # 额外的批归一化
                nn.ReLU(inplace=True)
            ) for _ in range(3)  # 对应short, mid, long
        ])
        
        # 特征融合层，处理所有输入特征
        self.fusion = nn.Sequential(
            # 使用固定大小的卷积层，始终支持完整特征
            nn.Conv2d(5 + 16*3, 32, 1),  # 5个几何特征 + 3个跳跃连接
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 2. 特征回传投影层 - 用于与Transformer各层连接
        self.feedback_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, feature_dim, 1),  # 投影到Transformer维度
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(3)  # 对应short, mid, long层
        ])
        
        # 几何引导注意力层
        self.geo_attention_weight = nn.Parameter(torch.tensor(0.1))  # λ参数
        
        # 添加内部跳跃连接缓存
        self._skip_features_cache = None
        
    def set_skip_features(self, skip_features):
        """设置用于下一次前向传播的跳跃连接特征"""
        self._skip_features_cache = skip_features
        
    def forward(self, features):
        """提取几何特征
        
        Args:
            features: 输入特征图 [B, C, H, W] 或 [C, H, W]
            
        Returns:
            tuple: (几何特征图, 回传特征列表)
        """
        # 从缓存获取跳跃连接
        skip_features = self._skip_features_cache
        
        # 确保输入是4D张量 [B, C, H, W]
        input_dim = features.dim()
        if input_dim == 3:
            features = features.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
        # 如果输入是多通道，取均值或第一个通道
        if features.shape[1] > 1:
            feature_map = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            feature_map = features  # [B, 1, H, W]
            
        # 应用边缘检测
        grad_h = F.conv2d(feature_map, self.horizontal_kernel, padding=1)
        grad_v = F.conv2d(feature_map, self.vertical_kernel, padding=1)
        
        # 计算梯度幅度
        edge_magnitude = torch.sqrt(grad_h**2 + grad_v**2)
        
        # 应用对角线检测
        diag1 = F.conv2d(feature_map, self.diag1_kernel, padding=1)
        diag2 = F.conv2d(feature_map, self.diag2_kernel, padding=1)
        
        # 应用TAD结构检测
        tad_feature = F.conv2d(feature_map, self.tad_kernel, padding=1)
        
        # 基础几何特征
        base_features = [edge_magnitude, grad_h, grad_v, diag1, diag2]
        
        # 处理所有跳跃连接
        skip_processed = []
        for i, skip in enumerate(skip_features[:3]):
            # 调整跳跃连接特征的尺寸以匹配当前特征图
            if skip.shape[2:] != feature_map.shape[2:]:
                skip = F.interpolate(skip, size=feature_map.shape[2:], 
                                   mode='nearest')  # 修改为最近邻插值
            # 处理跳跃连接特征
            skip_processed.append(self.skip_processors[i](skip))
        
        # 拼接所有特征
        all_features = torch.cat(base_features + skip_processed, dim=1)
        
        # 特征融合
        geometric_features = self.fusion(all_features)
        
        # 3. 生成回传特征 - 确保与Transformer层的正确连接
        feedback_features = []
        for i, skip in enumerate(skip_features[:3]):
            feedback = self.feedback_projectors[i](skip)
            feedback_features.append(feedback)
        
        # 如果原始输入是3D，则移除批次维度
        if input_dim == 3:
            geometric_features = geometric_features.squeeze(0)
            
        return geometric_features, feedback_features
        
    def get_attention_bias(self, x_shape, device):
        """生成几何引导的注意力偏置
        
        Args:
            x_shape: 输入张量形状
            device: 计算设备
            
        Returns:
            几何注意力偏置矩阵
        """
        B, L, D = x_shape
        # 生成默认的方形偏置 (如果没有缓存的几何特征)
        if not hasattr(self, '_geo_attention_cache'):
            return torch.zeros(B, L, L, device=device)
            
        # 使用缓存的几何特征构建注意力偏置
        geo_feats = self._geo_attention_cache
        if isinstance(geo_feats, torch.Tensor):
            # 将一维特征重塑为二维张量
            H = W = int(math.sqrt(L))
            geo_2d = geo_feats.view(B, H, W)
            
            # 构建注意力偏置矩阵
            bias = torch.zeros(B, L, L, device=device)
            
            # 对每个位置，根据其几何特征值影响周围注意力
            for i in range(L):
                row, col = i // W, i % W
                # 获取周围的位置
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            j = nr * W + nc
                            # 增强TAD边界位置对周围的注意力
                            bias[:, i, j] = geo_2d[:, nr, nc] * self.geo_attention_weight
            
            return bias
        
        return torch.zeros(B, L, L, device=device) 