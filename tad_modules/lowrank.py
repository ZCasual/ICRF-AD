import numpy as np
from scipy.sparse.linalg import svds

class LowRank:
    """基于tSVD的CUR分解核心类"""
    def __init__(self, p: float, alpha: float, lambda_reg: float = 1e-6):
        self.p = p            # 累积贡献率阈值
        self.alpha = alpha    # 混合权重系数
        self.lambda_reg = lambda_reg  # 正则化系数
        self.U_k = None       # 左奇异向量矩阵
        self.V_k = None       # 右奇异向量矩阵
        self.C = None         # 列采样矩阵
        self.R = None         # 行采样矩阵
        self.U = None         # 连接矩阵

    def _calc_leverage_scores(self, matrix: np.ndarray) -> tuple:
        """计算行列重要性分数（含稀疏性优化）"""
        u, s, vt = svds(matrix.astype(np.float64), k=min(matrix.shape)-1, which='LM', solver='arpack')   # 使用ARPACK的svds实现Lanczos算法 // 需要更清楚原理
        
        # 自动截断
        s_sorted = s[::-1]  # 将奇异值降序排列
        total = np.sum(s_sorted)
        cum_contrib = np.cumsum(s_sorted) / total  # 计算累积贡献率
        k = np.argmax(cum_contrib >= self.p) + 1   # 应用公式选择最小满足条件的秩
        
        # 获取截断后的奇异向量
        self.U_k = u[:, -k:]  # ARPACK返回奇异值升序排列
        self.V_k = vt[-k:, :]
        
        # 计算原始重要性分数
        col_scores = np.linalg.norm(self.V_k, axis=0)**2 / k
        row_scores = np.linalg.norm(self.U_k, axis=1)**2 / k
        
        # 混合局部接触频率 (优化点3)
        if self.alpha < 1:
            col_density = np.asarray(matrix.sum(axis=0)).ravel()
            row_density = np.asarray(matrix.sum(axis=1)).ravel()
            col_scores = self.alpha * col_scores + (1-self.alpha) * col_density/col_density.sum()
            row_scores = self.alpha * row_scores + (1-self.alpha) * row_density/row_density.sum()
        
        return col_scores, row_scores, k

    def _sampling_columns_rows(self, matrix: np.ndarray, col_scores: np.ndarray, 
                             row_scores: np.ndarray, k: int) -> None:
        """执行行列采样（含对称性优化）"""
        n = matrix.shape[0]
        # 修复采样数量计算，确保不超过矩阵维度
        c = r = min(int(2 * k * np.log(k + 1)), n-1)  # 使用min限制最大采样数
        
        # 列采样（无放回）
        col_indices = np.random.choice(n, size=c, replace=False, p=col_scores/col_scores.sum())
        self.C = matrix[:, col_indices]
        
        # 处理对称矩阵的特殊情况
        if np.allclose(matrix, matrix.T):
            self.R = self.C.T
        else:
            # 对行采样同样应用数量限制
            r = min(r, n-1)
            row_indices = np.random.choice(n, size=r, replace=False, p=row_scores/row_scores.sum())
            self.R = matrix[row_indices, :]

    def _compute_U_matrix(self, matrix: np.ndarray) -> None:
        """计算连接矩阵（含正则化）"""
        # 计算正则化伪逆
        C_pinv = np.linalg.pinv(self.C.T @ self.C + self.lambda_reg * np.eye(self.C.shape[1])) @ self.C.T
        # 修复维度不匹配问题：添加转置操作
        R_pinv = self.R.T @ np.linalg.pinv(self.R @ self.R.T + self.lambda_reg * np.eye(self.R.shape[0]))
        
        # 计算连接矩阵
        self.U = C_pinv @ matrix @ R_pinv

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        """执行完整CUR分解流程"""
        # 步骤1：计算重要性分数
        col_scores, row_scores, k = self._calc_leverage_scores(matrix)
        
        # 步骤2：行列采样
        self._sampling_columns_rows(matrix, col_scores, row_scores, k)
        
        # 步骤3：计算连接矩阵
        self._compute_U_matrix(matrix)
        
        # 构建近似矩阵并评估误差
        approx = self.C @ self.U @ self.R
        rel_error = np.linalg.norm(matrix - approx) / np.linalg.norm(matrix)
        print(f"[CUR] 相对误差: {rel_error:.2%}, 采样列数: {self.C.shape[1]}, 采样行数: {self.R.shape[0]}")
        
        return approx.astype(np.float32) 