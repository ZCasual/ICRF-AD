import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

# 导入模块
from tad_modules import (LowRank, GeometricFeatureExtractor, AVIT, 
                        EdgeAwareBiLSTM, SimplifiedUNet, 
                        find_chromosome_files, fill_hic)  # 从模块导入函数

# 创建基础配置类
class TADBaseConfig:
    """基础配置类：集中管理所有共享参数"""
    def __init__(self):
        # 基本环境参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_root = "./tad_results"
        self.resolution = 10000
        
        # 模型结构参数
        self.patch_size = 8
        self.embed_dim = 64
        self.num_layers = 15
        self.num_heads = 4
        
        # 训练参数
        self.use_amp = (self.device == "cuda")
        self.ema_decay = 0.996
        self.mask_ratio = 0.3
        self.gamma_base = 0.01
        self.epsilon_base = 0.05
        self.use_theory_gamma = True
        self.boundary_weight = 0.3
        self.num_epochs = 40

    def get_model_params(self):
        """获取模型相关参数字典"""
        return {
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'use_amp': self.use_amp,
            'ema_decay': self.ema_decay,
            'mask_ratio': self.mask_ratio,
            'gamma_base': self.gamma_base,
            'epsilon_base': self.epsilon_base,
            'use_theory_gamma': self.use_theory_gamma,
            'boundary_weight': self.boundary_weight,
        }

"""
以下模块已迁移到 (tad_modules)
0. 数据加载
1. CUR特征分解
2. 几何特征提取模块
3. Backbone: AVIT
4. BiLSTM + U-Net
"""

"""
5. A-VIT型DINO自监督框架
"""

class AdversarialTAD(nn.Module):
    """对抗学习框架（完整实现）"""
    def __init__(self, embed_dim=64, freeze_ratio=0.75):
        super().__init__()
        # 生成器 (分割网络)
        self.generator = SimplifiedUNet(1, 1, adv_mode=True)
        
        # 判别器 (边界检测+真实性判断)
        self.discriminator = EdgeAwareBiLSTM(
            input_dim=1,  # 输入为生成器的单通道输出
            hidden_dim=32,
            with_classifier=True
        )
        
        # 冻结参数机制
        self._frozen_params = set()
        self._freeze_parameters(freeze_ratio)
        
        # 对抗训练参数
        self.adv_weight = 0.1
        
    def _freeze_parameters(self, ratio):
        """智能冻结机制，优先冻结深层参数"""
        params = []
        # 按网络深度收集参数
        params += list(self.generator.enc3.parameters())  # 深层
        params += list(self.generator.enc2.parameters())
        params += list(self.generator.enc1.parameters())  # 浅层
        
        # 计算需要冻结的参数数量
        total = sum(p.numel() for p in params)
        freeze_num = int(total * ratio)
        
        # 从深层到浅层冻结
        frozen = 0
        for p in params:
            if frozen < freeze_num:
                p.requires_grad = False
                self._frozen_params.add(p)
                frozen += p.numel()
        print(f"冻结参数比例: {frozen/total:.1%}")
    
    def forward(self, matrix):
        # 统一处理输入
        if matrix.dim() == 4:  # [B,C,H,W]
            B, C, H, W = matrix.shape
            if C != 1:
                matrix = matrix.mean(1, keepdim=True)  # 合并通道
        elif matrix.dim() == 3:  # [B,H,W]
            matrix = matrix.unsqueeze(1)  # [B,1,H,W]
        elif matrix.dim() == 2:  # [H,W]
            matrix = matrix.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        
        # 确保输入连续
        matrix = matrix.contiguous()
        
        # 生成器前向处理
        gen_seg = self.generator(matrix)  # [B,1,H,W]
        
        # 确保生成器输出连续
        gen_seg = gen_seg.contiguous()
        
        # 直接处理完整输出，避免不必要的分块
        B, C, H, W = gen_seg.shape
        
        # 整形为序列 [B,1,H*W]
        seq_features = gen_seg.view(B, C, H*W)
        seq_features = seq_features.contiguous()  # 显式确保连续
        
        # 判别器处理
        _, _, real_prob = self.discriminator(matrix)
        _, _, fake_prob = self.discriminator(seq_features, is_sequence=True)
        
        return {
            'gen_seg': gen_seg,
            'real_prob': real_prob,
            'fake_prob': fake_prob
        }

    def _merge_results(self, results):
        """合并分块结果"""
        merged = {
            'gen_seg': torch.cat([r['gen_seg'] for r in results], dim=1),
            'real_prob': torch.cat([r['real_prob'] for r in results], dim=0),
            'fake_prob': torch.cat([r['fake_prob'] for r in results], dim=0)
        }
        return merged

    def load_pretrained(self, pretrain_path):
        """加载预训练参数（兼容AVIT参数格式）"""
        pretrained_dict = torch.load(pretrain_path)
        
        # 参数名转换映射
        name_mapping = {
            'student_unet': 'generator',
            'teacher_bilstm': 'discriminator'
        }
        
        # 转换参数名称并加载
        model_dict = self.state_dict()
        for old_name, new_name in name_mapping.items():
            for k in list(pretrained_dict.keys()):
                if k.startswith(old_name):
                    model_k = k.replace(old_name, new_name)
                    if model_k in model_dict:
                        model_dict[model_k] = pretrained_dict[k]
        
        self.load_state_dict(model_dict)
        print("成功加载预训练参数，保持冻结状态")

    def _compute_loss(self, outputs, real_labels):
        """增强的损失计算，添加TAD长度约束"""
        # 获取生成结果
        gen_seg = outputs['gen_seg']  # [B,1,H,W]
        
        # 原有损失计算
        gen_loss = F.binary_cross_entropy(gen_seg, real_labels)
        
        # 对抗损失
        adv_loss = F.binary_cross_entropy(
            outputs['fake_prob'], 
            torch.ones_like(outputs['fake_prob'])
        )
        
        # ----- 新增TAD长度约束 -----
        # 1. 计算标签中的TAD长度分布
        tad_sizes = self._compute_tad_sizes(real_labels)
        mean_size, std_size = tad_sizes['mean'], tad_sizes['std']
        
        # 2. 计算生成结果中的TAD长度分布
        gen_tad_sizes = self._compute_tad_sizes(gen_seg)
        gen_mean, gen_std = gen_tad_sizes['mean'], gen_tad_sizes['std']
        
        # 3. 对生成的TAD中过短的区域施加惩罚
        if gen_mean < mean_size * 0.5:  # 如果平均长度小于参考长度一半
            size_penalty = torch.exp(-gen_mean / (mean_size * 0.25)) * 0.2
        else:
            size_penalty = torch.tensor(0.0, device=gen_seg.device)
            
        # 4. 对长度方差大的情况施加轻微惩罚(鼓励均匀性)
        variance_penalty = torch.clamp(gen_std / (mean_size * 0.5), 0, 1) * 0.1
        
        # 汇总所有损失
        total_loss = gen_loss + self.adv_weight * adv_loss + size_penalty + variance_penalty
        
        return {
            'total': total_loss,
            'gen': gen_loss,
            'adv': adv_loss,
            'size_penalty': size_penalty,
            'variance_penalty': variance_penalty
        }
        
    def _compute_tad_sizes(self, segmentation):
        """计算TAD尺寸统计信息"""
        # 预处理分割结果
        seg_binary = (segmentation > 0.5).float()
        B, C, H, W = seg_binary.shape
        
        # 连通区域分析
        sizes = []
        areas = []
        
        # 对每个批次处理
        for b in range(B):
            img = seg_binary[b, 0].cpu().numpy()
            
            # 计算行差异 (简单近似连通区域)
            row_diff = np.abs(np.diff(img, axis=0))
            # 找出变化点 (TAD边界)
            boundaries = np.where(row_diff > 0.5)[0]
            
            if len(boundaries) > 1:
                # 计算相邻边界间距离 (TAD大小)
                tad_sizes = np.diff(boundaries)
                sizes.extend(tad_sizes.tolist())
                
                # 计算区域面积
                for i in range(len(boundaries)-1):
                    start, end = boundaries[i], boundaries[i+1]
                    area = np.sum(img[start:end, :])
                    areas.append(area)
        
        # 计算统计信息
        if len(sizes) > 0:
            mean_size = torch.tensor(np.mean(sizes), device=segmentation.device)
            std_size = torch.tensor(np.std(sizes), device=segmentation.device)
            max_size = torch.tensor(np.max(sizes), device=segmentation.device)
            min_size = torch.tensor(np.min(sizes), device=segmentation.device)
        else:
            # 防止空列表
            mean_size = torch.tensor(H/10, device=segmentation.device)  # 默认期望值
            std_size = torch.tensor(H/20, device=segmentation.device)
            max_size = torch.tensor(H/5, device=segmentation.device)
            min_size = torch.tensor(H/20, device=segmentation.device)
            
        return {
            'mean': mean_size,
            'std': std_size,
            'max': max_size,
            'min': min_size,
            'count': len(sizes)
        }

class TADFeatureExtractor(TADBaseConfig):
    """TAD特征提取器：从HiC数据中提取特征矩阵"""
    
    def __init__(self, use_cur=True, **kwargs):
        # 调用父类初始化
        super().__init__(**kwargs)
        self.use_cur = use_cur
        self.model = None
        self.cur_projector = None
    
    def load_model(self, model_path=None, chr_name=None):
        """加载预训练模型"""
        # 如果未指定模型路径但指定了染色体，尝试加载该染色体的默认模型
        if model_path is None and chr_name is not None:
            model_path = os.path.join(self.output_root, chr_name, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载模型参数
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 从检查点提取模型参数
            if 'model_params' in checkpoint:
                for key in ['patch_size', 'embed_dim', 'num_layers', 'num_heads']:
                    if key in checkpoint['model_params']:
                        setattr(self, key, checkpoint['model_params'][key])
            
            # 创建模型实例 - 只使用教师模型用于特征提取
            model_params = self.get_model_params()
            self.model = AVIT(
                embed_dim=model_params['embed_dim'],
                patch_size=model_params['patch_size'],
                num_layers=model_params['num_layers'],
                num_heads=model_params['num_heads'],
                use_amp=model_params['use_amp']
            ).to(self.device)
            
            # 只加载教师模型状态字典
            if 'teacher' in checkpoint:
                # 使用教师模型替代学生模型进行特征提取
                self.model.load_state_dict(checkpoint['teacher'])
            elif 'student' in checkpoint:
                # 如果没有教师模型，使用学生模型
                self.model.load_state_dict(checkpoint['student'])
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"成功加载模型: {model_path}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
    
    def preprocess(self, matrix):
        """预处理HiC矩阵"""
        if self.use_cur:
            # 如果没有CUR投影器，创建一个
            if self.cur_projector is None:
                self.cur_projector = LowRank(p=0.7, alpha=0.7)
            
            # 应用CUR分解
            cur_matrix = self.cur_projector.fit_transform(matrix)
            return cur_matrix
        else:
            # 不使用CUR预处理，直接返回原矩阵
            return matrix
    
    def extract_features(self, hic_matrix, return_reconstruction=False):
        """
        从HiC矩阵中提取特征
        Args:
            hic_matrix: 输入的HiC矩阵
            return_reconstruction: 是否同时返回重建矩阵
        """
        # 确保模型已加载
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        # 预处理矩阵
        processed_matrix = self.preprocess(hic_matrix)
        
        # 将矩阵转换为tensor
        if isinstance(processed_matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(processed_matrix).float().to(self.device)
        else:
            matrix_tensor = processed_matrix.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            # 使用模型前向传播
            reconstructed, z_mean, z_logvar = self.model(matrix_tensor)
            
            # 获取编码器输出作为特征
            if hasattr(self.model, '_encoder_output_cache'):
                features = self.model._encoder_output_cache.cpu().numpy()
            else:
                # 如果没有缓存，返回空特征
                features = None
        
        # 准备返回结果
        result = {
            'features': features
        }
        
        if return_reconstruction:
            result['reconstructed'] = reconstructed.cpu().numpy()
        
        return result

# 创建全局特征提取器单例
_feature_extractor = None

def get_feature_extractor(**kwargs):
    """获取或创建全局TAD特征提取器"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = TADFeatureExtractor(**kwargs)
    return _feature_extractor

# 修改全局特征提取函数
def extract_features_from_hic(hic_matrix, chr_name=None, model_path=None, **kwargs):
    """
    从HiC矩阵中提取特征的简化函数 - 使用基类配置
    Returns:
        dict: 包含特征矩阵
    """
    # 创建临时特征提取器
    extractor = TADFeatureExtractor(**kwargs)
    # 确定模型路径
    if model_path is None and chr_name is not None:
        model_path = os.path.join(extractor.output_root, chr_name, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    extractor.load_model(model_path)
    # 提取特征
    result = extractor.extract_features(hic_matrix, return_reconstruction=True)
    return result
        
if __name__ == "__main__":
    print("请使用train.py进行模型训练")
  
  