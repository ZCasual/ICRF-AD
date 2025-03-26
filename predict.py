import os
import sys
import torch
import logging
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import traceback
from collections import defaultdict

# 导入detect_tad.py中的模型和函数
from net1a import (
    AVIT_DINO,
    find_chromosome_files,
    fill_hic,
)

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TAD-Predictor")

# 全局配置
output_root = "./tad_results"
resolution = 10000

class TADPredictor:
    """TAD预测器：使用预训练的AVIT_DINO模型预测TAD边界"""
    
    def __init__(self, model_path=None, device='cuda', resolution=10000, 
                 min_tad_size=5, nms_threshold=0.3, score_threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.resolution = resolution
        self.filepath = None
        
        # TAD预测配置
        self.min_tad_size = min_tad_size  # 最小TAD大小（bin数）
        self.nms_threshold = nms_threshold  # NMS阈值
        self.score_threshold = score_threshold  # 检测置信度阈值
        
        # 加载模型
        self.model = None
        self.edge_detector = None

    def set_filepath(self, filepath):
        """设置当前处理的文件路径"""
        self.filepath = filepath
        
    def set_resolution(self, resolution):
        """设置分辨率（单位：bp）"""
        self.resolution = resolution

    def load_model(self, chr_name=None):
        """加载预训练模型"""
        # 如果未指定模型路径但指定了染色体，尝试加载该染色体的默认模型
        if self.model_path is None and chr_name is not None:
            self.model_path = os.path.join(output_root, chr_name, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            # 加载模型参数
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 从检查点提取模型参数
            model_params = {}
            if 'model_params' in checkpoint:
                model_params = checkpoint['model_params']
            
            # 创建模型实例
            self.model = AVIT_DINO(
                embed_dim=model_params.get('embed_dim', 64),
                patch_size=model_params.get('patch_size', 8),
                num_layers=model_params.get('num_layers', 15),
                num_heads=model_params.get('num_heads', 4),
                use_amp=model_params.get('use_amp', True),
                ema_decay=model_params.get('ema_decay', 0.996),
                mask_ratio=model_params.get('mask_ratio', 0.3),
                gamma_base=model_params.get('gamma_base', 0.01),
                epsilon_base=model_params.get('epsilon_base', 0.05),
                use_theory_gamma=model_params.get('use_theory_gamma', True),
                boundary_weight=model_params.get('boundary_weight', 0.3)
            ).to(self.device)
            
            # 加载网络权重 - 尝试加载所有可能的组件
            for key in checkpoint:
                if key in ['teacher', 'student', 'teacher_bilstm', 'student_unet']:
                    try:
                        if hasattr(self.model, key):
                            getattr(self.model, key).load_state_dict(checkpoint[key])
                            logger.info(f"成功加载 {key} 模型")
                        else:
                            logger.warning(f"模型中没有 {key} 组件")
                    except Exception as e:
                        logger.warning(f"加载 {key} 时出错: {str(e)}")
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"成功加载模型: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            traceback.print_exc()
            return False

    def save_results(self, bed_entries, chr_name):
        """保存结果到BED文件"""
        # 确保染色体目录存在
        chr_dir = Path(output_root) / chr_name
        chr_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建完整的BED文件路径
        bed_path = chr_dir / f"{chr_name}_tad.bed"
        with open(bed_path, 'w') as f:
            f.write("\n".join(bed_entries))
        logger.info(f"TAD结果已保存到: {bed_path}")
        return str(bed_path)

    def predict(self, matrix):
        """预测TAD边界 - 使用预训练模型"""
        # 确保矩阵是NumPy数组
        if torch.is_tensor(matrix):
            matrix = matrix.detach().cpu().numpy()
        
        # 获取染色体名称
        chr_name = self._get_chr_name()
        
        # 加载模型
        if self.model is None:
            success = self.load_model(chr_name)
            if not success:
                logger.error(f"无法加载模型，返回空结果: {self.model_path}")
                return [], chr_name
        
        # 将矩阵转换为张量
        logger.info("通过预训练模型提取TAD边界...")
        matrix_tensor = torch.tensor(matrix, device=self.device).float()
        
        # 确保矩阵形状合适，如果需要调整大小
        orig_size = matrix.shape[0]
        if orig_size % 16 != 0:  # 如果不是16的倍数
            pad_size = ((orig_size // 16) + 1) * 16
            padded_matrix = np.zeros((pad_size, pad_size))
            padded_matrix[:orig_size, :orig_size] = matrix
            matrix_tensor = torch.tensor(padded_matrix, device=self.device).float()
            logger.info(f"调整矩阵大小从 {orig_size}x{orig_size} 到 {pad_size}x{pad_size}")
        
        # 通过模型提取特征和边界
        with torch.no_grad():
            # 直接把矩阵输入模型
            outputs = self.model(matrix_tensor)
            
            # 获取边界预测结果
            if isinstance(outputs, dict) and 'teacher_boundary_probs' in outputs:
                boundary_probs = outputs['teacher_boundary_probs']
            elif hasattr(self.model, 'get_boundary_predictions'):
                # 使用边界预测专用方法
                boundary_preds = self.model.get_boundary_predictions(matrix_tensor)
                boundary_probs = boundary_preds.get('boundary_probs')
            else:
                logger.error("模型输出中没有边界预测结果")
                return [], chr_name
        
        # 处理边界概率，将张量转换为numpy数组
        if boundary_probs is not None:
            if torch.is_tensor(boundary_probs):
                boundary_probs = boundary_probs.cpu().numpy()
            
            # 根据边界概率提取TAD边界
            if len(boundary_probs.shape) > 1 and boundary_probs.shape[0] == 1:
                boundary_probs = boundary_probs[0]  # 移除批次维度
            
            # 截取到原始大小
            if boundary_probs.shape[0] > orig_size:
                boundary_probs = boundary_probs[:orig_size]
            
            # 识别TAD
            tads = []
            # 寻找边界概率的峰值作为TAD边界
            peaks = []
            
            # 使用固定阈值而非统计阈值
            threshold = self.score_threshold  # 默认是0.5
            
            # 找出所有峰值
            for i in range(2, len(boundary_probs)-2):
                if (boundary_probs[i] > boundary_probs[i-1] and 
                    boundary_probs[i] > boundary_probs[i-2] and
                    boundary_probs[i] > boundary_probs[i+1] and
                    boundary_probs[i] > boundary_probs[i+2] and
                    boundary_probs[i] > threshold):
                    peaks.append(i)
            
            # 添加起始点和终止点
            all_boundaries = [0] + peaks + [orig_size-1]
            
            # 生成TAD区域
            for i in range(len(all_boundaries)-1):
                start = all_boundaries[i]
                end = all_boundaries[i+1]
                
                # 转换为bp
                start_bp = int(start * self.resolution)
                end_bp = int(end * self.resolution)
                
                # 确保TAD大小大于阈值 - 添加最小大小检查(30000bp)
                if end - start >= self.min_tad_size and (end_bp - start_bp) >= 100000:  # 至少5个bin且大于30000bp
                    # 计算区域内平均边界概率作为TAD质量评分
                    region_score = float(np.mean(boundary_probs[start:end]))
                    tads.append((start_bp, end_bp, region_score))
            
            # 创建BED格式条目
            bed_entries = self._create_bed_entries(tads)
            logger.info(f"找到 {len(tads)} 个潜在TAD边界")
            return bed_entries, chr_name
        else:
            logger.error("无法从模型获取边界概率")
            return [], chr_name

    def _create_bed_entries(self, tads):
        """生成BED条目"""
        bed_entries = []
        chr_name = self._get_chr_name()
        
        for i, tad_info in enumerate(tads):
            if len(tad_info) == 2:
                start_bp, end_bp = tad_info
                score = 1000  # 默认分数
            else:
                start_bp, end_bp, region_score = tad_info
                # 将区域评分转换为合适的BED分数(0-1000)
                score = min(1000, int(region_score * 1000))
            
            # 创建BED条目
            bed_entries.append(f"{chr_name}\t{start_bp}\t{end_bp}\tTAD_{i}\t{score}")
        
        logger.info(f"生成了 {len(bed_entries)} 个TAD条目")
        return bed_entries

    def _get_chr_name(self):
        """从文件路径获取染色体名称"""
        if self.filepath:
            path = Path(self.filepath)
            # 获取父目录名称，这应该是完整的染色体名称
            chr_name = path.parent.name
            return chr_name
        return "chr"  # 默认值

def main():
    """主函数：加载数据并执行TAD预测"""
    # 查找所有染色体文件
    logger.info("正在搜索Hi-C数据文件...")
    
    hic_paths = find_chromosome_files(output_root)
    if not hic_paths:
        logger.error("未找到Hi-C数据文件")
        return
    
    logger.info(f"找到 {len(hic_paths)} 个染色体数据文件")
    
    # 显示各文件路径
    for i, path in enumerate(hic_paths):
        logger.info(f"文件 {i+1}: {path}")
    
    # 对每个染色体执行预测
    saved_beds = []
    for hic_idx, hic_path in enumerate(hic_paths):
        try:
            # 从文件路径获取染色体名称
            hic_path_obj = Path(hic_path)
            chr_name = hic_path_obj.parent.name
            
            logger.info(f"处理染色体 {hic_idx+1}/{len(hic_paths)}: {chr_name}")
            
            # 检查模型文件路径
            model_path = str(hic_path_obj.parent / "best_model.pth")
            if not Path(model_path).exists():
                logger.warning(f"模型文件不存在: {model_path}，将尝试使用其他方法提取特征")
                model_path = None
            else:
                logger.info(f"找到模型文件: {model_path}")
            
            # 创建预测器并设置文件路径和模型路径
            predictor = TADPredictor(model_path=model_path)
            predictor.set_filepath(hic_path)
            
            # 加载Hi-C矩阵
            logger.info(f"正在加载Hi-C矩阵: {hic_path}")
            matrix = fill_hic(hic_path, resolution)
            logger.info(f"矩阵尺寸: {matrix.shape[0]}x{matrix.shape[1]}")
            
            if matrix.shape[0] < 5:
                logger.warning(f"矩阵太小 ({matrix.shape})，跳过 {hic_path}")
                continue
                
            # 执行TAD预测
            logger.info(f"开始TAD预测...")
            bed_entries, chr_name = predictor.predict(matrix)
            
            # 保存结果
            logger.info(f"保存结果...")
            bed_path = predictor.save_results(bed_entries, chr_name)
            saved_beds.append(bed_path)
            logger.info(f"完成染色体 {chr_name} 的处理")
            
        except Exception as e:
            logger.error(f"处理文件时出错 {hic_path}: {e}")
            traceback.print_exc()
            
    logger.info(f"所有染色体处理完成! 生成了 {len(saved_beds)} 个BED文件")
    if saved_beds:
        logger.info(f"BED文件路径: {saved_beds}")

if __name__ == "__main__":
    main() 