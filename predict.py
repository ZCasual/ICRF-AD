import os
import sys
import torch
import logging
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import traceback
from collections import defaultdict

# 导入对抗学习模型和相关函数
from net1a import (
    AdversarialTAD,
    find_chromosome_files,
    fill_hic,
    TADBaseConfig
)

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TAD-Predictor")

class TADPredictor(TADBaseConfig):
    """TAD预测器：使用对抗学习模型预测TAD边界"""
    
    def __init__(self, model_path=None, device='cuda', resolution=10000, 
                 min_tad_size=5, nms_threshold=0.3, score_threshold=0.5):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.resolution = resolution
        self.filepath = None
        
        # TAD预测配置
        self.min_tad_size = min_tad_size  # 最小TAD大小（bin数）
        self.nms_threshold = nms_threshold  # NMS阈值
        self.score_threshold = score_threshold  # 检测置信度阈值
        
        # 模型实例
        self.model = None

    def set_filepath(self, filepath):
        """设置当前处理的文件路径"""
        self.filepath = filepath
        
    def load_model(self, chr_name=None):
        """加载预训练的对抗学习模型"""
        # 如果未指定模型路径但指定了染色体，尝试加载该染色体的默认模型
        if self.model_path is None and chr_name is not None:
            self.model_path = os.path.join(self.output_root, chr_name, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            # 创建对抗学习模型实例并启用结构约束
            self.model = AdversarialTAD(use_structure_constraints=True).to(self.device)
            
            # 加载模型参数
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 加载状态字典
            self.model.load_state_dict(checkpoint)
            
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
        chr_dir = Path(self.output_root) / chr_name
        chr_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建完整的BED文件路径
        bed_path = chr_dir / f"{chr_name}_tad.bed"
        with open(bed_path, 'w') as f:
            f.write("\n".join(bed_entries))
        logger.info(f"TAD结果已保存到: {bed_path}")
        return str(bed_path)

    def _process_matrix(self, matrix):
        """预处理矩阵为模型输入格式"""
        # 确保是Tensor并添加批次和通道维度
        if not torch.is_tensor(matrix):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        
        # 处理维度
        if matrix.dim() == 2:  # [H,W]
            matrix = matrix.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif matrix.dim() == 3:  # [B,H,W]
            matrix = matrix.unsqueeze(1)  # [B,1,H,W]
            
        return matrix.to(self.device)
    
    def predict(self, matrix):
        """使用对抗学习模型预测TAD边界"""
        # 获取染色体名称
        chr_name = self._get_chr_name()
        
        # 加载模型
        if self.model is None:
            success = self.load_model(chr_name)
            if not success:
                logger.error(f"无法加载模型，返回空结果")
                return [], chr_name
        
        # 预处理矩阵
        logger.info("通过对抗学习模型提取TAD边界...")
        orig_shape = matrix.shape
        matrix_tensor = self._process_matrix(matrix)
        
        # 使用模型进行预测
        with torch.no_grad():
            try:
                # 调用模型
                outputs = self.model(matrix_tensor)
                
                # 提取边界概率 - AdversarialTAD模型输出包含boundary_probs
                if 'boundary_probs' in outputs:
                    boundary_probs = outputs['boundary_probs'].cpu().numpy()
                    logger.info(f"获取到边界概率，形状: {boundary_probs.shape}")
                else:
                    # 如果无法直接获取边界概率，则计算边缘
                    gen_seg = outputs['gen_seg'].squeeze().cpu().numpy()
                    logger.info(f"生成分割结果，形状: {gen_seg.shape}")
                    
                    # 根据分割结果计算边界
                    if len(gen_seg.shape) > 1:
                        # 检测水平和垂直边缘
                        edge_h = np.abs(np.diff(gen_seg, axis=0))
                        edge_v = np.abs(np.diff(gen_seg, axis=1))
                        
                        # 创建边界图
                        boundary_map = np.zeros_like(gen_seg)
                        boundary_map[:-1, :] += edge_h
                        boundary_map[:, :-1] += edge_v
                        
                        # 沿行计算平均值得到一维边界概率
                        boundary_probs = np.mean(boundary_map, axis=1)
                    else:
                        # 直接计算一维序列的梯度
                        boundary_probs = np.abs(np.diff(gen_seg))
                        boundary_probs = np.pad(boundary_probs, (0, 1), 'constant')
                
                # 确保边界概率是一维数组
                if len(boundary_probs.shape) > 1:
                    boundary_probs = boundary_probs.reshape(-1)
                
                # 检测边界峰值
                peaks = self._detect_tad_boundaries(boundary_probs)
                
                # 生成TAD区域
                tads = self._generate_tad_regions(peaks, matrix.shape[0])
                
                # 创建BED格式条目
                bed_entries = self._create_bed_entries(tads)
                logger.info(f"找到 {len(tads)} 个TAD区域")
                return bed_entries, chr_name
                
            except Exception as e:
                logger.error(f"预测过程中出错: {str(e)}")
                traceback.print_exc()
                return [], chr_name
    
    def _detect_tad_boundaries(self, boundary_probs):
        """检测TAD边界峰值"""
        peaks = []
        threshold = self.score_threshold
        
        # 寻找局部峰值
        for i in range(2, len(boundary_probs)-2):
            if (boundary_probs[i] > boundary_probs[i-1] and 
                boundary_probs[i] > boundary_probs[i-2] and
                boundary_probs[i] > boundary_probs[i+1] and
                boundary_probs[i] > boundary_probs[i+2] and
                boundary_probs[i] > threshold):
                peaks.append((i, float(boundary_probs[i])))
        
        # 按位置排序
        peaks.sort(key=lambda x: x[0])
        return peaks
    
    def _generate_tad_regions(self, peaks, matrix_size):
        """从边界峰值生成TAD区域"""
        # 添加起始和结束边界
        all_boundaries = [(0, 1.0)] + peaks + [(matrix_size-1, 1.0)]
        
        # 生成TAD区域
        tads = []
        for i in range(len(all_boundaries)-1):
            start, start_score = all_boundaries[i]
            end, end_score = all_boundaries[i+1]
            
            # 转换为碱基对位置
            start_bp = int(start * self.resolution)
            end_bp = int(end * self.resolution)
            
            # 确保TAD尺寸足够大
            if (end - start >= self.min_tad_size) and (end_bp - start_bp >= 30000):
                # 区域评分 - 基于边界强度
                region_score = (start_score + end_score) / 2
                tads.append((start_bp, end_bp, region_score))
        
        return tads

    def _create_bed_entries(self, tads):
        """生成BED条目"""
        bed_entries = []
        chr_name = self._get_chr_name()
        
        for i, tad_info in enumerate(tads):
            start_bp, end_bp, region_score = tad_info
            # 将区域评分转换为BED分数(0-1000)
            score = min(1000, int(region_score * 1000))
            
            # 创建BED条目
            bed_entries.append(f"{chr_name}\t{start_bp}\t{end_bp}\tTAD_{i}\t{score}")
        
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
    # 创建预测器实例
    predictor = TADPredictor()
    
    # 查找所有染色体文件
    logger.info("正在搜索Hi-C数据文件...")
    
    hic_paths = find_chromosome_files(predictor.output_root)
    if not hic_paths:
        logger.error("未找到Hi-C数据文件")
        return
    
    logger.info(f"找到 {len(hic_paths)} 个染色体数据文件")
    
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
                logger.warning(f"模型文件不存在: {model_path}，尝试查找默认模型")
                model_path = None
            else:
                logger.info(f"找到模型文件: {model_path}")
            
            # 设置文件路径和模型路径
            predictor.set_filepath(hic_path)
            predictor.model_path = model_path
            
            # 加载Hi-C矩阵
            logger.info(f"正在加载Hi-C矩阵: {hic_path}")
            matrix = fill_hic(hic_path, predictor.resolution)
            logger.info(f"矩阵尺寸: {matrix.shape[0]}x{matrix.shape[1]}")
            
            if matrix.shape[0] < 5:
                logger.warning(f"矩阵太小 ({matrix.shape})，跳过处理")
                continue
                
            # 执行TAD预测
            logger.info(f"开始TAD预测...")
            bed_entries, chr_name = predictor.predict(matrix)
            
            # 保存结果
            if bed_entries:
                logger.info(f"保存结果...")
                bed_path = predictor.save_results(bed_entries, chr_name)
                saved_beds.append(bed_path)
                logger.info(f"完成染色体 {chr_name} 的处理")
            else:
                logger.warning(f"未找到TAD区域，跳过保存")
            
        except Exception as e:
            logger.error(f"处理文件时出错 {hic_path}: {e}")
            traceback.print_exc()
            
    logger.info(f"所有染色体处理完成! 生成了 {len(saved_beds)} 个BED文件")
    if saved_beds:
        logger.info(f"BED文件路径: {saved_beds}")

if __name__ == "__main__":
    main() 