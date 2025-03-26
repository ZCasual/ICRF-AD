"""
TAD分析结果可视化脚本
依赖ck-tad.py的输出文件结构：
输入目录结构：
tad_results/
└── chr1/
    ├── chr1.txt          # Hi-C交互矩阵
    └── chr1_tads.xml     # TAD边界信息
"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import logging
import psutil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm  # 导入tqdm进度条库
import gc  # 添加在现有导入区块中

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """可视化配置参数（新增resolution字段）"""
    input_dir: str = "./tad_results"
    output_dir: str = "./tad_visualization"
    chromosomes: List[str] = field(default_factory=list)
    dpi: int = 300
    max_memory_usage: float = 0.8
    resolution: int = 10000  # 新增分辨率配置参数

    def __post_init__(self):
        """路径正则化处理（保持原样）"""
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动检测染色体目录（当未指定时）
        if not self.chromosomes:
            # 获取所有以'chr'开头的子目录
            chrom_dirs = [d for d in self.input_dir.glob("chr*") if d.is_dir()]
            if chrom_dirs:
                self.chromosomes = sorted([d.name for d in chrom_dirs], 
                                         key=lambda x: int(re.search(r'\d+', x).group()))
                logger.info(f"自动检测到染色体目录: {self.chromosomes}")
            else:
                logger.warning(f"在输入目录 {self.input_dir} 中未找到任何染色体目录（chr*）")

@dataclass
class TADRegion:
    """存储单个TAD区域数据"""
    position: int      # 参考位置
    score: float       # TAD评分
    start_bin: int     # 起始bin索引
    end_bin: int       # 结束bin索引

@dataclass
class ChromosomeVisualizationData:
    """染色体可视化数据集"""
    chromosome: str
    matrix: np.ndarray          # Hi-C交互矩阵
    tad_regions: List[TADRegion]  # TAD区域列表，更改名称

@dataclass
class BlockIndex:
    """分块索引信息"""
    chrom: str
    start: int  # 起始bin
    end: int    # 结束bin
    block_size: int  # 分块尺寸
    overlap: int = 0  # 块间重叠

@dataclass
class RenderBlock:
    """待渲染区块数据"""
    matrix_block: np.ndarray
    tad_regions: List[TADRegion]
    block_info: BlockIndex

class VisualizationEngine:
    """可视化引擎，负责数据加载和绘图"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._current_mem = lambda: psutil.virtual_memory().percent / 100

    def _safe_load_matrix(self, txt_path: Path) -> Optional[np.ndarray]:
        """安全加载Hi-C矩阵（添加空文件检查）"""
        try:
            with open(txt_path) as f:
                max_bin = 0
                valid_lines = 0  # 新增有效行计数器
                for line in f:
                    # 跳过空行和注释行
                    if not line.strip() or line.startswith('#'):
                        continue
                    try:
                        i, j, val = map(int, line.strip().split())
                        max_bin = max(max_bin, i, j)
                        valid_lines += 1
                    except ValueError:
                        continue
                
                # 检查有效数据
                if valid_lines == 0 or max_bin == 0:
                    logger.error(f"空矩阵文件或无效数据: {txt_path}")
                    return None

                matrix_size = max_bin
                matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)

            # 重新读取文件填充矩阵（仅有效数据）
            with open(txt_path) as f:
                for line in f:
                    if not line.strip() or line.startswith('#'):
                        continue
                    try:
                        i, j, val = map(int, line.strip().split())
                        matrix[i-1, j-1] = val
                        matrix[j-1, i-1] = val
                    except ValueError:
                        continue

            return matrix
            
        except Exception as e:
            logger.error(f"加载矩阵失败: {str(e)}")
            return None

    def _parse_xml(self, xml_path: Path) -> List[TADRegion]:
        """解析TAD边界XML文件"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            tad_regions = []
            for boundary_elem in root.findall(".//Boundary"):
                tad_region = TADRegion(
                    position=int(boundary_elem.get("position")),
                    score=float(boundary_elem.get("score")),
                    start_bin=int(boundary_elem.get("start_bin")),
                    end_bin=int(boundary_elem.get("end_bin"))
                )
                tad_regions.append(tad_region)
            return tad_regions
            
        except Exception as e:
            logger.error(f"XML解析失败: {str(e)}")
            return []

    def _parse_bed(self, bed_path: Path, chromosome: str) -> List[TADRegion]:
        """解析TAD区域BED文件"""
        tad_regions = []
        try:
            with open(bed_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                if not lines:
                    logger.warning(f"空BED文件: {bed_path}")
                    return tad_regions
                
                resolution = self.config.resolution
                
                for line in lines:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    
                    # 直接使用BED坐标，无需调整
                    start_pos = int(parts[1])
                    end_pos = int(parts[2])
                    
                    # 与detect_tad.py一致的转换逻辑
                    start_bin = start_pos // resolution
                    end_bin = (end_pos - 1) // resolution  # BED结束坐标是exclusive，转换为inclusive
                    
                    # 添加得分处理
                    score = 0.0
                    if len(parts) >= 4:
                        score_str = parts[3].split('_')[-1]  # 从类似"TAD_0.75"中提取分数
                        try:
                            score = float(score_str)
                        except ValueError:
                            pass
                    
                    tad_regions.append(TADRegion(
                        position=start_pos,
                        score=score,
                        start_bin=start_bin,
                        end_bin=end_bin  # 直接使用转换后的bin值
                    ))
                
            return tad_regions
            
        except Exception as e:
            logger.error(f"BED解析失败: {str(e)}")
            return []

    def load_chromosome_data(self, chromosome: str) -> Optional[ChromosomeVisualizationData]:
        """加载单个染色体的可视化数据"""
        chrom_dir = self.config.input_dir / chromosome
        if not chrom_dir.exists():
            logger.warning(f"染色体目录不存在: {chrom_dir}")
            return None
            
        txt_path = chrom_dir / f"{chromosome}.txt"
        bed_path = chrom_dir / f"{chromosome}_tad.bed"
        
        if not txt_path.exists() or not bed_path.exists():
            logger.warning(f"缺失数据文件: {chromosome}")
            return None
            
        matrix = self._safe_load_matrix(txt_path)
        if matrix is None:
            return None
            
        tad_regions = self._parse_bed(bed_path, chromosome)  # 传入当前染色体名称
        return ChromosomeVisualizationData(
            chromosome=chromosome,  # 直接从参数获取染色体名称
            matrix=matrix,
            tad_regions=tad_regions
        )

    def visualize(self, data: ChromosomeVisualizationData):
        """执行可视化流程（强制分块处理）"""
        output_dir = self.config.output_dir / data.chromosome
        output_dir.mkdir(exist_ok=True)
        
        # 无论矩阵大小都使用分块处理
        self.visualize_large(data)

    def split_large_matrix(self, matrix: np.ndarray, chrom: str, block_size=2000) -> List[BlockIndex]:
        """将大矩阵分割为可处理的区块索引"""
        n = matrix.shape[0]
        blocks = []
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            blocks.append(BlockIndex(
                chrom=chrom,
                start=start,
                end=end,
                block_size=block_size
            ))
        return blocks

    def extract_block_data(self, full_data: ChromosomeVisualizationData, block: BlockIndex) -> RenderBlock:
        """提取指定区块的数据"""
        # 矩阵切片
        sub_matrix = full_data.matrix[block.start:block.end, block.start:block.end]
        
        # 边界过滤
        block_tad_regions = []
        for tad in full_data.tad_regions:
            # 处理跨区块边界
            if (tad.start_bin >= block.start) and (tad.end_bin <= block.end):
                # 完全在区块内
                adjusted_tad = TADRegion(
                    position=tad.position,
                    score=tad.score,
                    start_bin=tad.start_bin - block.start,
                    end_bin=tad.end_bin - block.start
                )
                block_tad_regions.append(adjusted_tad)
            elif (tad.start_bin < block.end) and (tad.end_bin > block.start):
                # 跨区块边界，分割处理
                new_start = max(tad.start_bin, block.start) - block.start
                new_end = min(tad.end_bin, block.end) - block.start
                adjusted_tad = TADRegion(
                    position=tad.position,
                    score=tad.score,
                    start_bin=new_start,
                    end_bin=new_end
                )
                block_tad_regions.append(adjusted_tad)
        
        return RenderBlock(
            matrix_block=sub_matrix,
            tad_regions=block_tad_regions,
            block_info=block
        )

    def render_block(self, block: RenderBlock, temp_dir: Path, global_vmax: float):
        """渲染单个区块（严格保持热图尺寸）"""
        plt.figure(figsize=(10, 10), dpi=self.config.dpi)
        
        # 新增精确尺寸控制
        plt.gca().set_position([0, 0, 1, 1])  # 占据整个画布
        plt.xlim(0, block.matrix_block.shape[0])
        plt.ylim(0, block.matrix_block.shape[1])
        
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=global_vmax)
        
        im = plt.imshow(np.log1p(block.matrix_block), 
                      cmap=cmap,
                      norm=norm,
                      aspect='auto', 
                      origin='lower',
                      extent=(0, block.matrix_block.shape[0], 0, block.matrix_block.shape[1]),  # 精确坐标映射
                      interpolation='none')

        # 绘制TAD区域
        for tad in block.tad_regions:
            # 使用原始TAD坐标
            bin_start = tad.start_bin
            bin_end = tad.end_bin
            
            # 直接使用原始bin坐标，但改变映射关系
            # 根据主对角线对称，交换X和Y坐标
            left = bin_start  
            right = bin_end
            bottom = bin_start
            top = bin_end
            
            # 绘制矩形边框
            plt.plot([left, right], [bottom, bottom], color='cyan', linewidth=1.5, alpha=0.8)  # 下边框
            plt.plot([left, right], [top, top], color='cyan', linewidth=1.5, alpha=0.8)       # 上边框
            plt.plot([left, left], [bottom, top], color='cyan', linewidth=1.5, alpha=0.8)    # 左边框
            plt.plot([right, right], [bottom, top], color='cyan', linewidth=1.5, alpha=0.8)  # 右边框
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # 加强边距控制
        
        output_path = temp_dir / f"{block.block_info.chrom}_{block.block_info.start}_{block.block_info.end}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='none')  # 透明背景
        plt.close()

    def merge_and_render(self, blocks: List[RenderBlock], final_output: Path, temp_dir: Path, data: ChromosomeVisualizationData, global_vmax: float):
        """合并渲染（精确对齐热图边界）"""
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=global_vmax)
        
        fig = plt.figure(figsize=(15, 12), dpi=self.config.dpi, facecolor=cmap(0))
        ax = fig.add_subplot(111)
        
        # 设置坐标系与原始矩阵一致
        total_size = data.matrix.shape[0]
        ax.set_xlim(0, total_size)
        ax.set_ylim(0, total_size)  # 不再翻转Y轴
        
        # 流式绘制每个区块
        for b in tqdm(blocks, desc="合并区块", unit="块"):
            img_path = temp_dir / f"{b.block_info.chrom}_{b.block_info.start}_{b.block_info.end}.png"
            img = plt.imread(img_path)
            
            # 正确设置区块位置（保持原始坐标）
            ax.imshow(img, extent=(
                b.block_info.start,
                b.block_info.end,
                b.block_info.start,  # y起始位置
                b.block_info.end     # y结束位置
            ), origin='lower')  # 设置图像原点在左下角
            
            del img
            gc.collect()
        
        # 绘制TAD区域
        logger.info("绘制TAD区域...")
        for tad in tqdm(data.tad_regions, desc="绘制TAD区域", unit="TAD"):
            # 使用原始TAD坐标
            bin_start = tad.start_bin
            bin_end = tad.end_bin
            
            # 直接使用原始bin坐标，但改变映射关系
            # 根据主对角线对称，交换X和Y坐标
            left = bin_start  
            right = bin_end
            bottom = bin_start
            top = bin_end
            
            # 蓝色虚线已被注释掉，如果需要恢复，请使用下方代码
            # plt.plot([left, right], [bottom, bottom], color='blue', linestyle='--', linewidth=1, alpha=0.7)
            # plt.plot([left, right], [top, top], color='blue', linestyle='--', linewidth=1, alpha=0.7)
            # plt.plot([left, left], [bottom, top], color='blue', linestyle='--', linewidth=1, alpha=0.7)
            # plt.plot([right, right], [bottom, top], color='blue', linestyle='--', linewidth=1, alpha=0.7)
        
        # 添加标注
        ax.set_title(f"TAD Visualization - {data.chromosome}", fontsize=14)
        
        # 创建与热图匹配的colorbar
        # 计算数据范围以匹配热图
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='log(contact + 1)')
        
        # 设置坐标轴标签
        ax.set_xlabel('Genomic Position (bin)', fontsize=12)
        ax.set_ylabel('Genomic Position (bin)', fontsize=12)
        
        # 设置坐标轴和图形背景色
        ax.set_facecolor(cmap(0))
        
        # 保存最终图像
        logger.info(f"保存最终图像到 {final_output}")
        plt.savefig(final_output, bbox_inches='tight', facecolor=cmap(0))
        plt.close()

    def visualize_large(self, data: ChromosomeVisualizationData):
        """执行可视化（添加矩阵有效性检查）"""
        # 在计算全局vmax前添加检查
        if data.matrix.size == 0 or np.all(data.matrix == 0):
            logger.error(f"染色体 {data.chromosome} 的矩阵数据无效，跳过可视化")
            return
        
        output_dir = self.config.output_dir / data.chromosome
        output_dir.mkdir(exist_ok=True)
        
        # 后续原有代码保持不变...
        global_vmax = np.log1p(np.nanmax(data.matrix))
        
        temp_dir = self.config.output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 步骤1：分块索引
            blocks = self.split_large_matrix(data.matrix, data.chromosome)
            
            # 步骤2-3：流式处理每个区块
            for i, block in enumerate(tqdm(blocks, 
                                        desc=f"处理{data.chromosome}区块", 
                                        unit="块",
                                        mininterval=0.1,  # 强制更频繁更新
                                        leave=True)):     # 完成后保留进度条
                # 提取单个区块数据
                render_block = self.extract_block_data(data, block)
                
                # 渲染并立即释放内存
                self.render_block(render_block, temp_dir, global_vmax)
                del render_block.matrix_block
                del render_block
                gc.collect()  # 强制垃圾回收
                
                # 每处理5个区块清理一次matplotlib缓存
                if i % 5 == 0:
                    plt.close('all')
            
            # 步骤4：重新加载区块信息进行合并（避免持有原始数据）
            block_files = sorted(temp_dir.glob(f"{data.chromosome}_*.png"))
            render_blocks = [
                RenderBlock(
                    matrix_block=None,  # 不保留矩阵数据
                    tad_regions=[],
                    block_info=BlockIndex(
                        chrom=data.chromosome,
                        start=int(f.stem.split("_")[1]),
                        end=int(f.stem.split("_")[2]),
                        block_size=2000
                    )
                ) for f in block_files
            ]
            
            # 步骤5：合并渲染（新增data参数）
            logger.info(f"合并渲染区块图像为最终结果...")
            final_path = self.config.output_dir / data.chromosome / f"{data.chromosome}_tad_final.png"
            self.merge_and_render(render_blocks, final_path, temp_dir, data, global_vmax)
            
        finally:
            # 清理临时文件
            for f in temp_dir.glob("*"):
                f.unlink()
            temp_dir.rmdir()
            plt.close('all')  # 确保所有matplotlib资源释放

def main():
    try:
        config = VisualizationConfig()
        engine = VisualizationEngine(config)
        
        logger.info("开始可视化流程...")
        for chrom in config.chromosomes:
            logger.info(f"正在处理染色体: {chrom}")
            data = engine.load_chromosome_data(chrom)
            if data is not None:
                engine.visualize(data)
                del data
                gc.collect()
                
        logger.info("可视化流程完成！")
        
    except Exception as e:
        logger.error(f"可视化流程失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
