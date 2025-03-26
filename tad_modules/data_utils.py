import os
import re
import numpy as np
from collections import defaultdict

def find_chromosome_files(output_root="./tad_results") -> list[str]:
    """遍历输出目录获取染色体文件路径"""
    pattern = re.compile(r'^chr\d+$')  # 匹配chr开头+数字的目录
    file_paths = []
    for root, dirs, files in os.walk(output_root):
        # 匹配染色体目录
        dir_name = os.path.basename(root)
        if pattern.match(dir_name):
            # 构建预期文件名
            target_file = f"{dir_name}.txt"
            if target_file in files:
                file_paths.append(os.path.join(root, target_file))
    return sorted(file_paths)

def fill_hic(hic_path: str, resolution: int) -> tuple[np.ndarray, int]:
    """填充HiC矩阵
    
    Args:
        hic_path: HiC数据文件路径
        resolution: 分辨率
        
    Returns:
        filled_matrix: 填充后的矩阵
        matrix_size: 矩阵大小
    """
    # 优化1：使用完整对称矩阵存储格式
    coord_dict = defaultdict(float)
    max_idx = 0
    with open(hic_path, 'r') as f:
        for line in f:
            i, j, val = line.strip().split()
            i, j = int(i)-1, int(j)-1  # 不再排序坐标
            # 同时记录原始坐标和对称坐标
            coord_dict[(i, j)] += float(val)
            coord_dict[(j, i)] += float(val)  # 新增对称记录
            max_idx = max(max_idx, i, j)
    
    matrix_size = max_idx + 1
    filled_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    
    # 批量填充完整矩阵
    for (i, j), val in coord_dict.items():
        filled_matrix[i, j] = val  # 直接填充原始坐标
            
    return filled_matrix 