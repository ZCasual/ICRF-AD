import torch
import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from net1a import (
    TADBaseConfig,
    LowRank,
    find_chromosome_files,
    fill_hic,
    AdversarialTAD
)
import torch.nn.functional as F

class TADPipelineAdversarial(TADBaseConfig):
    """TAD Detection Training Pipeline (migrated from original detect_tad.py)"""
    
    def __init__(self, resume_training=False, **kwargs):
        super().__init__(**kwargs)
        self.resume_training = resume_training
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize components and checkpoint configurations"""
        self.adj_graph = None
        self.normalized_matrix = None
        torch.backends.cuda.checkpoint_activations = False
        torch.backends.cuda.checkpoint_layers_n = 10

    def load_data(self):
        """Data loading process: Load Hi-C data and return contact matrix"""
        hic_paths = find_chromosome_files(self.output_root)
        if not hic_paths:
            raise FileNotFoundError("Hi-C data files not found")
        hic_path = hic_paths[0]
        chr_dir = os.path.dirname(hic_path)
        self.chr_name = os.path.basename(chr_dir)
        print(f"[DEBUG] Current chromosome: {self.chr_name}")
        matrix = fill_hic(hic_path, self.resolution)
        print(f"[DEBUG] matrix_sum = {np.sum(matrix)}")
        return matrix

    def train_model(self, cur_tensor):
        # 初始化模型
        self.adv_net = AdversarialTAD(freeze_ratio=0.75).to(self.device)
        
        # 使用分组优化器，不同组件使用不同学习率
        optimizer_params = [
            {'params': self.adv_net.generator.parameters(), 'lr': 5e-3},  # 生成器使用更高的学习率
            {'params': self.adv_net.discriminator.parameters(), 'lr': 2e-8}  # 判别器学习率低
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params)
        
        # 添加学习率调度器, 包含预热和衰减阶段
        warmup_epochs = 10  # 预热训练10个epoch
        total_epochs = self.num_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:  # 预热阶段，学习率保持较高
                return 1.0
            else:  # 正常训练阶段，学习率逐渐衰减
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 创建可视化输出目录
        vis_dir = os.path.join(self.output_root, self.chr_name, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 使用更小的块大小减少内存占用
        tile_size = 128
        tiles = self._split_into_tiles(cur_tensor, tile_size)
        
        # 用于记录每个epoch的最佳生成结果
        best_gen_outputs = None
        
        # 分批训练，每批仅处理一个tile
        with tqdm(range(self.num_epochs), desc="Training Progress") as epoch_pbar:
            for epoch in epoch_pbar:
                # 确定当前是否处于预热期
                is_warmup = epoch < warmup_epochs
                warmup_factor = 1.0 if is_warmup else 0.0
                
                # 用于跟踪当前epoch的平均损失
                epoch_losses = {'total': 0.0, 'gen': 0.0, 'adv': 0.0, 
                               'size_penalty': 0.0, 'variance_penalty': 0.0,
                               'warmup': 0.0, 'global': 0.0}
                
                # 内部进度条跟踪tile处理
                with tqdm(enumerate(tiles), total=len(tiles), desc=f"{'Warmup' if is_warmup else 'Normal'} Epoch {epoch+1}/{self.num_epochs}") as tile_pbar:
                    for i, tile in tile_pbar:
                        # 确保输入连续
                        tile = tile.contiguous()
                        
                        # 前向传播
                        outputs = self.adv_net(tile)
                        
                        # 创建标签并计算损失
                        real_labels = torch.sigmoid(tile)
                        
                        # 添加预热阶段特殊处理
                        if is_warmup:
                            losses = self._compute_warmup_losses(outputs, real_labels, epoch, warmup_epochs)
                        else:
                            losses = self._compute_losses(outputs, real_labels)
                        
                        # 反向传播
                        self.optimizer.zero_grad()
                        losses['total'].backward()
                        self.optimizer.step()
                        
                        # 更新内部进度条
                        status_dict = {
                            'loss': f"{losses['total'].item():.4f}",
                            'gen': f"{losses['gen'].item():.4f}",
                            'adv': f"{losses['adv'].item():.4f}"
                        }
                        
                        # 根据当前阶段添加额外信息
                        if is_warmup:
                            status_dict['global'] = f"{losses.get('global', torch.tensor(0.0)).item():.4f}"
                        else:
                            status_dict['size'] = f"{losses.get('size_penalty', torch.tensor(0.0)).item():.4f}"
                        
                        tile_pbar.set_postfix(status_dict)
                        
                        # 保存第一个tile的输出用于可视化
                        if i == 0:
                            best_gen_outputs = {
                                'tile': tile.detach(),
                                'gen_seg': outputs['gen_seg'].detach(),
                                'real_prob': outputs['real_prob'].detach(),
                                'fake_prob': outputs['fake_prob'].detach()
                            }
                        
                        # 累积损失
                        for k in epoch_losses:
                            if k in losses:
                                epoch_losses[k] += losses[k].item()
                
                # 计算当前epoch的平均损失
                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k] /= len(tiles)
                
                # 更新学习率
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                
                # 更新外部进度条
                status_dict = {
                    'lr': f"{current_lr:.6f}",
                    'loss': f"{epoch_losses['total']:.4f}",
                    'gen': f"{epoch_losses['gen']:.4f}"
                }
                
                if is_warmup:
                    status_dict['global'] = f"{epoch_losses.get('global', 0.0):.4f}"
                else:
                    status_dict['adv'] = f"{epoch_losses['adv']:.4f}"
                    
                epoch_pbar.set_postfix(status_dict)
                
                # 每5个epoch可视化一次
                if epoch % 5 == 0 and best_gen_outputs is not None:
                    self._visualize_results(best_gen_outputs, epoch, vis_dir)
                
                # 每10个epoch保存模型
                if (epoch + 1) % 10 == 0:
                    model_path = os.path.join(self.output_root, self.chr_name, f"adv_model_epoch{epoch+1}.pth")
                    torch.save(self.adv_net.state_dict(), model_path)
            
            # 保存最终模型
            model_path = os.path.join(self.output_root, self.chr_name, "best_model.pth")
            torch.save(self.adv_net.state_dict(), model_path)

    def _compute_losses(self, outputs, real_labels):
        """整合网络内部和外部损失"""
        # 使用AdversarialTAD内部的损失计算
        losses = self.adv_net._compute_loss(outputs, real_labels)
        
        # 提取额外的尺度信息
        if hasattr(self.adv_net.generator, 'scale_info'):
            scale_info = self.adv_net.generator.scale_info
            
            # 记录尺度权重用于可视化
            if not hasattr(self, 'scale_weight_history'):
                self.scale_weight_history = []
            
            self.scale_weight_history.append(scale_info['weights'].detach().cpu())
            
            # 每10个epoch可视化尺度分布
            if len(self.scale_weight_history) % 10 == 0:
                self._visualize_scale_weights()
        
        return losses

    def _compute_warmup_losses(self, outputs, real_labels, epoch, warmup_epochs):
        """预热阶段的损失计算，强制关注长区域TAD，减少细节处理"""
        # 获取基本损失
        gen_seg = outputs['gen_seg']  # [B,1,H,W]
        
        # 计算当前预热进度（0-1范围）
        warmup_progress = epoch / warmup_epochs
        
        # ----- 应用降分辨率处理，减少计算量和关注细节 -----
        # 下采样系数随预热进度变化，从较粗糙逐渐变精细
        downscale_factor = max(4, int(8 * (1 - warmup_progress)))
        
        # 下采样输入和标签
        if epoch < warmup_epochs * 0.75:  # 在预热期的75%使用降分辨率
            B, C, H, W = gen_seg.shape
            
            # 下采样
            gen_seg_low = F.avg_pool2d(gen_seg, kernel_size=downscale_factor, stride=downscale_factor)
            real_labels_low = F.avg_pool2d(real_labels, kernel_size=downscale_factor, stride=downscale_factor)
            
            # 上采样回原始尺寸
            gen_seg_up = F.interpolate(gen_seg_low, size=(H, W), mode='bilinear', align_corners=False)
            
            # 对下采样后的低分辨率版本计算损失
            gen_loss_low = F.binary_cross_entropy(gen_seg_low, real_labels_low)
            
            # 同时计算上采样回原分辨率的结果与原分辨率目标的相似度
            gen_loss_high = F.binary_cross_entropy(gen_seg_up, real_labels)
            
            # 混合损失，随预热进度增加原始分辨率的比重
            gen_loss = gen_loss_low * (1.0 - warmup_progress) + gen_loss_high * warmup_progress
        else:
            # 预热后期使用原始分辨率
            gen_loss = F.binary_cross_entropy(gen_seg, real_labels)
        
        # 对抗损失
        adv_loss = F.binary_cross_entropy(
            outputs['fake_prob'], 
            torch.ones_like(outputs['fake_prob'])
        )
        
        # ----- 全局连贯性激励 -----
        # 计算全局特征相关性，鼓励生成长区域TAD
        B, C, H, W = gen_seg.shape
        
        # 计算横向和纵向的连贯性 - 惩罚频繁变化
        horizontal_coherence = torch.abs(gen_seg[:, :, :, 1:] - gen_seg[:, :, :, :-1]).mean()
        vertical_coherence = torch.abs(gen_seg[:, :, 1:, :] - gen_seg[:, :, :-1, :]).mean()
        
        # 对大幅度变化进行惩罚，鼓励平滑变化和大块区域
        coherence_loss = (horizontal_coherence + vertical_coherence) * 2.0
        
        # 块大小奖励因子 - 鼓励大尺度结构
        downscaled = F.avg_pool2d(gen_seg, kernel_size=8, stride=8)  # 大幅下采样
        upscaled = F.interpolate(downscaled, size=(H, W), mode='bilinear', align_corners=False)
        
        # 鼓励下采样和上采样的结果接近，相当于鼓励大块区域
        global_consistency = F.mse_loss(gen_seg, upscaled) * 0.3
        
        # ----- 添加稀疏性惩罚，减少过多边界 -----
        # 计算梯度大小作为边界指示
        edge_h = torch.abs(gen_seg[:, :, :, 1:] - gen_seg[:, :, :, :-1])
        edge_v = torch.abs(gen_seg[:, :, 1:, :] - gen_seg[:, :, :-1, :])
        
        # 平均边界密度
        edge_density = (edge_h.mean() + edge_v.mean()) * 0.5
        
        # 惩罚高边界密度
        sparsity_loss = edge_density * 2.0
        
        # 组合全局特征损失
        global_loss = coherence_loss + global_consistency + sparsity_loss
        
        # 当预热期接近结束时，逐渐减小全局特征损失的权重
        warmup_weight = max(0.0, 1.0 - warmup_progress * 1.2) * 0.8  # 从0.8逐渐降至0
        
        # 总损失 - 预热期更强调生成器通用特征
        total_loss = gen_loss + 0.01 * adv_loss + warmup_weight * global_loss
        
        return {
            'total': total_loss,
            'gen': gen_loss,
            'adv': adv_loss,
            'global': global_loss,
            'warmup_weight': torch.tensor(warmup_weight, device=gen_seg.device),
            'resolution': torch.tensor(downscale_factor, device=gen_seg.device)
        }

    def _visualize_scale_weights(self):
        """可视化尺度权重分布"""
        # 计算平均尺度权重
        avg_weights = torch.cat(self.scale_weight_history[-10:], dim=0)
        avg_weights = avg_weights.mean(dim=0).numpy()
        
        # 创建可视化图
        plt.figure(figsize=(10, 6))
        labels = ['Small', 'Medium', 'Large', 'Extra Large']
        plt.bar(labels, avg_weights.flatten())
        plt.title('TAD Scale Preference')
        plt.ylabel('Average Weight')
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(os.path.join(self.output_root, self.chr_name, "scale_vis"), exist_ok=True)
        plt.savefig(os.path.join(self.output_root, self.chr_name, "scale_vis", 
                               f"scale_weights_{len(self.scale_weight_history)}.png"))
        plt.close()

    def _get_real_labels(self, matrix):
        """生成真实标签（示例实现）"""
        # 这里需要根据实际数据格式实现标签生成逻辑
        return torch.sigmoid(matrix)  # 示例处理

    def _visualize_results(self, outputs, epoch, output_dir):
        """增强的可视化函数"""
        try:
            # 提取需要可视化的数据
            original = outputs['tile'].cpu().numpy()[0, 0]
            segmentation = outputs['gen_seg'].cpu().numpy()[0, 0]
            real_prob = outputs['real_prob'].cpu().numpy()[0, 0]
            fake_prob = outputs['fake_prob'].cpu().numpy()[0, 0]
            
            # 创建当前epoch的可视化文件夹
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 使用matplotlib设置，避免中文问题
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            # 1. 原始矩阵与分割结果对比
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            im0 = axes[0].imshow(original, cmap='viridis')
            axes[0].set_title("Original Hi-C Matrix")
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(segmentation, cmap='plasma')
            axes[1].set_title("Generator Segmentation")
            plt.colorbar(im1, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "matrix_segmentation.png"), dpi=300)
            plt.close()
            
            # 2. 判别器结果
            plt.figure(figsize=(10, 6))
            prob_values = [real_prob.mean(), fake_prob.mean()]
            plt.bar(['Real', 'Fake'], prob_values, color=['blue', 'red'])
            plt.title("Discriminator Evaluation")
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "discriminator_results.png"), dpi=300)
            plt.close()
            
            # 3. 边界概率分布
            edge_h = np.abs(np.diff(segmentation, axis=0))
            edge_v = np.abs(np.diff(segmentation, axis=1))
            edge_map = np.maximum(np.pad(edge_h, ((0,1), (0,0))), np.pad(edge_v, ((0,0), (0,1))))
            
            plt.figure(figsize=(12, 10))
            plt.imshow(edge_map, cmap='hot')
            plt.colorbar()
            plt.title("TAD Boundary Probabilities")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "boundary_map.png"), dpi=300)
            plt.close()
            
            # 4. 分析TAD区域大小分布
            labeled_image = self._analyze_tad_regions(segmentation)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(labeled_image, cmap='nipy_spectral')
            plt.colorbar(label='TAD Region ID')
            plt.title("TAD Regions")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "tad_regions.png"), dpi=300)
            plt.close()
            
            # 保存处理后的numpy数组以供进一步分析
            np.save(os.path.join(epoch_dir, "segmentation.npy"), segmentation)
            np.save(os.path.join(epoch_dir, "boundary_map.npy"), edge_map)
            
            # 添加尺度偏好可视化
            if hasattr(self.adv_net.generator, 'scale_info'):
                scale_info = self.adv_net.generator.scale_info
                # 使用detach()确保不需要梯度计算
                scale_weights = scale_info['weights'].detach().cpu().numpy()[0].flatten()
                
                plt.figure(figsize=(10, 6))
                labels = ['Small', 'Medium', 'Large', 'Extra Large']
                plt.bar(labels, scale_weights)
                plt.title('Current TAD Scale Preference')
                plt.ylabel('Weight')
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "scale_preference.png"), dpi=300)
                plt.close()
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            # 打印更详细的错误栈跟踪，便于调试
            import traceback
            traceback.print_exc()

    def _analyze_tad_regions(self, segmentation, threshold=0.5):
        """分析TAD区域分布，返回标记后的图像"""
        # 二值化
        binary = (segmentation > threshold).astype(np.int32)
        
        # 简化的连通区域标记算法
        def label_regions(binary_img):
            h, w = binary_img.shape
            labeled = np.zeros_like(binary_img)
            current_label = 1
            
            # 对每个像素处理
            for i in range(h):
                for j in range(w):
                    if binary_img[i, j] == 1 and labeled[i, j] == 0:
                        # 使用广度优先搜索标记连通区域
                        queue = [(i, j)]
                        labeled[i, j] = current_label
                        
                        while queue:
                            x, y = queue.pop(0)
                            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < h and 0 <= ny < w and binary_img[nx, ny] == 1 and labeled[nx, ny] == 0:
                                    labeled[nx, ny] = current_label
                                    queue.append((nx, ny))
                        
                        current_label += 1
            
            return labeled
        
        # 标记区域
        labeled_image = label_regions(binary)
        
        return labeled_image

    def run(self):
        """Main process execution: Data loading -> CUR decomposition -> Model training and parameter saving"""
        matrix = self.load_data()  # 原始矩阵 [N,N]
        
        # 数据标准化
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
        
        # 转换为tensor并添加批次维度
        cur_tensor = torch.from_numpy(matrix).float().to(self.device)
        cur_tensor = cur_tensor.unsqueeze(0)  # [1,N,N]
        
        self.train_model(cur_tensor)

    def _split_into_tiles(self, matrix, tile_size=256):
        """安全分块方法，确保输出保持4D"""
        # 输入矩阵格式 [B, C, H, W]
        if matrix.dim() == 3:
            matrix = matrix.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        elif matrix.dim() != 4:
            raise ValueError(f"输入维度必须为3或4，当前维度：{matrix.dim()}")
        
        B, C, H, W = matrix.shape
        # 自动调整tile_size避免产生过多分块
        tile_size = min(tile_size, H, W)
        
        tiles = []
        for h in range(0, H, tile_size):
            for w in range(0, W, tile_size):
                h_end = min(h + tile_size, H)
                w_end = min(w + tile_size, W)
                tile = matrix[:, :, h:h_end, w:w_end]
                tiles.append(tile)
        return tiles

    def _merge_results(self, tiles, original_shape):
        """将处理后的tiles合并回原始形状"""
        B, C, H, W = original_shape
        tile_size = tiles[0].size(-1)
        n_h = H // tile_size
        n_w = W // tile_size
        
        merged = torch.zeros(original_shape, device=tiles[0].device)
        idx = 0
        for h in range(n_h):
            for w in range(n_w):
                h_start = h * tile_size
                h_end = (h+1) * tile_size
                w_start = w * tile_size
                w_end = (w+1) * tile_size
                
                merged[:, :, h_start:h_end, w_start:w_end] = tiles[idx]
                idx += 1
        return merged

def main():
    pipeline = TADPipelineAdversarial(resume_training=True)
    pipeline.run()
    print("Model training and parameter saving process completed")

if __name__ == "__main__":
    main()