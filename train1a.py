import torch
import os
import numpy as np
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

    def _create_model(self, patch_size=None):
        """Create DINO-V2 model based on A-VIT backbone network"""
        model_params = self.get_model_params()
        if patch_size is not None:
            model_params['patch_size'] = patch_size
        model = AVIT_DINO(
            embed_dim=model_params['embed_dim'], 
            patch_size=model_params['patch_size'], 
            num_layers=model_params['num_layers'], 
            num_heads=model_params['num_heads'],
            use_amp=model_params['use_amp'],
            ema_decay=model_params['ema_decay'],
            mask_ratio=model_params['mask_ratio'],
            gamma_base=model_params['gamma_base'],
            epsilon_base=model_params['epsilon_base'],
            use_theory_gamma=model_params['use_theory_gamma'],
            boundary_weight=model_params['boundary_weight']
        ).to(self.device)
        print(f"Creating DINO-V2 model with A-VIT backbone, parameters: {model_params}")
        return model

    def train_model(self, cur_tensor):
        # 初始化模型
        self.adv_net = AdversarialTAD(freeze_ratio=0.75).to(self.device)
        
        # 使用分组优化器，为判别器设置更低的学习率
        optimizer_params = [
            {'params': self.adv_net.generator.parameters(), 'lr': 1e-4},
            {'params': self.adv_net.discriminator.parameters(), 'lr': 2e-5}  # 判别器学习率降低5倍
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params)
        
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
                # 用于跟踪当前epoch的平均损失
                epoch_losses = {'total': 0.0, 'gen': 0.0, 'disc': 0.0, 'adv': 0.0}
                
                # 内部进度条跟踪tile处理
                with tqdm(enumerate(tiles), total=len(tiles), desc=f"Epoch {epoch+1}/{self.num_epochs}") as tile_pbar:
                    for i, tile in tile_pbar:
                        # 确保输入连续
                        tile = tile.contiguous()
                        
                        # 前向传播
                        outputs = self.adv_net(tile)
                        
                        # 创建标签并计算损失
                        real_labels = torch.sigmoid(tile)
                        losses = self._compute_losses(outputs, real_labels)
                        
                        # 反向传播
                        self.optimizer.zero_grad()
                        losses['total'].backward()
                        self.optimizer.step()
                        
                        # 更新内部进度条
                        tile_pbar.set_postfix({
                            'loss': f"{losses['total'].item():.4f}",
                            'gen': f"{losses['gen'].item():.4f}",
                            'disc': f"{losses['disc'].item():.4f}"
                        })
                        
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
                    epoch_losses[k] /= len(tiles)
                
                # 更新外部进度条
                epoch_pbar.set_postfix({
                    'avg_loss': f"{epoch_losses['total']:.4f}",
                    'gen_loss': f"{epoch_losses['gen']:.4f}",
                    'disc_loss': f"{epoch_losses['disc']:.4f}"
                })
                
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
        # 生成器损失
        gen_loss = F.binary_cross_entropy(outputs['gen_seg'], real_labels)
        
        # 对抗损失（生成器欺骗判别器）
        adv_loss = F.binary_cross_entropy(
            outputs['fake_prob'], 
            torch.ones_like(outputs['fake_prob'])
        )
        
        # 判别器损失
        real_loss = F.binary_cross_entropy(
            outputs['real_prob'],
            torch.ones_like(outputs['real_prob'])
        )
        fake_loss = F.binary_cross_entropy(
            outputs['fake_prob'],
            torch.zeros_like(outputs['fake_prob'])
        )
        disc_loss = (real_loss + fake_loss) / 2
        
        return {
            'total': gen_loss + self.adv_net.adv_weight * adv_loss + disc_loss,
            'gen': gen_loss,
            'adv': adv_loss,
            'disc': disc_loss
        }

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
            plt.plot([real_prob, fake_prob], ['b', 'r'], label=['Real Probability', 'Fake Probability'])
            plt.title("Discriminator Evaluation")
            plt.xlabel("Probability")
            plt.ylabel("Type")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "discriminator_results.png"), dpi=300)
            plt.close()
            
            # 3. 边界概率分布（从分割结果中提取边界）
            # 计算分割结果的梯度作为边界
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
            
            # 保存处理后的numpy数组以供进一步分析
            np.save(os.path.join(epoch_dir, "segmentation.npy"), segmentation)
            np.save(os.path.join(epoch_dir, "boundary_map.npy"), edge_map)
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")

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