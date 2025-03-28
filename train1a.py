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
    """TAD检测训练流程(支持贝叶斯U-Net与自我反思机制)"""
    
    def __init__(self, resume_training=False, use_bayesian=True, mc_samples=5, **kwargs):
        super().__init__(**kwargs)
        self.resume_training = resume_training
        self.use_bayesian = use_bayesian  # 新增：是否使用贝叶斯U-Net
        self.mc_samples = mc_samples      # 新增：蒙特卡洛采样数量
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """初始化组件和检查点配置"""
        self.adj_graph = None
        self.normalized_matrix = None
        torch.backends.cuda.checkpoint_activations = False
        torch.backends.cuda.checkpoint_layers_n = 10

    def load_data(self):
        """数据加载过程：加载Hi-C数据并返回接触矩阵"""
        hic_paths = find_chromosome_files(self.output_root)
        if not hic_paths:
            raise FileNotFoundError("Hi-C数据文件未找到")
        hic_path = hic_paths[0]
        chr_dir = os.path.dirname(hic_path)
        self.chr_name = os.path.basename(chr_dir)
        print(f"[调试] 当前染色体: {self.chr_name}")
        matrix = fill_hic(hic_path, self.resolution)
        print(f"[调试] 矩阵和 = {np.sum(matrix)}")
        return matrix

    def train_model(self, cur_tensor):
        # 初始化模型 - 支持贝叶斯选项
        self.adv_net = AdversarialTAD(freeze_ratio=0.75, use_bayesian=self.use_bayesian).to(self.device)
        
        # 使用分组优化器，为判别器设置更低的学习率
        optimizer_params = [
            {'params': self.adv_net.generator.parameters(), 'lr': 5e-3},
            {'params': self.adv_net.discriminator.parameters(), 'lr': 2e-6}  # 判别器学习率降低5倍
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params)
        
        # 创建可视化输出目录
        vis_dir = os.path.join(self.output_root, self.chr_name, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 使用更小的块大小减少内存占用
        tile_size = 128
        tiles = self._split_into_tiles(cur_tensor, tile_size)
        
        # 开始训练循环
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 为每个图块执行一次前向和后向传播
            total_loss = 0.0
            gen_loss = 0.0
            adv_loss = 0.0
            disc_loss = 0.0
            
            # 随机打乱图块顺序以减少相关性
            tile_indices = np.random.permutation(len(tiles))
            
            for i, idx in enumerate(tile_indices):
                tile = tiles[idx].to(self.device)
                
                # 清除梯度
                self.optimizer.zero_grad()
                
                # 前向传播 - 支持贝叶斯输出
                outputs = self.adv_net(tile)
                
                # 生成真实标签 - 使用图像本身作为目标
                real_labels = tile
                
                # 计算损失 - 使用支持不确定性的损失计算
                losses = self._compute_losses(outputs, real_labels)
                
                # 计算梯度和更新参数
                losses['total'].backward()
                self.optimizer.step()
                
                # 累积损失统计
                total_loss += losses['total'].item()
                gen_loss += losses['gen'].item()
                adv_loss += losses['adv'].item()
                disc_loss += losses['disc'].item()
                
                # 定期打印训练进度
                if (i + 1) % 10 == 0:
                    print(f"Tile {i+1}/{len(tile_indices)} - Loss: {losses['total'].item():.4f}")
                    
                # 每50个tile可视化一次结果
                if (i + 1) % 50 == 0:
                    self._visualize_results({
                        'tile': tile,
                        'gen_seg': outputs['gen_seg'],
                        'real_prob': outputs['real_prob'],
                        'fake_prob': outputs['fake_prob'],
                        'uncertainty': outputs.get('uncertainty', None)  # 支持不确定性可视化
                    }, epoch, vis_dir)
            
            # 每个epoch结束后保存模型
            avg_loss = total_loss / len(tile_indices)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Gen: {gen_loss/len(tile_indices):.4f}, "
                  f"Adv: {adv_loss/len(tile_indices):.4f}, Disc: {disc_loss/len(tile_indices):.4f}")
            
            model_path = os.path.join(self.output_root, self.chr_name, f"model_epoch_{epoch+1}.pth")
            torch.save(self.adv_net.state_dict(), model_path)
            
            # 保存最新的模型作为最佳模型
            best_model_path = os.path.join(self.output_root, self.chr_name, "best_model.pth")
            torch.save(self.adv_net.state_dict(), best_model_path)
            
        print("训练完成!")
    
    def _compute_losses(self, outputs, real_labels):
        """支持贝叶斯不确定性的损失计算"""
        # 生成器损失
        gen_loss = F.binary_cross_entropy(outputs['gen_seg'], real_labels)
        
        # 不确定性引导损失（如果可用）
        if 'uncertainty' in outputs:
            # 不确定性引导：高不确定性区域增加权重
            uncertainty_weight = 1.0 + torch.sigmoid(outputs['uncertainty'])
            weighted_loss = gen_loss * uncertainty_weight
            gen_loss = weighted_loss.mean()
            
            # 如果是贝叶斯模型，加入KL散度正则化
            if hasattr(self.adv_net.generator, 'get_kl_divergence'):
                kl_div = self.adv_net.generator.get_kl_divergence()
                kl_weight = 1.0 / real_labels.shape[0]  # 按批次大小缩放
                gen_loss = gen_loss + kl_weight * kl_div
        
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

    def _visualize_results(self, outputs, epoch, output_dir):
        """增强的可视化函数，专为对抗性TAD检测设计"""
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
            axes[0].set_title("原始 Hi-C 矩阵")
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(segmentation, cmap='plasma')
            axes[1].set_title("生成器分割结果")
            plt.colorbar(im1, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "矩阵_分割.png"), dpi=300)
            plt.close()
            
            # 2. 判别器结果 - 修正可视化方法
            plt.figure(figsize=(10, 6))
            probabilities = [real_prob.mean(), fake_prob.mean()]
            labels = ['真实', '伪造']
            colors = ['blue', 'red']
            
            plt.bar(labels, probabilities, color=colors)
            plt.title("判别器评估")
            plt.ylabel("概率")
            plt.ylim(0, 1)
            for i, prob in enumerate(probabilities):
                plt.text(i, prob + 0.02, f"{prob:.3f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "判别器结果.png"), dpi=300)
            plt.close()
            
            # 3. 如果有不确定性图，则可视化
            if outputs['uncertainty'] is not None:
                uncertainty = outputs['uncertainty'].cpu().numpy()[0, 0]
                
                plt.figure(figsize=(10, 8))
                im = plt.imshow(uncertainty, cmap='hot')
                plt.title("预测不确定性")
                plt.colorbar(im)
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "不确定性.png"), dpi=300)
                plt.close()
                
                # 4. 将不确定性与分割结果叠加
                plt.figure(figsize=(10, 8))
                plt.imshow(segmentation, cmap='Blues', alpha=0.7)
                plt.imshow(uncertainty, cmap='hot', alpha=0.5)
                plt.title("分割结果与不确定性叠加")
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "分割_不确定性叠加.png"), dpi=300)
                plt.close()
            
            # 5. 边界概率分布（从分割结果中提取边界）
            # 计算分割结果的梯度作为边界
            edge_h = np.abs(np.diff(segmentation, axis=0))
            edge_v = np.abs(np.diff(segmentation, axis=1))
            edge_map = np.maximum(np.pad(edge_h, ((0,1), (0,0))), np.pad(edge_v, ((0,0), (0,1))))
            
            # 边界热图
            plt.figure(figsize=(10, 8))
            im = plt.imshow(edge_map, cmap='hot')
            plt.title("TAD边界概率")
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "边界概率.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"可视化过程中出错: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """主流程执行：数据加载 -> 模型训练和参数保存"""
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
    # 支持贝叶斯选项
    pipeline = TADPipelineAdversarial(
        resume_training=True,
        use_bayesian=True,  # 启用贝叶斯U-Net
        mc_samples=5        # 蒙特卡洛采样数量
    )
    pipeline.run()
    print("模型训练与参数保存流程完成")

if __name__ == "__main__":
    main()