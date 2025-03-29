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
import math

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
        torch.backends.cuda.checkpoint_layers_n = 50

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
        
        # 初始学习率
        initial_lr_gen = 1e-7
        initial_lr_disc = 2e-10
        
        # 目标学习率
        target_lr_gen = 1e-5
        target_lr_disc = 2e-7
        
        # 使用分组优化器，为判别器设置更低的学习率
        optimizer_params = [
            {'params': self.adv_net.generator.parameters(), 'lr': initial_lr_gen},
            {'params': self.adv_net.discriminator.parameters(), 'lr': initial_lr_disc}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params)
        
        # 创建可视化输出目录
        vis_dir = os.path.join(self.output_root, self.chr_name, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 使用更小的块大小减少内存占用
        tile_size = 128
        tiles = self._split_into_tiles(cur_tensor, tile_size)
        
        # 损失记录字典，用于tqdm显示
        loss_meters = {
            'gen': AverageMeter('生成器损失'),
            'adv': AverageMeter('对抗损失'),
            'disc': AverageMeter('判别器损失'),
            'total': AverageMeter('总损失')
        }
        
        # 计算总步数
        total_steps = self.num_epochs * len(tiles)
        warmup_steps = int(0.1 * total_steps)  # 前10%的步骤保持初始学习率
        print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
        # 计算余弦退火截止步数 (总进度的60%)
        cosine_end_step = int(0.6 * total_steps)
        
        # 当前步数计数器
        global_step = 0
        
        # 开始训练循环
        for epoch in range(self.num_epochs):
            # 重置损失记录器
            for meter in loss_meters.values():
                meter.reset()
            
            # 创建带有进度条的tqdm实例
            pbar = tqdm(enumerate(tiles), total=len(tiles), desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 为每个图块执行一次前向和后向传播
            for batch_idx, tile in pbar:
                # 学习率调度策略
                # 1. 前10步保持初始学习率
                if global_step < warmup_steps:
                    current_lr_gen = initial_lr_gen
                    current_lr_disc = initial_lr_disc
                    lr_phase = "预热"
                
                # 2. 从第11步到总进度的60%使用余弦递增(而非退火)
                elif global_step < cosine_end_step:
                    # 计算余弦进度 [0,1]
                    progress = (global_step - warmup_steps) / (cosine_end_step - warmup_steps)
                    
                    # 余弦递增公式 - 从预热期结束值平滑增加到目标值
                    # 注意：修改公式以适应从小到大的变化，使用(1-cos)替代(1+cos)
                    cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                    
                    # 计算当前学习率 - 注意参数顺序，确保是从小到大
                    current_lr_gen = initial_lr_gen + cosine_factor * (target_lr_gen - initial_lr_gen)
                    current_lr_disc = initial_lr_disc + cosine_factor * (target_lr_disc - initial_lr_disc)
                    lr_phase = "余弦增长"
                
                # 3. 从总进度的60%到结束保持目标学习率
                else:
                    current_lr_gen = target_lr_gen
                    current_lr_disc = target_lr_disc
                    lr_phase = "稳定"
                
                # 更新优化器中的学习率
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if i == 0:  # 生成器参数组
                        param_group['lr'] = current_lr_gen
                    else:  # 判别器参数组
                        param_group['lr'] = current_lr_disc
                
                # 显示当前学习率信息
                lr_info = f"{lr_phase}[G={current_lr_gen:.1e},D={current_lr_disc:.1e}]"
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.adv_net(tile)
                
                # 计算损失
                losses = self._compute_losses(outputs, tile)
                
                # 反向传播
                losses['total'].backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), 1.0)
                
                # 参数更新
                self.optimizer.step()
                
                # 更新损失记录
                for k, v in losses.items():
                    loss_meters[k].update(v.item())
                
                # 更新tqdm进度条显示当前损失和学习率
                pbar.set_postfix({
                    'Gen': f'{loss_meters["gen"].avg:.4f}',
                    'Adv': f'{loss_meters["adv"].avg:.4f}',
                    'Disc': f'{loss_meters["disc"].avg:.4f}',
                    'Total': f'{loss_meters["total"].avg:.4f}',
                    'LR': lr_info
                })
                
                # 更新全局步数
                global_step += 1
            
            # 第一个epoch完成后立即进行可视化
            if epoch == 0:
                self._visualize_results(epoch, tiles[0], vis_dir)
            
            # 每5个epoch执行一次可视化
            elif (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self._visualize_results(epoch, tiles[0], vis_dir)
            
            # 每个epoch结束后保存模型
            model_path = os.path.join(self.output_root, self.chr_name, f"model_epoch_{epoch+1}.pth")
            torch.save(self.adv_net.state_dict(), model_path)
            
            # 保存最新的模型作为最佳模型
            best_model_path = os.path.join(self.output_root, self.chr_name, "best_model.pth")
            torch.save(self.adv_net.state_dict(), best_model_path)
        
        print("训练完成!")
    
    def _compute_losses(self, outputs, real_labels):
        """支持贝叶斯不确定性的损失计算，增强数值稳定性"""
        # 生成器重建损失
        gen_loss = F.binary_cross_entropy(
            outputs['gen_seg'], 
            real_labels,
            reduction='none'  # 不立即求平均，以便进行异常值处理
        )
        
        # 检测并修正异常值
        if torch.isnan(gen_loss).any() or torch.isinf(gen_loss).any():
            print("警告: 检测到NaN或Inf在生成器损失中")
            gen_loss = torch.where(torch.isnan(gen_loss) | torch.isinf(gen_loss), 
                                   torch.tensor(0.1, device=gen_loss.device), 
                                   gen_loss)
        
        # 首先平均每个样本的损失
        gen_loss = gen_loss.mean(dim=[1, 2, 3])
        
        # 应用梯度裁剪（值裁剪）防止极端大值
        gen_loss = torch.clamp(gen_loss, 0, 100.0)
        
        # 最终平均
        gen_loss = gen_loss.mean()
        
        # 如果使用了内部优化，我们可以添加内部激励历史记录到输出中
        if 'reward_history' in outputs and outputs['reward_history'] is not None:
            last_reward = outputs['reward_history'][-1]
            # 可以将激励历史记录到日志或TensorBoard
            if self.global_step % 10 == 0:  # 每10步记录一次
                print(f"内部优化指标: IC={last_reward['internal_consistency']:.4f}, "
                      f"ES={last_reward['edge_significance']:.4f}, "
                      f"CP={last_reward['change_point']:.4f}")
        
        # 不确定性引导损失（如果可用）
        if 'uncertainty' in outputs:
            # 检查不确定性值是否正常
            uncertainty = outputs['uncertainty']
            if torch.isnan(uncertainty).any() or torch.isinf(uncertainty).any():
                print("警告: 检测到NaN或Inf在不确定性中")
                uncertainty = torch.zeros_like(uncertainty)
            
            # 如果是贝叶斯模型，加入KL散度正则化
            kl_div = 0.0
            if hasattr(self.adv_net.generator, 'get_kl_divergence'):
                kl_div = self.adv_net.generator.get_kl_divergence()
                # 限制KL散度的大小，防止它主导总损失
                kl_weight = min(1e-3, 1.0 / real_labels.shape[0])  # 降低KL权重
                kl_div = torch.clamp(kl_div, 0, 1000.0)  # 限制最大值
                gen_loss = gen_loss + kl_weight * kl_div
                
                # # 打印KL散度值以进行调试
                # if self.adv_net.generator.training:
                #     print(f"KL散度: {kl_div.item():.4f}, 权重: {kl_weight:.6f}")
        
        # 对抗损失（生成器欺骗判别器）
        fake_prob = outputs['fake_prob']
        if torch.isnan(fake_prob).any() or torch.isinf(fake_prob).any():
            print("警告: 检测到NaN或Inf在fake_prob中")
            fake_prob = torch.ones_like(fake_prob) * 0.5
        
        adv_loss = F.binary_cross_entropy(
            fake_prob, 
            torch.ones_like(fake_prob)
        )
        
        # 判别器损失
        real_prob = outputs['real_prob']
        if torch.isnan(real_prob).any() or torch.isinf(real_prob).any():
            print("警告: 检测到NaN或Inf在real_prob中")
            real_prob = torch.ones_like(real_prob) * 0.5
        
        real_loss = F.binary_cross_entropy(
            real_prob,
            torch.ones_like(real_prob)
        )
        fake_loss = F.binary_cross_entropy(
            fake_prob,
            torch.zeros_like(fake_prob)
        )
        disc_loss = (real_loss + fake_loss) / 2
        
        # 动态调整对抗权重，开始时降低对抗权重
        adv_weight = min(self.adv_net.adv_weight, 0.01)  # 初始阶段减小对抗权重
        
        # 确保总损失合理
        total_loss = gen_loss + adv_weight * adv_loss + disc_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告: 总损失是NaN或Inf，使用只有生成器损失的备份")
            total_loss = gen_loss  # 只用生成器损失作为备份
        
        return {
            'total': total_loss,
            'gen': gen_loss,
            'adv': adv_loss,
            'disc': disc_loss
        }

    def _visualize_results(self, epoch, sample, output_dir):
        """生成训练结果的详细可视化"""
        # 创建当前epoch的可视化目录
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 设置Matplotlib字体和样式以避免中文显示问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 将模型设置为评估模式
        self.adv_net.eval()
        
        with torch.no_grad():  # 确保不计算梯度
            # 获取模型预测
            outputs = self.adv_net(sample)
            
            # 提取原始样本和预测结果
            original = sample.cpu().numpy()[0, 0]
            segmentation = outputs['gen_seg'].cpu().numpy()[0, 0]
            
            # 1. 原始Hi-C矩阵与分割结果对比可视化
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # 原始矩阵
            im0 = axes[0].imshow(original, cmap='Blues')
            axes[0].set_title("Original Hi-C Matrix", fontsize=14)
            axes[0].set_xlabel("Genomic Position", fontsize=12)
            axes[0].set_ylabel("Genomic Position", fontsize=12)
            plt.colorbar(im0, ax=axes[0], label='Contact Frequency')
            
            # 分割结果
            im1 = axes[1].imshow(segmentation, cmap='viridis')
            axes[1].set_title("TAD Segmentation Result", fontsize=14)
            axes[1].set_xlabel("Genomic Position", fontsize=12)
            axes[1].set_ylabel("Genomic Position", fontsize=12)
            plt.colorbar(im1, ax=axes[1], label='TAD Probability')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "matrix_segmentation.png"), dpi=300)
            plt.close()
            
            # 2. 判别器结果可视化
            plt.figure(figsize=(8, 6))
            probabilities = [
                outputs['real_prob'].mean().item(),
                outputs['fake_prob'].mean().item()
            ]
            labels = ['Real', 'Generated']
            colors = ['#3498db', '#e74c3c']
            
            plt.bar(labels, probabilities, color=colors)
            plt.title("Discriminator Evaluation", fontsize=14)
            plt.ylabel("Probability", fontsize=12)
            plt.ylim(0, 1)
            for i, prob in enumerate(probabilities):
                plt.text(i, prob + 0.02, f"{prob:.3f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "discriminator_results.png"), dpi=300)
            plt.close()
            
            # 3. 不确定性可视化（贝叶斯模型特有）
            if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                uncertainty = outputs['uncertainty'].cpu().numpy()[0, 0]
                
                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                
                # 不确定性热图
                im0 = axes[0].imshow(uncertainty, cmap='hot')
                axes[0].set_title("Prediction Uncertainty", fontsize=14)
                axes[0].set_xlabel("Genomic Position", fontsize=12)
                axes[0].set_ylabel("Genomic Position", fontsize=12)
                plt.colorbar(im0, ax=axes[0], label='Uncertainty Level')
                
                # 不确定性与分割结果叠加
                axes[1].imshow(segmentation, cmap='Blues', alpha=0.7)
                im1 = axes[1].imshow(uncertainty, cmap='hot', alpha=0.5)
                axes[1].set_title("Segmentation with Uncertainty", fontsize=14)
                axes[1].set_xlabel("Genomic Position", fontsize=12)
                axes[1].set_ylabel("Genomic Position", fontsize=12)
                plt.colorbar(im1, ax=axes[1], label='Uncertainty Level')
                
                # 高不确定性区域掩码
                threshold = np.percentile(uncertainty, 75)  # 75%分位数作为阈值
                high_uncertainty = (uncertainty > threshold).astype(float)
                im2 = axes[2].imshow(high_uncertainty, cmap='Reds')
                axes[2].set_title(f"High Uncertainty Regions (>{threshold:.3f})", fontsize=14)
                axes[2].set_xlabel("Genomic Position", fontsize=12)
                axes[2].set_ylabel("Genomic Position", fontsize=12)
                plt.colorbar(im2, ax=axes[2], label='High Uncertainty Mask')
                
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "uncertainty_analysis.png"), dpi=300)
                plt.close()
            
            # 4. TAD边界分析
            # 计算分割结果的梯度作为边界
            edge_h = np.abs(np.diff(segmentation, axis=0))
            edge_v = np.abs(np.diff(segmentation, axis=1))
            edge_map = np.maximum(np.pad(edge_h, ((0,1), (0,0))), np.pad(edge_v, ((0,0), (0,1))))
            
            # 创建三图布局：边界、原始矩阵与边界叠加、TAD区域
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # 边界热图
            im0 = axes[0].imshow(edge_map, cmap='hot')
            axes[0].set_title("TAD Boundary Probabilities", fontsize=14)
            axes[0].set_xlabel("Genomic Position", fontsize=12)
            axes[0].set_ylabel("Genomic Position", fontsize=12)
            plt.colorbar(im0, ax=axes[0], label='Boundary Strength')
            
            # 边界与原始矩阵叠加
            axes[1].imshow(original, cmap='Blues', alpha=0.7)
            im1 = axes[1].imshow(edge_map, cmap='hot', alpha=0.5)
            axes[1].set_title("Hi-C Matrix with TAD Boundaries", fontsize=14)
            axes[1].set_xlabel("Genomic Position", fontsize=12)
            axes[1].set_ylabel("Genomic Position", fontsize=12)
            plt.colorbar(im1, ax=axes[1], label='Boundary Strength')
            
            # TAD区域标识
            # 尝试使用watershed或备用方法
            from scipy import ndimage
            
            # 阈值处理以识别TAD（非边界区域）
            boundary_threshold = np.percentile(edge_map, 90)
            binary_tad_regions = edge_map <= boundary_threshold
            
            # 使用连通组件分析标记TAD
            tad_regions, num_features = ndimage.label(binary_tad_regions)
            
            # 显示TAD区域
            im2 = axes[2].imshow(tad_regions, cmap='tab20b', alpha=0.8)
            axes[2].set_title(f"Detected TAD Regions (n={np.max(tad_regions)})", fontsize=14)
            axes[2].set_xlabel("Genomic Position", fontsize=12)
            axes[2].set_ylabel("Genomic Position", fontsize=12)
            plt.colorbar(im2, ax=axes[2], label='TAD ID')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "tad_boundary_analysis.png"), dpi=300)
            plt.close()
            
            # 5. 训练进度摘要图
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.8, f"Epoch: {epoch+1}/{self.num_epochs}", 
                     ha='center', fontsize=16, weight='bold')
            plt.text(0.5, 0.65, f"Number of TADs detected: {np.max(tad_regions)}", 
                     ha='center', fontsize=14)
            plt.text(0.5, 0.55, f"Avg. TAD size: {original.shape[0]/max(1, np.max(tad_regions)):.1f} bins", 
                     ha='center', fontsize=14)
            plt.text(0.5, 0.45, f"Mean boundary strength: {np.mean(edge_map):.3f}", 
                     ha='center', fontsize=14)
            
            if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
                plt.text(0.5, 0.35, f"Mean uncertainty: {np.mean(uncertainty):.3f}", 
                         ha='center', fontsize=14)
                plt.text(0.5, 0.25, f"Uncertainty/boundary correlation: {np.corrcoef(uncertainty.flatten(), edge_map.flatten())[0,1]:.3f}", 
                         ha='center', fontsize=14)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "training_summary.png"), dpi=300)
            plt.close()
            
            # 6. 贝叶斯权重分布可视化
            if hasattr(self.adv_net.generator, 'enc1') and hasattr(self.adv_net.generator.enc1[0], 'conv'):
                # 提取第一层贝叶斯卷积层的权重分布
                bayes_layer = self.adv_net.generator.enc1[0].conv
                mu = bayes_layer.weight.mu.detach().cpu().numpy()
                sigma = torch.log1p(torch.exp(bayes_layer.weight.rho)).detach().cpu().numpy()
                
                plt.figure(figsize=(14, 6))
                
                # 添加数据范围检查，动态调整bin数量
                # μ分布
                plt.subplot(1, 2, 1)
                mu_range = np.max(mu) - np.min(mu)
                mu_bins = min(50, max(5, int(mu_range * 100)))  # 动态设置bin数量
                try:
                    plt.hist(mu.flatten(), bins=mu_bins, alpha=0.7, color='blue')
                    plt.title(f"Weight Mean (μ) Distribution", fontsize=14)
                    plt.xlabel("Value", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                except ValueError as e:
                    # 如果出错，用更简单的方式显示
                    plt.text(0.5, 0.5, f"μ值范围太小: {mu_range:.6f}\n平均值: {np.mean(mu):.6f}", 
                            ha='center', fontsize=12)
                    plt.axis('on')
                
                # σ分布
                plt.subplot(1, 2, 2)
                sigma_range = np.max(sigma) - np.min(sigma)
                sigma_bins = min(50, max(5, int(sigma_range * 100)))  # 动态设置bin数量
                try:
                    plt.hist(sigma.flatten(), bins=sigma_bins, alpha=0.7, color='red')
                    plt.title(f"Weight Std Dev (σ) Distribution", fontsize=14)
                    plt.xlabel("Value", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                except ValueError as e:
                    # 如果出错，用更简单的方式显示
                    plt.text(0.5, 0.5, f"σ值范围太小: {sigma_range:.6f}\n平均值: {np.mean(sigma):.6f}", 
                            ha='center', fontsize=12)
                    plt.axis('on')
                
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "bayesian_weight_distribution.png"), dpi=300)
                plt.close()
        
        # 将模型恢复为训练模式
        self.adv_net.train()

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

# 添加用于跟踪平均值的工具类
class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

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