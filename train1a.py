import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from net import (
    TADBaseConfig,
    AVIT_DINO,
    LowRank,
    find_chromosome_files,
    fill_hic
)

class TADPipelineDINO(TADBaseConfig):
    """TAD Detection Training Pipeline (migrated from original detect_tad.py)"""
    
    def __init__(self, resume_training=False, **kwargs):
        super().__init__(**kwargs)
        self.resume_training = resume_training
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize components and checkpoint configurations"""
        self.adj_graph = None
        self.normalized_matrix = None
        self.avit_dino = None
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
        """Train model using DINO-V2 and save best model parameters"""
        print("=== Starting DINO-V2 Self-supervised Training ===")
        
        chr_output_dir = os.path.join(self.output_root, self.chr_name)
        os.makedirs(chr_output_dir, exist_ok=True)
        best_model_path = os.path.join(chr_output_dir, "best_model.pth")
        
        best_loss = float('inf')
        
        if self.resume_training and os.path.exists(best_model_path):
            print(f"找到现有模型文件，从断点继续训练: {best_model_path}")
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                
                if not self.avit_dino:
                    self.avit_dino = self._create_model()
                    self.avit_dino.init_optimizer()
                
                self.avit_dino.student.load_state_dict(checkpoint['student'])
                self.avit_dino.teacher.load_state_dict(checkpoint['teacher'])
                
                if 'student_unet' in checkpoint:
                    self.avit_dino.student_unet.load_state_dict(checkpoint['student_unet'])
                if 'teacher_bilstm' in checkpoint:
                    self.avit_dino.teacher_bilstm.load_state_dict(checkpoint['teacher_bilstm'])
                if 'mask_token' in checkpoint:
                    self.avit_dino.mask_token.data.copy_(checkpoint['mask_token'])
                
                if 'optimizer' in checkpoint:
                    self.avit_dino.optimizer.load_state_dict(checkpoint['optimizer'])
                
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                
                print(f"成功加载模型状态，继续训练，当前最佳损失: {best_loss:.4f}")
                
            except Exception as e:
                print(f"加载模型发生错误: {str(e)}，将从头开始训练")
                best_loss = float('inf')
        
        if not self.avit_dino:
            self.avit_dino = self._create_model()
            self.avit_dino.init_optimizer()
        
        networks = self.avit_dino.get_network_for_training()
        
        with tqdm(range(self.num_epochs), desc="Training DINO-V2 Model", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}') as pbar:
            for epoch in pbar:
                losses = self.avit_dino.train_epoch(cur_tensor)
                pbar.set_postfix({
                    'total_loss': f"{losses['total']:.4f}",
                    'cls_loss': f"{losses['cls']:.4f}",
                    'patch_loss': f"{losses['patch']:.4f}",
                    'seg_loss': f"{losses.get('segmentation', 0):.4f}"
                })
                
                if losses['total'] < best_loss:
                    best_loss = losses['total']
                    save_dict = {
                        'student': self.avit_dino.student.state_dict(),
                        'teacher': self.avit_dino.teacher.state_dict(),
                        'student_unet': self.avit_dino.student_unet.state_dict(),
                        'teacher_bilstm': self.avit_dino.teacher_bilstm.state_dict(),
                        'mask_token': self.avit_dino.mask_token,
                        'model_params': self.get_model_params(),
                        'optimizer': self.avit_dino.optimizer.state_dict(),
                        'best_loss': best_loss
                    }
                    torch.save(save_dict, best_model_path)
                
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
                    self._visualize_boundary_predictions(cur_tensor, epoch, chr_output_dir)
                
        print(f"DINO-V2 training completed, best loss: {best_loss:.4f}")
        del self.avit_dino.optimizer
        if hasattr(self.avit_dino, 'scaler'):
            del self.avit_dino.scaler
        torch.cuda.empty_cache()
        print("Training resources released")

    def _visualize_boundary_predictions(self, matrix, epoch, output_dir):
        """可视化边界预测结果"""
        try:
            # 设置全局字体支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            boundary_preds = self.avit_dino.get_boundary_predictions(matrix)
            segmentation = boundary_preds.get('segmentation')
            boundary_probs = boundary_preds.get('boundary_probs')
            
            if segmentation is not None:
                seg_np = segmentation.cpu().numpy()[0, 0]
                
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(seg_np, cmap='viridis')
                plt.colorbar(im)
                # 使用英文替代中文避免字体问题
                plt.title(f"Epoch {epoch}: Segmentation Prediction")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"seg_pred_epoch{epoch}.png"))
                plt.close()
                
            if boundary_probs is not None:
                boundary_np = boundary_probs.cpu().numpy()[0]
                
                plt.figure(figsize=(12, 5))
                plt.plot(boundary_np)
                # 使用英文替代中文避免字体问题
                plt.title(f"Epoch {epoch}: Boundary Probabilities")
                plt.ylabel("Probability")
                plt.xlabel("Position")
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"boundary_probs_epoch{epoch}.png"))
                plt.close()
        except Exception as e:
            print(f"Error visualizing boundary predictions: {str(e)}")

    def run(self):
        """Main process execution: Data loading -> CUR decomposition -> Model training and parameter saving"""
        matrix = self.load_data()
        projector = LowRank(p=0.7, alpha=0.7)
        cur_matrix = projector.fit_transform(matrix)
        print(f"[DEBUG] CUR matrix shape: {cur_matrix.shape}, sparsity: {np.mean(cur_matrix < 1e-6):.2%}")
        del matrix
        
        print(f"[DEBUG] CUR matrix shape: {cur_matrix.shape}, sparsity: {np.mean(cur_matrix < 1e-6):.2%}")
        
        cur_tensor = torch.from_numpy(cur_matrix).float().to(self.device)
        del cur_matrix
        
        self.train_model(cur_tensor)

def main():
    pipeline = TADPipelineDINO(resume_training=True)
    pipeline.run()
    print("Model training and parameter saving process completed")

if __name__ == "__main__":
    main()