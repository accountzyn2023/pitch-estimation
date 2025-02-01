import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
from mymodel.model import get_model
from mymodel.data_load import get_dataloader

def create_gaussian_target(pitch, num_bins=360, sigma=25):
    """
    创建高斯模糊处理的目标向量
    
    参数:
        pitch: 真实基频值 (batch_size, sequence_length)
        num_bins: 输出向量维度
        sigma: 高斯分布的标准差（音分）
    
    返回:
        torch.Tensor: 高斯模糊后的目标向量 (batch_size, num_bins)
    """
    device = pitch.device
    batch_size, seq_length = pitch.shape
    target = torch.zeros((batch_size, num_bins), device=device)
    
    # 将频率值转换为音分值
    pitch_mean = pitch.mean(dim=1)  # (batch_size,)
    cents = 1200 * torch.log2(pitch_mean / 10.0)  # 使用10Hz作为参考频率
    
    # 将音分值映射到bin索引
    bins = (cents - 1997.3794084376191) * (360 / 7180)
    bins = torch.clamp(bins, 0, num_bins-1).long()
    
    # 为每个样本创建高斯分布
    x = torch.arange(num_bins, device=device).float()
    for i in range(batch_size):
        if pitch_mean[i] > 0:  # 只处理有效的音高值
            center = bins[i].item()
            gaussian = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
            gaussian = gaussian / gaussian.max()
            target[i] = gaussian
    
    return target

class PitchTrainer:
    def __init__(self, model_size='tiny', device='cuda'):
        """
        初始化训练器
        
        参数:
            model_size: 模型大小 ('tiny', 'small', 'medium', 'large', 'full')
            device: 训练设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                # 获取数据
                waveforms = batch['waveform'].to(self.device)  # [B, 1, 1024, 1]
                pitches = batch['pitch'].to(self.device)      # [B, T]
                
                # 创建高斯模糊目标
                targets = create_gaussian_target(pitches)     # [B, 360]
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(waveforms)              # [B, 360]
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新统计
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch['waveform'].to(self.device)
                pitches = batch['pitch'].to(self.device)
                
                targets = create_gaussian_target(pitches)
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, data_dir, num_epochs=50, batch_size=1, patience=32):
        """训练模型"""
        try:
            # 记录训练历史
            history = {'train_loss': [], 'val_loss': []}
            
            # 进行5折交叉验证
            for fold in range(5):
                try:
                    print(f"\n开始第 {fold + 1} 折训练")
                    
                    # 获取数据加载器
                    try:
                        train_loader = get_dataloader(data_dir, batch_size=batch_size, 
                                                    fold=fold, mode='train')
                        val_loader = get_dataloader(data_dir, batch_size=batch_size, 
                                                  fold=fold, mode='val')
                    except Exception as e:
                        print("数据加载器创建失败:")
                        print(f"Error: {str(e)}")
                        raise
                    
                    # 重置模型和优化器
                    try:
                        self.model = get_model('medium').to(self.device)
                        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)
                        self.best_val_loss = float('inf')
                        self.patience_counter = 0
                    except Exception as e:
                        print("模型初始化失败:")
                        print(f"Error: {str(e)}")
                        raise
                    
                    for epoch in range(num_epochs):
                        try:
                            print(f"\nEpoch {epoch + 1}/{num_epochs}")
                            
                            # 训练和验证
                            try:
                                train_loss = self.train_epoch(train_loader)
                                print(f"训练阶段完成，损失: {train_loss:.4f}")
                            except Exception as e:
                                print(f"训练阶段失败:")
                                print(f"Error: {str(e)}")
                                raise
                            
                            try:
                                val_loss = self.validate(val_loader)
                                print(f"验证阶段完成，损失: {val_loss:.4f}")
                            except Exception as e:
                                print(f"验证阶段失败:")
                                print(f"Error: {str(e)}")
                                raise
                            
                            # 记录历史
                            history['train_loss'].append(train_loss)
                            history['val_loss'].append(val_loss)
                            
                            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                            
                            # 检查是否需要保存模型
                            if val_loss < self.best_val_loss:
                                try:
                                    self.best_val_loss = val_loss
                                    self.patience_counter = 0
                                    # 保存模型
                                    save_path = Path(f"models/pitch_model_fold{fold}.pt")
                                    save_path.parent.mkdir(parents=True, exist_ok=True)
                                    torch.save({
                                        'epoch': epoch,
                                        'model_state_dict': self.model.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict(),
                                        'loss': val_loss,
                                    }, save_path)
                                    print(f"模型已保存到 {save_path}")
                                except Exception as e:
                                    print(f"模型保存失败:")
                                    print(f"Error: {str(e)}")
                                    raise
                            else:
                                self.patience_counter += 1
                            
                            # 早停检查
                            if self.patience_counter >= patience:
                                print(f"验证损失在 {patience} 个epoch内未改善，停止训练")
                                break
                        
                        except Exception as e:
                            print(f"Epoch {epoch + 1} 训练失败:")
                            print(f"Error: {str(e)}")
                            print("尝试继续下一个epoch...")
                            continue
                    
                    # 绘制损失曲线
                    try:
                        self.plot_history(history, fold)
                    except Exception as e:
                        print("绘制历史记录失败:")
                        print(f"Error: {str(e)}")
                
                except Exception as e:
                    print(f"第 {fold + 1} 折训练失败:")
                    print(f"Error: {str(e)}")
                    print("尝试继续下一折...")
                    continue
            
        except Exception as e:
            print("\n训练过程发生致命错误:")
            print(f"Error: {str(e)}")
            import traceback
            print("\n完整的错误追踪:")
            print(traceback.format_exc())
            raise
    
    def plot_history(self, history, fold):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title(f'第 {fold + 1} 折训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 保存图表
        save_path = Path(f"plots/training_history_fold{fold}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def train_model(data_dir, model_size='tiny', device='cuda'):
    """
    训练模型的主函数
    
    参数:
        data_dir: 数据集目录
        model_size: 模型大小
        device: 训练设备
    """
    trainer = PitchTrainer(model_size, device)
    trainer.train(data_dir)

if __name__ == "__main__":
    # 训练模型
    train_model("./MIR-1K")
