import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import random
from mymodel.train import PitchTrainer, create_gaussian_target
from mymodel.data_load import get_dataloader

def demo_train(data_dir, model_size='tiny', device='cuda'):
    """
    演示训练脚本，使用较小的数据集和简化的训练参数
    
    参数:
        data_dir: 数据集目录
        model_size: 模型大小，默认为'tiny'
        device: 训练设备，默认为'cuda'
    """
    # 训练参数
    batch_size = 1  # 增大批次大小
    num_epochs = 50
    sample_ratio = 1  # 增大采样比例
    step_size = 10  # 毫秒
    
    try:
        # 检查数据目录是否存在
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        print(f"使用数据目录: {data_dir}")
        trainer = PitchTrainer(model_size, device)
        
        # 获取数据加载器并打印信息
        print("\n加载数据...")
        train_loader = get_dataloader(data_dir, batch_size=batch_size, 
                                    hop_length=step_size, fold=0, mode='train')
        val_loader = get_dataloader(data_dir, batch_size=batch_size, 
                                  hop_length=step_size, fold=0, mode='val')
        
        print(f"训练集大小: {len(train_loader)} 批次")
        print(f"验证集大小: {len(val_loader)} 批次")
        
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("数据集为空，请检查数据目录和数据加载器")
            
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练阶段
            trainer.model.train()  # 设置为训练模式
            train_loss = 0
            valid_batches = 0
            max_train_batches = int(len(train_loader) * sample_ratio)
            
            for i, batch in enumerate(train_loader):
                if i >= max_train_batches:
                    print(f"训练批次 {i} 达到采样比例 ({sample_ratio*100:.1f}%), 停止训练")
                    break
                
                if batch is None:
                    continue
                    
                # 获取数据
                waveforms = batch['waveform'].to(trainer.device)  # [batch_size, 1, 1024]
                pitches = batch['pitch'].to(trainer.device)
                
                # 添加最后一个维度 [batch_size, 1, 1024] -> [batch_size, 1, 1024, 1]
                waveforms = waveforms.unsqueeze(-1)
                
                # 创建目标向量
                targets = create_gaussian_target(pitches)
                
                # 前向传播和反向传播
                trainer.optimizer.zero_grad()
                outputs = trainer.model(waveforms)
                loss = trainer.criterion(outputs, targets)
                loss.backward()
                trainer.optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
                
                if (i + 1) % 10 == 0:  # 每10个批次打印一次
                    print(f"训练批次 {i+1}/{max_train_batches}, "
                          f"损失: {loss.item():.4f}, "
                          f"平均损失: {train_loss/valid_batches:.4f}")
            
            train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
            
            # 验证阶段
            trainer.model.eval()  # 设置为评估模式
            val_loss = 0
            valid_batches = 0
            max_val_batches = int(len(val_loader) * sample_ratio)
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= max_val_batches:
                        break
                        
                    if batch is None:
                        continue
                        
                    waveforms = batch['waveform'].to(trainer.device)
                    pitches = batch['pitch'].to(trainer.device)
                    
                    # 添加最后一个维度
                    waveforms = waveforms.unsqueeze(-1)
                    
                    targets = create_gaussian_target(pitches)
                    outputs = trainer.model(waveforms)
                    loss = trainer.criterion(outputs, targets)
                    val_loss += loss.item()
                    valid_batches += 1
                    
                val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
            
            print(f"Epoch {epoch + 1} - "
                  f"训练损失: {train_loss:.4f}, "
                  f"验证损失: {val_loss:.4f}")
            
            # 保存模型
            save_path = Path(f"models/demo_pitch_model_epoch{epoch+1}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_size': model_size,
            }, save_path)
            print(f"模型已保存到 {save_path}")
            
    except Exception as e:
        print("\n演示训练过程发生错误:")
        print(f"错误信息: {str(e)}")
        raise

if __name__ == "__main__":
    # 演示训练
    demo_train("./MIR-1K")
