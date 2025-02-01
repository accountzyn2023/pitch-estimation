import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from mymodel.test import PitchTester
from mymodel.data_load import get_dataloader

def demo_test(data_dir="./MIR-1K", model_path="models/demo_pitch_model_epoch2.pt", device='cuda'):
    """
    演示测试脚本
    
    参数:
        data_dir: 数据目录
        model_path: 模型文件路径
        device: 测试设备
    """
    try:
        # 检查数据目录是否存在
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        print(f"使用数据目录: {data_dir}")
        print(f"使用模型文件: {model_path}")
        
        # 初始化测试器
        tester = PitchTester(model_path, device)
        
        # 加载测试数据
        print("\n加载测试数据...")
        test_loader = get_dataloader(
            data_dir, 
            batch_size=1,  # 测试时使用batch_size=1
            hop_length=512,  # 使用与训练相同的hop_length
            fold=0, 
            mode='test'
        )
        
        if len(test_loader) == 0:
            raise ValueError("测试数据集为空")
            
        print(f"测试集总大小: {len(test_loader)} 批次")
        print(f"将测试 {min(len(test_loader), 100)} 个批次\n")  # 限制测试批次数
        
        # 开始测试
        print("采样测试数据...")
        all_predictions = []
        all_targets = []
        filenames = []
        
        print("\n开始测试...")
        tester.model.eval()  # 设置为评估模式
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 100:  # 限制测试100个批次
                    break
                    
                if batch is None:
                    continue
                    
                # 获取数据
                waveform = batch['waveform'].to(tester.device)  # [1, 1, 1024]
                pitch = batch['pitch'].to(tester.device)  # [1, num_frames]
                filename = batch['filename'][0]  # 只取第一个文件名
                
                # 添加最后一个维度 [1, 1, 1024] -> [1, 1, 1024, 1]
                waveform = waveform.unsqueeze(-1)
                
                # 预测
                pred_pitch, confidence = tester.predict_pitch(waveform)
                
                # 获取目标音高（取有效音高的平均值）
                target_pitch = pitch[pitch > 0].mean().item() if torch.any(pitch > 0) else 0
                
                if target_pitch > 0:  # 只保存有效的预测结果
                    all_predictions.append(pred_pitch)
                    all_targets.append(target_pitch)
                    filenames.append(filename)
                    
                if (i + 1) % 10 == 0:  # 每10个批次打印一次进度
                    print(f"已处理 {i+1} 个批次")
        
        # 计算评估指标
        print("\n计算整体评估指标...")
        if len(all_predictions) > 0:
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            # 计算各种指标
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            
            print(f"平均绝对误差 (MAE): {mae:.2f} Hz")
            print(f"均方根误差 (RMSE): {rmse:.2f} Hz")
            
            # 绘制散点图
            print("\n绘制结果散点图...")
            tester.plot_results(predictions, targets)
        else:
            print("警告：没有有效的预测结果，无法计算评估指标")
            print("\n绘制结果散点图...")
            print("警告：没有有效的预测结果，无法绘制散点图")
            
    except Exception as e:
        print("\n演示测试过程发生错误:")
        print(f"错误信息: {str(e)}")
        print("\n完整的错误追踪:")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    # 演示测试
    demo_test(
        data_dir="./MIR-1K",
        model_path="models/demo_pitch_model_epoch2.pt",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
