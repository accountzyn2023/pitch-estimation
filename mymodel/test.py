import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from mymodel.model import get_model
from mymodel.data_load import get_dataloader, hz_to_cent
from mymodel.train import create_gaussian_target
from tqdm import tqdm

class PitchTester:
    """音高检测测试器"""
    
    def __init__(self, model_path, device='cuda'):
        """
        初始化测试器
        
        参数:
            model_path: 模型权重文件路径
            device: 测试设备
        """
        self.device = device
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model_size = checkpoint.get('model_size', 'medium')  # 从checkpoint中获取模型大小，默认为medium
        
        # 初始化模型
        self.model = get_model(model_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 设置为评估模式
        
        print(f"已加载{model_size}型号模型")
    
    def predict_pitch(self, waveform):
        """预测单个音频片段的音高"""
        with torch.no_grad():
            # 确保输入形状正确并归一化
            # 输入可能是 [batch_size, 1, 1024] 或其他形状
            # 我们需要将其转换为 [batch_size, 1, 1024, 1]
            
            # 首先将其转换为3D张量 [batch_size, 1, 1024]
            if waveform.dim() == 5:  # [batch_size, 1, 1, 1024, 1]
                waveform = waveform.squeeze(2).squeeze(-1)  # [batch_size, 1, 1024]
            elif waveform.dim() == 4:  # [batch_size, 1, 1024, 1]
                waveform = waveform.squeeze(-1)  # [batch_size, 1, 1024]
            elif waveform.dim() == 2:  # [batch_size, 1024]
                waveform = waveform.unsqueeze(1)  # [batch_size, 1, 1024]
            
            # 归一化 - 使用更稳定的方法
            waveform = waveform - waveform.mean(dim=2, keepdim=True)  # 在时间维度上计算均值
            std = torch.std(waveform, dim=2, keepdim=True)  # 在时间维度上计算标准差
            std_mean = std.mean()  # 计算所有通道的平均标准差
            
            if std_mean > 0:  # 只在标准差大于0时进行归一化
                waveform = waveform / (std + 1e-8)
            
            # 添加最后一个维度，得到 [batch_size, 1, 1024, 1]
            waveform = waveform.unsqueeze(-1)
            
            # 确保设备正确
            waveform = waveform.to(self.device)
            
            # 前向传播
            output = self.model(waveform)  # [batch_size, 360]
            
            # 计算置信度
            confidence = torch.max(output, dim=1)[0]
            
            # 使用加权平均方法获取更精确的频率预测
            cents = self.to_local_average_cents(output.cpu().numpy())
            frequency = 10.0 * (2 ** (cents / 1200))
            
            return frequency.item(), confidence.item()
    
    def to_local_average_cents(self, salience, center=None):
        """计算局部加权平均音分值"""
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            
            # 音分映射
            cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
            product_sum = np.sum(salience * cents_mapping[start:end])
            weight_sum = np.sum(salience)
            
            return product_sum / weight_sum if weight_sum > 0 else 0
        
        if salience.ndim == 2:
            return np.array([self.to_local_average_cents(salience[i, :]) 
                            for i in range(salience.shape[0])])
    
    def test_model(self, test_loader):
        """
        在测试集上评估模型
        
        参数:
            test_loader: 测试数据加载器
        """
        total_loss = 0
        predictions = []
        ground_truths = []
        
        print("\n开始测试...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # 获取数据
                waveforms = batch['waveform'].to(self.device)
                pitches = batch['pitch'].to(self.device)
                
                # 创建目标向量
                targets = create_gaussian_target(pitches)
                
                # 前向传播
                outputs = self.model(waveforms)
                
                # 计算损失
                loss = torch.nn.functional.binary_cross_entropy(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测结果
                for i in range(len(waveforms)):
                    pred_freq, pred_confidence = self.predict_pitch(waveforms[i:i+1])
                    true_freq = pitches[i].mean().item()  # 使用片段的平均音高作为真实值
                    
                    if true_freq > 0:  # 只收集有效的音高值
                        predictions.append(pred_freq)
                        ground_truths.append(true_freq)
        
        # 计算平均损失
        avg_loss = total_loss / len(test_loader)
        print(f"测试损失: {avg_loss:.4f}")
        
        # 计算评估指标
        self.calculate_metrics(predictions, ground_truths)
        
        # 绘制散点图
        self.plot_results(predictions, ground_truths)
    
    def calculate_metrics(self, predictions, ground_truths):
        """计算评估指标"""
        if not predictions or not ground_truths:
            print("警告：没有有效的预测结果，无法计算评估指标")
            return
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        # 计算音分误差
        cents_error = 1200 * np.abs(np.log2(predictions / ground_truths))
        
        # 计算各种指标
        mean_error = np.mean(cents_error)
        median_error = np.median(cents_error)
        std_error = np.std(cents_error)
        
        # 计算准确率
        accuracy_25 = np.mean(cents_error < 25)  # 音分误差小于25的比例
        accuracy_50 = np.mean(cents_error < 50)  # 音分误差小于50的比例
        accuracy_100 = np.mean(cents_error < 100)  # 音分误差小于100的比例
        
        print("\n评估指标:")
        print(f"平均音分误差: {mean_error:.2f} cents")
        print(f"中位数音分误差: {median_error:.2f} cents")
        print(f"标准差: {std_error:.2f} cents")
        print(f"准确率 (<25 cents): {accuracy_25*100:.2f}%")
        print(f"准确率 (<50 cents): {accuracy_50*100:.2f}%")
        print(f"准确率 (<100 cents): {accuracy_100*100:.2f}%")
    
    def plot_results(self, predictions, ground_truths):
        """绘制预测结果的散点图"""
        if not predictions or not ground_truths:
            print("警告：没有有效的预测结果，无法绘制散点图")
            return
        
        # 计算频率范围
        min_val = min(min(predictions), min(ground_truths))
        max_val = max(max(predictions), max(ground_truths))
        
        # 创建图表
        plt.figure(figsize=(10, 10))
        plt.scatter(ground_truths, predictions, alpha=0.5)
        
        # 添加对角线
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 设置坐标轴
        plt.xlabel('真实频率 (Hz)')
        plt.ylabel('预测频率 (Hz)')
        plt.title('音高预测结果')
        
        # 使用对数刻度
        plt.xscale('log')
        plt.yscale('log')
        
        # 保存图表
        save_path = Path("plots/test_results.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"结果图表已保存到: {save_path}")

def test_model(data_dir, model_path, device='cuda'):
    """测试模型"""
    try:
        # 初始化测试器
        tester = PitchTester(model_path, device)
        print(f"模型加载自: {model_path}")
        
        # 加载测试数据
        print("\n加载测试数据...")
        test_loader = get_dataloader(
            data_dir, 
            batch_size=1,  # 测试时使用batch_size=1
            hop_length=10,  # 使用与训练相同的hop_length（10毫秒）
            segment_length=1024,  # 添加段长度参数
            sample_rate=16000,  # 添加采样率参数
            fold=0, 
            mode='test'
        )
        
        if len(test_loader) == 0:
            raise ValueError("测试数据集为空")
            
        print(f"测试集总大小: {len(test_loader)} 批次")
        print(f"将测试 {min(len(test_loader), 1000)} 个批次\n")  # 限制测试批次数
        
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
                output = tester.model(waveform)  # [1, num_bins]
                pred_pitch = tester.predict_pitch(waveform)[0]  # [1]
                
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
            plot_results(predictions, targets, filenames)
        else:
            print("警告：没有有效的预测结果，无法计算评估指标")
            print("\n绘制结果散点图...")
            print("警告：没有有效的预测结果，无法绘制散点图")
            
    except Exception as e:
        print("\n测试过程发生错误:")
        print(f"错误信息: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def plot_results(predictions, targets, filenames):
    """绘制预测结果的散点图"""
    if len(predictions) == 0 or len(targets) == 0:  # 修改检查逻辑
        print("警告：没有有效的预测结果，无法绘制散点图")
        return
    
    # 计算频率范围
    min_val = min(min(predictions), min(targets))
    max_val = max(max(predictions), max(targets))
    
    # 创建图表
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # 添加对角线
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 设置坐标轴
    plt.xlabel('真实频率 (Hz)')
    plt.ylabel('预测频率 (Hz)')
    plt.title('音高预测结果')
    
    # 使用对数刻度
    plt.xscale('log')
    plt.yscale('log')
    
    # 保存图表
    save_path = Path("plots/test_results.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"结果图表已保存到: {save_path}")

if __name__ == "__main__":
    # 测试模型
    test_model(
        data_dir="./MIR-1K",
        model_path="models/demo_pitch_model_epoch50.pt",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
