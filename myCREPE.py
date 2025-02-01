import torch
import librosa
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from mymodel.model import get_model
from music21 import stream, note, tempo, meter
from scipy.io import wavfile

# 配置 MuseScore
import music21
us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
us['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

def load_model(model_path, model_size='tiny', device='cuda'):
    """加载训练好的模型"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_model(model_size).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def process_audio(audio_path, model, device, step_size=10):
    """处理音频文件并进行音高检测"""
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 居中填充
    audio = np.pad(audio, 512, mode='constant', constant_values=0)
    
    # 分帧
    hop_length = int(16000 * step_size / 1000)  # 步长(毫秒转样本数)
    frames = librosa.util.frame(audio, frame_length=1024, hop_length=hop_length).T
    
    # 归一化
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
    
    # 转换为tensor
    frames = torch.from_numpy(frames).float().to(device)
    
    # 批处理预测
    predictions = []
    confidences = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            output = model(batch)
            
            # 计算频率和置信度
            cents = to_local_average_cents(output.cpu().numpy())
            frequency = 10.0 * (2 ** (cents / 1200))
            confidence = torch.max(output, dim=1)[0].cpu().numpy()
            
            predictions.extend(frequency)
            confidences.extend(confidence)
    
    # 计算时间戳
    times = librosa.frames_to_time(np.arange(len(predictions)), 
                                 sr=16000, hop_length=hop_length)
    
    return times, np.array(predictions), np.array(confidences)

def hz_to_midi_note(hz):
    """将赫兹转换为MIDI音符编号"""
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz/440.0)

def create_score(audio_path, model_path, confidence_threshold=0.5, min_note_length=0.1):
    """使用训练好的模型创建乐谱"""
    print("加载模型...")
    model, device = load_model(model_path, model_size='tiny')  # 确保使用tiny模型
    
    print("加载音频文件...")
    # 进行音高检测
    print("正在进行音高检测...")
    time, frequency, confidence = process_audio(audio_path, model, device)
    
    # 创建音乐流
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=120))
    s.append(meter.TimeSignature('4/4'))
    
    # 处理音高序列
    current_note = None
    current_start = None
    pending_notes = []
    
    print("处理检测到的音高...")
    print(f"检测到的频率范围: {frequency.min():.2f}Hz - {frequency.max():.2f}Hz")
    print(f"置信度范围: {confidence.min():.2f} - {confidence.max():.2f}")
    
    # 应用中值滤波来平滑频率序列
    window_size = 5
    frequency = np.pad(frequency, (window_size//2, window_size//2), mode='edge')
    frequency = np.array([np.median(frequency[i:i+window_size]) 
                         for i in range(len(frequency)-window_size+1)])
    
    for i in range(len(time)):
        if confidence[i] >= confidence_threshold:
            midi_note = hz_to_midi_note(frequency[i])
            if midi_note is not None and 21 <= midi_note <= 108:  # 限制在钢琴音域内
                if current_note is None:
                    # 开始新音符
                    current_note = midi_note
                    current_start = time[i]
                elif abs(current_note - midi_note) >= 0.5:  # 音高变化超过半音
                    # 结束当前音符
                    duration = time[i] - current_start
                    if duration >= min_note_length:
                        pending_notes.append({
                            'pitch': int(round(current_note)),
                            'start_time': current_start,
                            'duration': duration
                        })
                    # 开始新音符
                    current_note = midi_note
                    current_start = time[i]
        elif current_note is not None:
            # 结束当前音符
            duration = time[i] - current_start
            if duration >= min_note_length:
                pending_notes.append({
                    'pitch': int(round(current_note)),
                    'start_time': current_start,
                    'duration': duration
                })
            current_note = None
    
    # 添加音符到音乐流
    print("添加音符到乐谱...")
    for note_info in pending_notes:
        n = note.Note(note_info['pitch'])
        n.duration.quarterLength = note_info['duration'] * 2  # 将秒转换为四分音符长度
        s.insert(note_info['start_time'], n)
        print(f"添加音符: MIDI音高 {note_info['pitch']}, 时长: {note_info['duration']:.2f}秒")
    
    return s

def to_local_average_cents(salience, center=None):
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
        return np.array([to_local_average_cents(salience[i, :]) 
                        for i in range(salience.shape[0])])

if __name__ == "__main__":
    try:
        # 设置模型路径
        model_path = "models/demo_pitch_model_epoch2.pt"
        
        # 创建乐谱
        score = create_score("output.wav", model_path, 
                           confidence_threshold=0.5,  # 降低置信度阈值
                           min_note_length=0.1)      # 设置最小音符长度
        
        # 保存MIDI文件
        print("正在保存MIDI文件...")
        score.write('midi', fp='output_my.mid')
        print("MIDI文件保存成功")
        
        # 检查音符数量
        note_count = len(score.flatten().notes)
        print(f"生成的音符数量: {note_count}")
        
        if note_count == 0:
            print("警告：没有检测到任何音符！")
            exit(1)
        
        # 显示乐谱
        print("正在显示乐谱...")
        score.show('midi')
        print("已打开MIDI文件")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
