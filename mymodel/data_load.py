import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import random
from mymodel.augment import AudioAugmentor, augment_pitch_labels
import librosa

def hz_to_cent(freq, ref=10.0):
    """将Hz转换为音分"""
    return 1200 * np.log2(freq / ref) if freq > 0 else 0

def gaussian_blur_cent(target_cent, num_bins=360, std=25):
    """生成高斯模糊目标向量"""
    centers = np.linspace(0, 7180, num_bins)  # C1(32.70Hz)到B7(1975.5Hz)对应0~7180音分
    target_vector = np.exp(-(centers - target_cent)**2 / (2 * std**2))
    return target_vector / target_vector.sum()  # 归一化

def frame_audio(audio, frame_length=1024, hop_length=None):
    """将音频分帧"""
    if hop_length is None:
        hop_length = int(frame_length / 4)
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    
    # 对每一帧进行归一化
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
    
    return frames

def midi_to_hz(midi_note):
    """将MIDI音符号转换为频率(Hz)"""
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

class MIR1KDataset(Dataset):
    """MIR-1K数据集加载器"""
    
    def __init__(self, root_dir, segment_length=1024, hop_length=512, sample_rate=16000, 
                 augment=False, augment_prob=0.5, fold=0, mode='train', num_folds=5):
        """
        参数:
            root_dir (str): MIR-1K数据集的根目录
            segment_length (int): 音频片段长度（样本数）
            hop_length (int): 帧移（样本数），增大以减少重叠
            sample_rate (int): 目标采样率
            augment (bool): 是否使用数据增强
            augment_prob (float): 数据增强的概率
            fold (int): 当前使用的折数 (0-4)
            mode (str): 'train', 'val', 或 'test'
            num_folds (int): 交叉验证的折数
        """
        self.root_dir = Path(root_dir)
        self.wav_dir = self.root_dir / "Wavfile"
        self.pitch_dir = self.root_dir / "PitchLabel"
        self.segment_length = segment_length
        self.hop_length = hop_length  # 直接使用样本数
        self.sample_rate = sample_rate
        self.augment = augment
        self.augment_prob = augment_prob
        self.fold = fold
        self.mode = mode
        self.num_folds = num_folds
        
        # 获取所有音频文件
        self.wav_files = sorted([f for f in self.wav_dir.glob("*.wav")])
        print(f"找到 {len(self.wav_files)} 个音频文件")
        
        # 预加载所有音高标签
        self.pitch_cache = {}
        for pitch_file in self.pitch_dir.glob("*.pv"):
            self.pitch_cache[pitch_file.stem] = np.loadtxt(pitch_file)
        
        # 划分数据集
        self.split_dataset()
        
        # 预处理参数
        self.hop_samples = int(hop_length * sample_rate / 1000)  # 将毫秒转换为样本数
        
        # 构建数据索引
        self.build_index()
        
        if augment:
            self.augmentor = AudioAugmentor(sample_rate=sample_rate)
    
    def split_dataset(self):
        """按歌曲ID分组划分数据集"""
        # 按歌曲ID分组
        song_groups = {}
        for wav_file in self.wav_files:
            song_id = wav_file.stem.split('_')[0]
            if song_id not in song_groups:
                song_groups[song_id] = []
            song_groups[song_id].append(wav_file)
        
        # 打乱歌曲ID
        song_ids = list(song_groups.keys())
        random.seed(42)
        random.shuffle(song_ids)
        
        # 划分歌曲ID
        fold_size = len(song_ids) // self.num_folds
        start = self.fold * fold_size
        end = start + fold_size
        
        if self.mode == 'train':
            train_ids = song_ids[:start] + song_ids[end:]
            self.wav_files = [f for sid in train_ids for f in song_groups[sid]]
        elif self.mode == 'val':
            val_ids = song_ids[start:end]
            self.wav_files = [f for sid in val_ids for f in song_groups[sid]]
        else:  # test
            # 修改测试集的划分方式
            test_ids = song_ids[end:min(end + fold_size, len(song_ids))]  # 确保不越界
            self.wav_files = [f for sid in test_ids for f in song_groups[sid]]
        
        print(f"{self.mode} 集大小: {len(self.wav_files)} 个文件")
    
    def build_index(self):
        """构建数据索引，将音频分割成片段"""
        self.segments = []
        valid_segments = 0
        total_segments = 0
        pitch_stats = {
            'total_frames': 0,
            'valid_frames': 0,
            'max_pitch': float('-inf'),
            'min_pitch': float('inf'),
            'pitch_histogram': [],
            'filtered_by_ratio': 0,
            'filtered_by_range': 0,
            'debug_info': {
                'total_pitch_segments': 0,
                'valid_pitch_segments': 0,
                'ratio_check_passed': 0,
                'range_check_passed': 0
            }
        }
        
        for wav_file in self.wav_files:
            try:
                # 获取对应的音高标签文件
                pitch_file = self.pitch_dir / (wav_file.stem + ".pv")
                if not pitch_file.exists():
                    print(f"找不到音高文件: {pitch_file}")
                    continue
                    
                # 加载音高标签并转换为Hz
                midi_labels = self.pitch_cache[wav_file.stem]
                pitch_labels = np.array([midi_to_hz(note) if note > 0 else 0 for note in midi_labels])
                
                # 加载音频
                waveform, sr = torchaudio.load(wav_file)
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                # 计算可用的片段数量
                num_segments = (waveform.shape[1] - self.segment_length) // self.hop_length + 1
                
                # 限制每个文件的最大片段数
                max_segments_per_file = 100
                if num_segments > max_segments_per_file:
                    step = num_segments // max_segments_per_file
                    segment_indices = range(0, num_segments, step)
                else:
                    segment_indices = range(num_segments)
                
                # 添加每个片段的索引信息
                for i in segment_indices:
                    total_segments += 1
                    start_sample = i * self.hop_length
                    start_frame = start_sample // self.hop_length
                    num_frames = self.segment_length // self.hop_length
                    
                    # 检查这个片段是否有有效的音高标签
                    if start_frame + num_frames <= len(pitch_labels):
                        pitch_segment = pitch_labels[start_frame:start_frame + num_frames]
                        pitch_stats['debug_info']['total_pitch_segments'] += 1
                        
                        # 确保片段长度足够长（至少32帧）
                        if len(pitch_segment) >= 32:  # 降低最小帧数要求
                            valid_pitches = pitch_segment[pitch_segment > 0]
                            valid_ratio = len(valid_pitches) / len(pitch_segment)
                            
                            # 更新有效片段计数
                            pitch_stats['debug_info']['valid_pitch_segments'] += 1
                            
                            # 放宽筛选条件
                            if valid_ratio >= 0.1:  # 降低到10%
                                pitch_stats['debug_info']['ratio_check_passed'] += 1
                                if len(valid_pitches) > 0:
                                    pitch_mean = np.mean(valid_pitches)
                                    if 40 <= pitch_mean <= 1000:
                                        pitch_stats['debug_info']['range_check_passed'] += 1
                                        self.segments.append({
                                            'wav_file': wav_file,
                                            'pitch_file': pitch_file,
                                            'start_sample': start_sample,
                                            'waveform_length': waveform.shape[1],
                                            'valid_ratio': valid_ratio,
                                            'pitch_mean': pitch_mean
                                        })
                                        valid_segments += 1
                                    else:
                                        pitch_stats['filtered_by_range'] += 1
                                else:
                                    pitch_stats['filtered_by_ratio'] += 1
                            
                            # 更新音高统计信息
                            if len(valid_pitches) > 0:
                                pitch_stats['max_pitch'] = max(pitch_stats['max_pitch'], valid_pitches.max())
                                pitch_stats['min_pitch'] = min(pitch_stats['min_pitch'], valid_pitches.min())
                                pitch_stats['pitch_histogram'].extend(valid_pitches.tolist())
                            pitch_stats['total_frames'] += len(pitch_segment)
                            pitch_stats['valid_frames'] += len(valid_pitches)
                            
            except Exception as e:
                print(f"处理文件 {wav_file} 时出错: {str(e)}")
                print(f"错误详情: ", e.__class__.__name__)
                import traceback
                print(traceback.format_exc())
                continue
        
        # 打印详细的统计信息
        print(f"\n数据集统计信息:")
        print(f"总片段数: {total_segments}")
        print(f"有效片段数: {valid_segments} ({valid_segments/max(1,total_segments)*100:.1f}%)")
        print(f"被有效帧比例过滤掉的片段数: {pitch_stats['filtered_by_ratio']}")
        print(f"被音高范围过滤掉的片段数: {pitch_stats['filtered_by_range']}")
        print(f"总帧数: {pitch_stats['total_frames']}")
        print(f"有效帧数: {pitch_stats['valid_frames']} ({pitch_stats['valid_frames']/max(1,pitch_stats['total_frames'])*100:.1f}%)")
        
        print("\n调试信息:")
        print(f"总音高片段数: {pitch_stats['debug_info']['total_pitch_segments']}")
        print(f"有效音高片段数: {pitch_stats['debug_info']['valid_pitch_segments']}")
        print(f"通过比例检查的片段数: {pitch_stats['debug_info']['ratio_check_passed']}")
        print(f"通过范围检查的片段数: {pitch_stats['debug_info']['range_check_passed']}")
        
        if pitch_stats['valid_frames'] > 0:
            print(f"音高范围: {pitch_stats['min_pitch']:.1f}Hz - {pitch_stats['max_pitch']:.1f}Hz")
            
            # 打印音高分布统计
            hist = np.histogram(pitch_stats['pitch_histogram'], 
                              bins=np.linspace(0, 1000, 21))  # 0-1000Hz分20个区间
            print("\n音高分布:")
            for i in range(len(hist[0])):
                if hist[0][i] > 0:
                    print(f"{hist[1][i]:.0f}-{hist[1][i+1]:.0f}Hz: {hist[0][i]} 帧")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        try:
            segment_info = self.segments[idx]
            
            # 首先检查音高标签是否有效
            midi_labels = self.pitch_cache[segment_info['wav_file'].stem]
            start_frame = segment_info['start_sample'] // self.hop_samples
            num_frames = self.segment_length // self.hop_samples
            
            # 确保音高标签长度足够
            if start_frame + num_frames > len(midi_labels):
                return None
            
            # 获取MIDI音高片段并转换为Hz
            midi_segment = midi_labels[start_frame:start_frame + num_frames]
            pitch_segment = np.array([midi_to_hz(note) if note > 0 else 0 for note in midi_segment])
            
            # 检查音高段是否有效（至少有一个有效的音高值）
            valid_pitches = pitch_segment[pitch_segment > 0]
            if len(valid_pitches) == 0:
                return None
            
            # 检查音高是否在合理范围内
            pitch_mean = np.mean(valid_pitches)
            if not (50 <= pitch_mean <= 1000):
                return None
            
            # 加载音频
            waveform, sr = torchaudio.load(segment_info['wav_file'])
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # 提取歌声信号
            if waveform.shape[0] > 1:
                vocal = waveform[0] - waveform[1]  # 人声分离
            else:
                vocal = waveform[0]
            
            # 提取片段并归一化
            start_sample = segment_info['start_sample']
            segment = vocal[start_sample:start_sample + self.segment_length]
            
            # 如果片段长度不足，直接返回None
            if segment.shape[0] < self.segment_length:
                return None
            
            # 归一化
            segment = segment - segment.mean()
            segment = segment / (segment.std() + 1e-8)
            
            # 重塑为模型需要的输入形状 [1, 1024]
            segment = segment.reshape(1, -1)
            
            return {
                'waveform': segment,  # [1, 1024]
                'pitch': torch.from_numpy(pitch_segment).float(),
                'filename': segment_info['wav_file'].name
            }
            
        except Exception as e:
            print(f"\n加载样本 {idx} 时出错:")
            print(f"文件: {segment_info['wav_file']}")
            print(f"错误信息: {str(e)}")
            print(f"错误详情: ", e.__class__.__name__)
            import traceback
            print(traceback.format_exc())
            return None

def collate_fn(batch):
    """自定义的数据批次整理函数"""
    # 过滤掉无效的样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # 获取每个样本的数据
    waveforms = [item['waveform'] for item in batch]  # 每个元素形状为 [1, 1024]
    pitches = [item['pitch'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    # 堆叠所有样本
    waveforms = torch.stack(waveforms)  # [batch_size, 1, 1024]
    pitches = torch.stack([p if len(p.shape) > 0 else p.unsqueeze(0) for p in pitches])
    
    return {
        'waveform': waveforms,  # [batch_size, 1, 1024]
        'pitch': pitches,
        'filename': filenames
    }

def get_dataloader(root_dir, batch_size=32, num_workers=0, augment=False, 
                  fold=0, mode='train', hop_length=512, **kwargs):
    """创建数据加载器"""
    dataset = MIR1KDataset(
        root_dir, 
        augment=augment, 
        fold=fold,
        mode=mode,
        hop_length=hop_length,  # 确保使用相同的hop_length
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False  # 测试时不丢弃最后一个批次
    )

# 测试代码
if __name__ == "__main__":
    # 测试5折交叉验证的数据加载
    for fold in range(5):
        print(f"\n测试第 {fold + 1} 折:")
        # 加载训练集
        train_loader = get_dataloader("./MIR-1K", batch_size=32, fold=fold, mode='train')
        # 加载验证集
        val_loader = get_dataloader("./MIR-1K", batch_size=32, fold=fold, mode='val')
        # 加载测试集
        test_loader = get_dataloader("./MIR-1K", batch_size=32, fold=fold, mode='test')
        
        print(f"训练集大小: {len(train_loader.dataset)} 个样本")
        print(f"验证集大小: {len(val_loader.dataset)} 个样本")
        print(f"测试集大小: {len(test_loader.dataset)} 个样本")