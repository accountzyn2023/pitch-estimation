import torch
import torchaudio
import random
import numpy as np

class AudioAugmentor:
    """音频数据增强器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def add_noise(self, waveform, noise_level=0.005):
        """添加高斯噪声
        
        Args:
            waveform (Tensor): 输入波形 [channels, samples]
            noise_level (float): 噪声强度
        """
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def time_mask(self, waveform, num_masks=1, mask_len=1000):
        """时间掩码
        
        Args:
            waveform (Tensor): 输入波形
            num_masks (int): 掩码数量
            mask_len (int): 最大掩码长度（样本数）
        """
        aug_waveform = waveform.clone()
        for _ in range(num_masks):
            mask_start = random.randint(0, waveform.shape[1] - mask_len)
            aug_waveform[:, mask_start:mask_start + mask_len] = 0
        return aug_waveform
    
    def random_gain(self, waveform, min_gain=-6, max_gain=6):
        """随机增益
        
        Args:
            waveform (Tensor): 输入波形
            min_gain (float): 最小增益(dB)
            max_gain (float): 最大增益(dB)
        """
        gain = random.uniform(min_gain, max_gain)
        return waveform * (10 ** (gain / 20))
    
    def freq_mask(self, spec, F=30, num_masks=1):
        """频率掩码（用于频谱图）
        
        Args:
            spec (Tensor): 频谱图 [channels, freq, time]
            F (int): 最大频率掩码宽度
            num_masks (int): 掩码数量
        """
        aug_spec = spec.clone()
        num_freq = spec.size(1)
        
        for _ in range(num_masks):
            f = random.randint(0, F)
            f0 = random.randint(0, num_freq - f)
            aug_spec[:, f0:f0 + f, :] = 0
        
        return aug_spec

def augment_pitch_labels(pitch_labels, n_steps=0):
    """调整音高标签
    
    Args:
        pitch_labels (Tensor): 音高标签
        n_steps (float): 偏移半音数
    """
    # 只对非零（有效）音高值进行调整
    mask = pitch_labels > 0
    pitch_labels[mask] = pitch_labels[mask] * (2 ** (n_steps / 12))
    return pitch_labels 