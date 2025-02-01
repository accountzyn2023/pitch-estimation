# 音频音高检测与乐谱生成系统

这是一个基于深度学习的音频音高检测系统，可以将音频文件转换为乐谱。该项目复现了CREPE音高检测模型，并集成了音乐理论分析功能。

## 功能特点

- 音频音高实时检测
- MIDI文件生成
- 乐谱可视化
- 支持多种音频格式
- 集成MuseScore显示功能

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- music21
- librosa
- numpy
- scipy
- MuseScore 4 (用于乐谱显示)

## 安装说明

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/pitch-detection.git
cd pitch-detection
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 安装MuseScore 4 并配置环境变量

## 使用方法

1. 基础音高检测:
```python
from mymodel import create_score

# 创建乐谱
score = create_score("input.wav", "model.pt")

# 保存MIDI文件
score.write('midi', fp='output.mid')
```

2. 使用演示脚本:
```python
python demo.py --input audio.wav --output score.mid
```

## 项目结构

```
pitch-detection/
├── mymodel/
│   ├── __init__.py
│   ├── data_load.py      # 数据加载模块
│   └── test_demo.py      # 测试演示模块
├── CREPE.py             # CREPE模型实现
├── demo.py              # 演示脚本
└── README.md
```

## 核心代码示例

音高检测核心功能:

```77:136:myCREPE.py
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
```


数据处理部分:

```166:252:mymodel/data_load.py
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
```


## 开发指南

请参考以下编码规范:

```206:216:ffmpeg-master-latest-win64-gpl-shared/doc/developer.html
<p>There are the following guidelines regarding the code style in files:
</p>
<ul class="itemize mark-bullet">
<li>Indent size is 4.

</li><li>The TAB character is forbidden outside of Makefiles as is any
form of trailing whitespace. Commits containing either will be
rejected by the git repository.

</li><li>You should try to limit your code lines to 80 characters; however, do so if
and only if this improves readability.
```


## 许可证

本项目采用 GNU General Public License v3.0 许可证。详情请参见 LICENSE 文件。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至: your.email@example.com

## 致谢

- CREPE项目团队
- music21库开发者
- FFmpeg社区
