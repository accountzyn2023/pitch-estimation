import music21
import librosa
import numpy as np
from music21 import stream, note, tempo, converter, meter

# 在文件开头添加这些配置
import music21
us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
us['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

# 1. 加载音频文件
audio_path = "output.wav"
y, sr = librosa.load(audio_path, sr=44100)

# 2. 提取音高和音符起始点
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# 3. 获取音符的名称
notes = librosa.hz_to_note(pitches[np.where(pitches > 0)])
notes = [n.replace('♯', '#').replace('♭', 'b') for n in notes]

# 打印调试信息
print(f"onset_times 长度: {len(onset_times)}")
print(f"notes 长度: {len(notes)}")

# 打印 onset_times 和 notes
for i in range(len(onset_times)):
    print(f"onset_time[{i}]: {onset_times[i]}, note[{i}]: {notes[i] if i < len(notes) else 'N/A'}")

# 4. 创建音乐流并添加音符
s = stream.Stream()
s.insert(0, tempo.MetronomeMark(number=120))
s.append(meter.TimeSignature('4/4'))

# 创建一个列表来存储待处理的音符信息
pending_notes = []

# 添加音符到音乐流
for i in range(len(onset_times) - 1):
    start_time = onset_times[i]
    end_time = onset_times[i + 1]
    duration = end_time - start_time
    quarter_length = duration * 2
    
    frame_start = librosa.time_to_frames(start_time, sr=sr)
    frame_end = librosa.time_to_frames(end_time, sr=sr)
    
    if frame_start < pitches.shape[1] and frame_end < pitches.shape[1]:
        pitch_segment = pitches[:, frame_start:frame_end]
        magnitude_segment = magnitudes[:, frame_start:frame_end]
        
        if np.any(pitch_segment > 0):
            max_magnitude_idx = np.unravel_index(magnitude_segment.argmax(), magnitude_segment.shape)
            strongest_pitch = pitch_segment[max_magnitude_idx[0], max_magnitude_idx[1]]
            strongest_magnitude = magnitude_segment[max_magnitude_idx[0], max_magnitude_idx[1]]
            
            if strongest_pitch > 0 and strongest_magnitude > np.mean(magnitudes) * 50:
                pitch_name = librosa.hz_to_note(strongest_pitch)
                pitch_name = pitch_name.replace('♯', '#').replace('♭', 'b')
                
                # 检查是否可以与前一个音符合并
                if pending_notes and pending_notes[-1]['pitch'] == pitch_name:
                    # 合并音符，更新持续时间
                    pending_notes[-1]['duration'] += quarter_length
                else:
                    # 添加新的音符信息
                    pending_notes.append({
                        'pitch': pitch_name,
                        'start_time': start_time,
                        'duration': quarter_length
                    })

# 处理最后一个时间段
if len(onset_times) > 0:
    start_time = onset_times[-1]
    end_time = librosa.get_duration(y=y, sr=sr)
    duration = end_time - start_time
    quarter_length = duration * 2
    
    frame_start = librosa.time_to_frames(start_time, sr=sr)
    frame_end = librosa.time_to_frames(end_time, sr=sr)
    
    if frame_start < pitches.shape[1] and frame_end < pitches.shape[1]:
        pitch_segment = pitches[:, frame_start:frame_end]
        magnitude_segment = magnitudes[:, frame_start:frame_end]
        
        if np.any(pitch_segment > 0):
            max_magnitude_idx = np.unravel_index(magnitude_segment.argmax(), magnitude_segment.shape)
            strongest_pitch = pitch_segment[max_magnitude_idx[0], max_magnitude_idx[1]]
            
            if strongest_pitch > 0:
                pitch_name = librosa.hz_to_note(strongest_pitch)
                pitch_name = pitch_name.replace('♯', '#').replace('♭', 'b')
                pending_notes.append({
                    'pitch': pitch_name,
                    'start_time': start_time,
                    'duration': quarter_length
                })

# 将合并后的音符添加到音乐流中
for note_info in pending_notes:
    n = note.Note(note_info['pitch'])
    n.duration.quarterLength = note_info['duration']
    s.insert(note_info['start_time'], n)
    print(f"添加音符: {note_info['pitch']}, 时长: {note_info['duration']}, 时间点: {note_info['start_time']}")

# 5. 保存为MIDI文件
try:
    print("正在保存MIDI文件...")
    s.write('midi', fp='output.mid')
    print("MIDI文件保存成功")
    
    # 检查生成的音符数量
    note_count = len(s.flatten().notes)
    print(f"生成的音符数量: {note_count}")
    
    if note_count == 0:
        print("警告：没有检测到任何音符！")
        exit(1)
    
    # 6. 渲染五线谱
    print("正在解析MIDI文件...")
    score = converter.parse('output.mid')
    
    print("正在显示五线谱...")
    # 直接使用 MIDI 格式显示
    score.show('midi')
    print("已使用MIDI格式打开文件")
    
except Exception as e:
    print(f"发生错误: {str(e)}")
    # 打印更详细的错误信息
    import traceback
    traceback.print_exc()
