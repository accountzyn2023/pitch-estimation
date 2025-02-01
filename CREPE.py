import crepe
import librosa
import numpy as np
from music21 import stream, note, tempo, converter, meter
from scipy.io import wavfile

# 配置 MuseScore
import music21
us = music21.environment.UserSettings()
us['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
us['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

def hz_to_midi_note(hz):
    """将赫兹转换为MIDI音符编号"""
    if hz <= 0:
        return None
    return 69 + 12 * np.log2(hz/440.0)

def create_score(audio_path, confidence_threshold=0.7, min_note_length=0.1):
    """
    使用CREPE创建乐谱
    
    参数:
    audio_path: 音频文件路径
    confidence_threshold: 音高检测的置信度阈值
    min_note_length: 最小音符长度（秒）
    """
    # 1. 加载音频文件
    print("加载音频文件...")
    sr, audio = wavfile.read(audio_path)
    
    # 确保音频是单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # 2. 使用CREPE进行音高检测
    print("正在进行音高检测...")
    time, frequency, confidence, activation = crepe.predict(
        audio, 
        sr, 
        viterbi=True,
        step_size=10  # 每10ms进行一次检测
    )
    
    # 3. 创建音乐流
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=120))
    s.append(meter.TimeSignature('4/4'))
    
    # 4. 处理音高序列
    current_note = None
    current_start = None
    pending_notes = []
    
    print("处理检测到的音高...")
    for i in range(len(time)):
        if confidence[i] >= confidence_threshold:
            midi_note = hz_to_midi_note(frequency[i])
            if midi_note is not None:
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
    
    # 5. 添加音符到音乐流
    print("添加音符到乐谱...")
    for note_info in pending_notes:
        n = note.Note(note_info['pitch'])
        n.duration.quarterLength = note_info['duration'] * 2  # 将秒转换为四分音符长度
        s.insert(note_info['start_time'], n)
        print(f"添加音符: MIDI音高 {note_info['pitch']}, 时长: {note_info['duration']:.2f}秒")
    
    return s

if __name__ == "__main__":
    try:
        # 创建乐谱
        score = create_score("output.wav")
        
        # 保存MIDI文件
        print("正在保存MIDI文件...")
        score.write('midi', fp='output.mid')
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
