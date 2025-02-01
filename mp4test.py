from pydub import AudioSegment
import os

# 手动指定 FFmpeg 路径
ffmpeg_path = "D:\\25winter\\toys\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe"  # 替换为你的 FFmpeg 路径
ffprobe_path = "D:\\path\\to\\ffmpeg\\bin\\ffprobe.exe"  # 替换为你的 FFprobe 路径

# 设置环境变量
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# 加载 .m4a 文件
audio = AudioSegment.from_file("test2.m4a", format="m4a")

# 导出为 .wav 文件
audio.export("output2.wav", format="wav")

print("转换完成！")