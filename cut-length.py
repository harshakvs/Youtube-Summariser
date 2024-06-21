from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

def cut_video(input_path, output_path, start_time=5, end_time=15):
    video_clip = VideoFileClip(input_path)
    video_clip = video_clip.subclip(start_time, end_time)
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def cut_audio(input_path, output_path, start_time=90, end_time=100):
    audio_segment = AudioSegment.from_wav(input_path)
    audio_segment = audio_segment[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
    audio_segment.export(output_path, format='wav')

# Example usage
video_input_path = 'D:\Youtube-Summariser\Youtube-Summariser\Dave2D.mp4'
video_output_path = 'D:\Youtube-Summariser\Youtube-Summariser\output_video.mp4'
cut_video(video_input_path, video_output_path)

audio_input_path = 'D:\\Youtube-Summariser\\Youtube-Summariser\\temp.wav'
audio_output_path = 'D:\Youtube-Summariser\Youtube-Summariser\output_audio5.wav'
cut_audio(audio_input_path, audio_output_path)
