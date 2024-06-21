from pytube import YouTube
from moviepy.editor import AudioFileClip
import os
def download_video(url):
    yt = YouTube(url)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path="D:\\Youtube-Summariser\\Youtube-Summariser",filename='video.mp4')

def download_audio(url, output_path='D:/Youtube-Summariser/Youtube-Summariser/audio/audio.wav'):

    # Download the YouTube video
    yt = YouTube(url)
    video_stream = yt.streams.filter(only_audio=True).first()
    video_stream.download(output_path="D:/Youtube-Summariser/Youtube-Summariser/audio",filename='temp_audio.mp4')
# Example usage
video_url = 'https://www.youtube.com/watch?v=cn5DsF9FXls&t=199s'
download_video(video_url)

audio_url = 'https://www.youtube.com/watch?v=cHxOBS97fDU'
download_audio(audio_url)

def convert_audio_to_wav(input_path, output_path='D:/Youtube-Summariser/Youtube-Summariser/audio/audio.wav'):
    audio_clip = AudioFileClip(input_path)

    audio_clip.write_audiofile(output_path, codec='pcm_s16le', bitrate='192k')


# Assuming the video and audio have been downloaded
convert_audio_to_wav('D:/Youtube-Summariser/Youtube-Summariser/temp_audio.mp4')
