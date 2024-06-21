from moviepy.editor import VideoFileClip, AudioFileClip

def combine_audio_video(audio_path, video_path, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the video clip to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the result to the output file
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Example usage
audio_file = 'D:\Youtube-Summariser\Youtube-Summariser\output_audio5.wav'
video_file = 'D:\Youtube-Summariser\Youtube-Summariser\output_video.mp4'
output_file = 'D:\Youtube-Summariser\Youtube-Summariser\output_video1.mp4'

combine_audio_video(audio_file, video_file, output_file)
