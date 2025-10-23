import yt_dlp
import os
import ffmpeg
from config import *


def extract_video_audio_and_info(video_url):
  video_id = video_url.split("?v=")[1]

  return {
    "video_id": video_id,
    "duration": 60.0,
    "sample_rate": 16000,
  }
  
  
def init_video_cache():
    os.makedirs(CONFIG_VIDEO_CACHE, exist_ok=True)
    # empties the cache dir
    for filename in os.listdir(CONFIG_VIDEO_CACHE):
        file_path = os.path.join(CONFIG_VIDEO_CACHE, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print("Video cache initialized.")
    
    


def youtube_dl_wav(video_url, video_id=None):
    video_id = video_url.split("?v=")[1]
    init_video_cache()
    cache_path = CONFIG_VIDEO_CACHE + video_id
    # Download audio as best quality m4a (no video)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': cache_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
        # Convert temp_audio.m4a to WAV using ffmpeg-python
        
        ffmpeg.input(cache_path).output(cache_path + ".wav", format='wav').run(overwrite_output=True)

        os.remove(cache_path)
        print(f"WAV file saved as {cache_path}.wav")
    return cache_path + ".wav"