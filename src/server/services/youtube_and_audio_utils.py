import yt_dlp
import os
import ffmpeg
import json
from config import *

FAKE_YOUTUBE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
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
    

def get_cached_video(video_id):
    cache_path_wav = CONFIG_VIDEO_CACHE + video_id + ".wav"
    cache_path_wav = os.path.expanduser(cache_path_wav)
    if os.path.exists(cache_path_wav):
        print(f"Found cached WAV for video ID {video_id}")
        video_duration = ffmpeg.probe(cache_path_wav)['format']['duration']
        return cache_path_wav, video_duration
    return None, -1


def youtube_dl_wav(video_url, video_id=None):
    if not video_id:
        video_id = video_url.split("?v=")[1]
    wav_path, video_duration = get_cached_video(video_id)
    if wav_path:
        return wav_path, video_duration
    init_video_cache()
    cache_path = CONFIG_VIDEO_CACHE + video_id
    # Download audio as best quality m4a (no video)
    http_headers = json.loads(FAKE_YOUTUBE_HEADERS)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': cache_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': False,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': http_headers.get('User-Agent', ''),
            'Accept-Language': http_headers.get('Accept-Language', ''),
            'Referer': http_headers.get('Referer', ''),
        },
        'cookiefile': 'src/server/fake_youtube_cookies.txt',
    }
    video_duration = -1
    cache_m4a = os.path.expanduser(cache_path + ".m4a")
    cache_wav = os.path.expanduser(cache_path + ".wav")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"[Extract Audio] Downloaded audio to {cache_m4a}")
        # Convert temp_audio.m4a to WAV using ffmpeg-python
        
    ffmpeg.input(cache_m4a).output(cache_wav, format='wav').run(overwrite_output=True)
    print(f"[Extract Audio] Converted to WAV: {cache_wav}")
    # os.remove(cache_path+".m4a")
    video_duration = ffmpeg.probe(cache_wav + "")['format']['duration']
    print(f"[Extract Audio] WAV file saved as {cache_wav}")
    return cache_wav, video_duration