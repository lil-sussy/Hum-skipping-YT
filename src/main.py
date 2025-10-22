import yt_dlp
import ffmpeg
import os
import argparse



def youtube_to_wav(url, output_wav):
    # Download audio as best quality m4a (no video)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    # with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #     ydl.download([url])

    # Convert temp_audio.m4a to WAV using ffmpeg-python
    input_audio = 'temp_audio.m4a'
    ffmpeg.input(input_audio).output(output_wav, format='wav').run(overwrite_output=True)

    os.remove(input_audio)
    print(f"WAV file saved as {output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube audio as WAV")
    parser.add_argument('url', help='YouTube video URL')
    # parser.add_argument('output', help='Output WAV file name')
    args = parser.parse_args()
    youtube_to_wav(args.url, "cringe.wav")
