import yt_dlp
from pathlib import Path
from pydub import AudioSegment

def dowload_audio(url : str) -> str:

  download_dir = Path("downloads")
  download_dir.mkdir(exist_ok=True)

  ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": str(download_dir / "%(title)s.%(ext)s"),
    "postprocessors": [
        {
          "key": "FFmpegExtractAudio",
          "preferredcodec": "wav",
          "preferredquality": "192",
        }
      ],
    "quiet": True,
  }
  
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url=url,download=True)
    filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")

  return filename

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
  audio = AudioSegment.from_wav(wav_path)

  chunks = []
  chunk_ms = chunk_minutes * 60 * 1000 
  overlap_ms = 5 * 1000

  for i, start in enumerate(range(0, len(audio), chunk_ms)):
    start_time = max(0, start - overlap_ms)
    end_time = start + chunk_ms

    chunk = audio[start_time:end_time]

    chunk_path = f"{wav_path}_chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")

    chunks.append(chunk_path)
    
  return chunks

def process_input(source : str) -> list:

  if source.startswith(("http://", "https://")):
    print("Downloading audio....")
    wav_path = dowload_audio(source)
  else:
    print("Detected local file.")
    wav_path = source

  print("Chunking audio file.")
  chunks = chunk_audio(wav_path)

  print(f"Audio ready — {len(chunks)} chunk(s) created.")
  return chunks

