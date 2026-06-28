from dotenv import load_dotenv
load_dotenv()
import os
from faster_whisper import WhisperModel
from sarvamai import SarvamAI
from pydub import AudioSegment
from pathlib import Path


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
_model = None

sarvam_model = SarvamAI()

def load_model():
  global _model

  if _model == None:
    print(f"Loading Whisper model: {WHISPER_MODEL} ...")
    _model = WhisperModel(
      WHISPER_MODEL,
      device="cpu",
      compute_type="int8"
    )
    print("Whisper model loaded.")
  
  return _model 


def transcribe_chunk_whisper(chunk_path : str) -> str:
  
  model = load_model()

  segments, _ = model.transcribe(chunk_path)  

  return " ".join(segment.text for segment in segments)


def send_to_sarvam(piece_path : str) -> str:
  model = sarvam_model

  with open(piece_path, "rb") as f:
    response = model.speech_to_text.translate(file=f, model="saaras:v2.5")

  return response.transcript


def transcribe_chunk_sarvam(chunk_path : str) -> str:
  
  chunk_secs = 25
  audio = AudioSegment.from_wav(chunk_path)

  chunk_ms = chunk_secs*1000
  overlap_ms = 5*1000

  total_pieces = (len(audio) + chunk_ms - 1)//chunk_ms

  full_text = ""

  for i, start in enumerate(range(0, len(audio), chunk_ms)):
    start_time = max(0, start - overlap_ms)
    end_time = start + chunk_ms

    piece = audio[start_time:end_time]
    piece_path = f"{chunk_path}_piece_{i}.wav"
    piece.export(piece_path,format="wav")

    try:
      print(f" -> Sarvam piece {i+1}/{total_pieces}")
      full_text += send_to_sarvam(piece_path)
    finally:
      if Path(piece_path).exists():
        Path(piece_path).unlink()
    
  return full_text.strip()


def transcribe_chunk(chunk_path : str, language : str) -> str:

  if language == "hinglish":
    return transcribe_chunk_sarvam(chunk_path)
  
  return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks : list, language : str = "english") -> str:

  engine = "SarvamAI" if language == "hinglish" else "Whisper"

  print(f"Using {engine} to transcribe")
  
  transcript = ""

  for i,chunk in enumerate(chunks):
    print(f"Transcribing chunk {i}:")
    transcript_chunk = transcribe_chunk(chunk, language)
    transcript+=transcript_chunk + " "

  return transcript
