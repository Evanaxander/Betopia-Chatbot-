import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import torch
import keyboard  # pip install keyboard

_stt_model = None

def record_and_transcribe(fs=16000):
    global _stt_model
    if _stt_model is None:
        print("📥 Loading Whisper Turbo...")
        _stt_model = whisper.load_model("turbo")

    print("\n🔴 Recording... (Press ENTER to stop)")
    recording = []
    
    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    # Start an input stream
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        input() # Wait for user to press Enter

    print("🛑 Recording stopped. Transcribing...")
    audio_data = np.concatenate(recording, axis=0)
    
    temp_file = "temp_in.wav"
    write(temp_file, fs, audio_data)
    
    result = _stt_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return result["text"].strip()