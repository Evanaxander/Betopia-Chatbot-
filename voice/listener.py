import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_and_transcribe(fs=16000):
    print("\n🔴 [SYSTEM LISTENING] (Press ENTER to finish)")
    recording = []
    
    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        input() 

    print("🛑 [PROCESSING AUDIO]...")
    audio_data = np.concatenate(recording, axis=0)
    temp_file = "temp_voice.wav"
    write(temp_file, fs, audio_data)
    
    try:
        with open(temp_file, "rb") as audio:
            # Using OpenAI API is 10x faster than local 'Turbo' model
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio
            )
        return transcript.text.strip()
    finally:
        if os.path.exists(temp_file): os.remove(temp_file)