import numpy as np
from kokoro_onnx import Kokoro
import sounddevice as sd
import os

_tts = None

def speak_text(text):
    global _tts
    try:
        if _tts is None:
            model_path = "kokoro-v0_19.onnx"
            voices_path = "voices.json"
            _tts = Kokoro(model_path, voices_path)

        all_voices = list(_tts.get_voices())
        # Try to find 'af_heart' or 'af_bella' for better accuracy
        preferred = ["af_heart", "af_bella", "af_nicole", "af"]
        chosen_voice = next((v for v in preferred if v in all_voices), all_voices[0])
        
        # We split long text to prevent the TTS from getting "confused" or robotic
        if len(text) > 250:
            chunks = [text[i:i+250] for i in range(0, len(text), 250)]
        else:
            chunks = [text]

        for chunk in chunks:
            samples, sample_rate = _tts.create(chunk, voice=chosen_voice, speed=1.1)
            sd.play(samples, sample_rate)
            sd.wait() 
        
    except Exception as e:
        print(f"🔊 TTS Error: {e}")