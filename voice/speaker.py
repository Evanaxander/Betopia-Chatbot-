import os
import json
import numpy as np
import onnxruntime as ort
import sounddevice as sd
import threading
import re
from misaki import en

_is_playing = False

class TTSWorker:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, "kokoro-v0_19.onnx")
        voices_path = os.path.join(cur_dir, "voices.json")

        if not os.path.exists(model_path):
            print(f"⚠️ TTS Error: Model not found at {model_path}")
            self.session = None
            return

        # 1. Initialize ONNX
        self.session = ort.InferenceSession(model_path)
        
        # 2. Initialize Phonemizer (Using the stable 'en' module from misaki)
        # This translates "Hello" -> phonemes
        try:
            self.g2p = en.G2P()
        except Exception as e:
            print(f"⚠️ Phonemizer Init Error: {e}. TTS may be silent.")
            self.g2p = None
        
        with open(voices_path, "r") as f:
            self.voices = json.load(f)

        # 3. Voice Selection
        v_name = "af_heart" if "af_heart" in self.voices else list(self.voices.keys())[0]
        self.voice_preset = np.array(self.voices[v_name], dtype=np.float32)

        # 4. Official Kokoro v0.19 Vocabulary
        characters = " $abcdðefghijkltuvwxyzɑɐɒæβɔɕçdʒðéəɚɛɜfɡɢħɥiɪjʝkɭɬɫmɱnɲŋɳoɔœøpɸrɾɻʀsʂʃθtʈuʊvʋwχyʏzʐʑʒʔ,.;:!?—\"() "
        self.vocab = {c: i for i, c in enumerate(characters)}

    def generate_and_play(self, text):
        global _is_playing
        if not self.session or not self.g2p or not text.strip():
            return

        try:
            _is_playing = True
            
            # Clean Markdown
            clean_text = re.sub(r'[*#_]', '', text)
            
            # Step 1: Convert English to Phonemes
            phonemes = self.g2p(clean_text)
            
            # Step 2: Map Phonemes to IDs
            # Kokoro expects a start (0) and end (0) token
            input_ids = [0]
            for p in phonemes:
                for char in p.phonemes:
                    if char in self.vocab:
                        input_ids.append(self.vocab[char])
            input_ids.append(0)
            
            if len(input_ids) <= 2:
                return

            # Step 3: Run Inference
            ort_inputs = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "style_vec": self.voice_preset,
                "speed_ratio": np.array([1.1], dtype=np.float32)
            }

            outputs = self.session.run(None, ort_inputs)
            audio = outputs[0]

            # Step 4: Normalize Volume
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Step 5: Play Audio
            sd.play(audio, 24000)
            sd.wait() 
            
        except Exception as e:
            print(f"⚠️ TTS Runtime Error: {e}")
        finally:
            _is_playing = False

tts_worker = TTSWorker()

def speak_text(text):
    if text.strip():
        threading.Thread(target=tts_worker.generate_and_play, args=(text,), daemon=True).start()

def stop_audio():
    global _is_playing
    sd.stop()
    _is_playing = False

def is_audio_playing():
    return _is_playing