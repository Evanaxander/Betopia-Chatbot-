import os
import threading
import io
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_is_playing_flag = False

def speak_text(text, voice="onyx"):
    """
    OpenAI Voices: alloy, echo, fable, onyx, nova, shimmer.
    Onyx is the recommended professional executive voice.
    """
    global _is_playing_flag
    
    def run_tts():
        global _is_playing_flag
        try:
            _is_playing_flag = True
            # tts-1 provides the lowest latency for real-time applications
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            # Use pydub to handle the byte stream and audio playback
            byte_stream = io.BytesIO(response.content)
            audio = AudioSegment.from_file(byte_stream, format="mp3")
            play(audio)
            
        except Exception as e:
            print(f"❌ Audio Sync Fault: {e}")
        finally:
            _is_playing_flag = False

    # Daemon thread ensures the UI remains responsive during playback
    threading.Thread(target=run_tts, daemon=True).start()

def stop_audio():
    global _is_playing_flag
    _is_playing_flag = False

def is_audio_playing():
    return _is_playing_flag