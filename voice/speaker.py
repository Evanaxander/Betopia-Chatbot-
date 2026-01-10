import subprocess
import os

_current_proc = None

def is_audio_playing():
    """Checks if the PowerShell TTS process is currently active."""
    global _current_proc
    return _current_proc is not None and _current_proc.poll() is None

def speak_text(text):
    """Starts a non-blocking background process for TTS."""
    global _current_proc
    clean_text = text.replace("'", "").replace('"', "").replace("\n", " ").replace("\r", " ")
    try:
        _current_proc = subprocess.Popen(
            [
                "powershell", 
                "-Command", 
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
    except Exception as e:
        print(f"❌ TTS Error: {e}")

def stop_audio():
    """Instantly kills the speech process tree."""
    global _current_proc
    if is_audio_playing():
        try:
            subprocess.call(
                ['taskkill', '/F', '/T', '/PID', str(_current_proc.pid)], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except:
            pass
    _current_proc = None