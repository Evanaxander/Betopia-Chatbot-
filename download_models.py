from huggingface_hub import hf_hub_download
import shutil

# 1. TTS: Kokoro Models (NeuML is a reliable source for ONNX format)
print("📥 Downloading Kokoro TTS assets...")
hf_hub_download(repo_id="NeuML/kokoro-base-onnx", filename="model.onnx", local_dir=".")
hf_hub_download(repo_id="NeuML/kokoro-base-onnx", filename="voices.json", local_dir=".")

# Rename model to match standard code
import os
if os.path.exists("model.onnx"):
    os.rename("model.onnx", "kokoro-v0_19.onnx")

# 2. STT: Whisper Large-v3-Turbo
# Note: Whisper downloads its model automatically the first time it's called.
# The 'turbo' model is ~800MB (1.5GB on disk).
print("✅ TTS models ready. Whisper will download 'turbo' on first run.")