
# 🤖 Chatopia: Strategic Executive Consultant

**Chatopia** is a high-performance RAG (Retrieval-Augmented Generation) chatbot designed for corporate strategic intelligence. It combines vector-based document retrieval with neural voice synthesis to provide a professional, voice-enabled consulting experience.

---

## 🏗️ Technical Architecture

Chatopia operates on a four-tier architecture to ensure low latency and high accuracy:

* **Voice Interface:** Powered by **OpenAI Whisper** (`whisper-1`) for transcription and **OpenAI TTS** (`tts-1`) for authoritative voice synthesis.
* **Vector Intelligence:** Documents are embedded via `text-embedding-3-small` and managed within **ChromaDB**.
* **Reasoning Engine:** **GPT-4o-mini** processes retrieved context to generate professional, concise business insights.
* **Data Persistence:** A state-machine booking flow captures and logs lead data to `data/leads.json`.

---

## 📂 Project Structure

```text
betopia-rag-chatbot/
├── app/
│   └── main.py          # Central logic & Executive interaction loop
├── rag/
│   ├── vector_store.py  # ChromaDB initialization & querying
│   └── chunker.py       # PDF text processing logic
├── voice/
│   ├── speaker.py       # OpenAI TTS implementation
│   └── listener.py      # OpenAI Whisper transcription
├── data/
│   └── leads.json       # Captured consultation requests
└── chroma_db/           # Persistent vector storage

```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

* **Python 3.10+**
* **FFmpeg** (Required for audio processing)
* **OpenAI API Key**

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here

```

### 3. Quick Start

```powershell
# Activate your virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## 🚀 Usage Guide

### Launching the System

Always ensure your virtual environment is active before starting. Navigate to the core directory and run:

```powershell
# Navigate to the inner source folder
cd "betopia-rag-chatbot v5"

# Run the application
python app/main.py

```

### Interaction Modes

* **⌨️ Text Mode:** Type your query directly into the prompt and press Enter.
* **🎤 Voice Mode:** Press **Enter** on an empty prompt to trigger the "Neural Interface." Press Enter again to stop recording.
* **💼 Consultation Booking:** If the AI suggests a strategic briefing, confirm with "Yes" to begin the data collection workflow.

---

## ⚠️ Maintenance & Troubleshooting

* **Secret Protection:** Never commit your `.env` file. Ensure it is listed in your `.gitignore`.
* **Neural Interface Offline:** If you see `ModuleNotFoundError`, verify that your `(venv)` is active in the terminal.
* **Audio Issues:** Ensure `ffmpeg` is installed and added to your system's Environment Variables (PATH).

