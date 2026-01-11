 Chatopia: Strategic Executive Consultant
Chatopia is a high-performance RAG (Retrieval-Augmented Generation) chatbot designed for corporate strategic intelligence. It combines vector-based document retrieval with neural voice synthesis to provide a professional, voice-enabled consulting experience.

 Technical Architecture
Chatopia operates on a four-tier architecture to ensure low latency and high accuracy:

Voice Interface: Uses OpenAI Whisper (whisper-1) for speech-to-text and OpenAI TTS (tts-1) for authoritative voice synthesis.

Vector Intelligence: Documents are processed and stored in ChromaDB using text-embedding-3-small embeddings.

Reasoning Engine: GPT-4o-mini processes retrieved context to generate professional, concise business insights.

Data Persistence: A state-machine booking flow captures and saves lead data to data/leads.json.

 Installation & Setup
1. Prerequisites
Python 3.10+

FFmpeg (required for pydub audio processing)

OpenAI API Key

2. Environment Configuration
Create a .env file in the root directory:

Code snippet

OPENAI_API_KEY=your_api_key_here
CHROMA_OPENAI_API_KEY=your_api_key_here
3. Installation
Bash

# Activate your virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
 Project Structure
Plaintext

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
 Usage Guide
Starting the System
Run the following command from the root directory:

PowerShell

python -m app.main
Interaction Modes
Text Mode: Type directly into the prompt.

Voice Mode: Press Enter on an empty prompt to start recording. Press Enter again to finish and send.

Consultation Booking: If the AI offers a consultation, type "Yes" to trigger the data collection workflow.