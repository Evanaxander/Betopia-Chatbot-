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

-----------------------------------------------------------------------------------------------------------------------------


Launch Documentation: Betopia Assistant
Follow these steps every time you restart your computer or open a new terminal to ensure the "Neural Interface" remains online.

1. Open the Project Folder
Open your terminal (Command Prompt or PowerShell) and navigate to the root folder of your project.

DOS

cd "C:\Users\abire\Documents\Evan\Betopia\betopia-rag-chatbot v5"
2. Activate the Environment
This is the most important step. You must "enter" the environment where the libraries (Whisper, OpenAI, etc.) are installed.

Command Prompt (CMD):

DOS

.\venv\Scripts\activate
PowerShell:

PowerShell

.\venv\Scripts\Activate.ps1
Note: You will know it is working when you see (venv) appear in parentheses at the start of your command line.

3. Navigate to the Source Code
Because your files are nested, move into the folder that contains the app directory.

DOS

cd "betopia-rag-chatbot v5"
4. Run the Chatbot
Launch the entry point using the path we identified earlier.

DOS

python app/main.py