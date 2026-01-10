import sys
import os
import json
import threading
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import pymupdf

# --- 1. PATH LOGIC ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
if root_dir not in sys.path: 
    sys.path.insert(0, root_dir)

# --- 2. CUSTOM MODULE IMPORTS ---
try:
    from voice.listener import record_and_transcribe
    from voice.speaker import speak_text, stop_audio, is_audio_playing
    from rag.chunker import chunk_text
    from rag.embeddings import embed_texts
    from rag.vector_store import create_faiss_index
    from rag.retriever import retrieve_chunks
    from rag.prompt import build_prompt
except ModuleNotFoundError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

# --- 3. INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 4. UPLOAD HELPER ---
def handle_manual_upload():
    """Opens a file dialog to pick a PDF and adds it to the RAG content."""
    root = tk.Tk()
    root.withdraw() 
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    root.destroy()
    
    if file_path:
        print(f"📄 Processing new file: {os.path.basename(file_path)}")
        text = ""
        with pymupdf.open(file_path) as doc:
            for page in doc: text += page.get_text() + "\n"
        return chunk_text(text)
    return []

def sync_initial_knowledge():
    all_content = []
    data_path = os.path.join(root_dir, "data")
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.lower().endswith(".pdf"):
                text = ""
                with pymupdf.open(os.path.join(data_path, filename)) as doc:
                    for page in doc: text += page.get_text() + "\n"
                chunks = chunk_text(text)
                all_content.extend(chunks)
    return all_content

# --- 5. THE MAIN CHAT LOOP ---

def start_chat_loop():
    content = sync_initial_knowledge()
    index = create_faiss_index(embed_texts(content)) if content else None

    print("\n" + "="*50)
    print("🤖 ANALYTICAL ASSISTANT ACTIVE")
    print("👉 UPLOAD: Type 'upload' to add a new PDF.")
    print("👉 VOICE: Press ENTER on an empty line.")
    print("👉 STOP AUDIO: Press ENTER while bot is speaking.")
    print("="*50 + "\n")
    
    history = []
    booking_state = None
    temp_lead = {}
    
    while True:
        bot_is_talking = is_audio_playing()
        raw_input = input("You: ")
        
        if bot_is_talking:
            stop_audio()
            if not raw_input.strip():
                print("🛑 Audio Interrupted.")
                continue 

        user_input = raw_input.strip()

        # --- UPLOAD TRIGGER ---
        if user_input.lower() == "upload":
            new_chunks = handle_manual_upload()
            if new_chunks:
                content.extend(new_chunks)
                index = create_faiss_index(embed_texts(content))
                print("✅ Knowledge Base Updated with new document!")
            continue

        is_voice_mode = False
        if not user_input:
            is_voice_mode = True
            print("🎤 [VOICE MODE] Listening...")
            user_input = record_and_transcribe()
            if not user_input: continue
            print(f"You (Voice): {user_input}")

        if user_input.lower() == "exit": break

        # --- BOOKING STATE LOGIC ---
        if booking_state:
            if any(q in user_input.lower() for q in ["what", "who", "tell me", "how", "why", "about"]):
                booking_state = None 
            else:
                if booking_state == "NAME":
                    temp_lead['name'] = user_input
                    print("Assistant: Got it. Phone Number?")
                    booking_state = "PHONE"; continue
                elif booking_state == "PHONE":
                    temp_lead['phone'] = user_input
                    print("Assistant: Email address?"); booking_state = "EMAIL"; continue
                elif booking_state == "EMAIL":
                    temp_lead['email'] = user_input
                    print("Assistant: Thank you. We will contact you soon.")
                    booking_state = None; continue

        # --- IMPROVED RAG PROCESSING ---
        try:
            # 1. Expand the search: Increase 'k' to 10 for better row-level retrieval
            local_retrieved = retrieve_chunks(user_input, content, index, lambda q: embed_texts([q]), k=10)
            full_context = "--- DOCUMENT DATA ---\n" + "\n".join(local_retrieved)
            
            # 2. Adjusted System Prompt: More analytical and focused on strict data
            system_msg = (
                "You are a professional Data Analyst. Use the provided DOCUMENT DATA to answer the query. "
                "If the user mentions a specific entity (like 'Customer 1'), search the context thoroughly for that label. "
                "Provide a detailed summary of the events recorded. "
                "Do NOT mention Betopia or scheduling unless specifically asked about company services."
            )
            
            res = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Context Data:\n{full_context}\n\nQuestion: {user_input}"}
                ],
                temperature=0.1 # Lower temperature for higher accuracy and less hallucination
            )
            answer = res.choices[0].message.content
            
            # 3. Clean any potential sales boilerplate if it's a data-specific query
            if any(dk in user_input.lower() for dk in ["customer", "report", "data", "what happened"]):
                answer = answer.split("Based on your interest")[0].split("Would you like to proceed")[0].strip()

            print(f"\nAssistant: {answer}")

            if is_voice_mode:
                threading.Thread(target=speak_text, args=(answer,), daemon=True).start()
            
            # Trigger booking ONLY for sales-intent queries
            product_words = ["product", "service", "pricing", "erp", "app", "buy"]
            if any(pw in user_input.lower() for pw in product_words):
                if "consultation" in answer.lower() and any(w in user_input.lower() for w in ["yes", "book", "schedule"]):
                    print("\nAssistant: Great. What is your Full Name?")
                    booking_state = "NAME"
            
            history.append((user_input, answer))
            if len(history) > 5: history.pop(0)

        except Exception as e: 
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat_loop()