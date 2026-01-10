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
    from rag.embeddings import sync_to_chroma
    from rag.vector_store import collection, query_db 
except ModuleNotFoundError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

# --- 3. INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 4. BACKEND HELPERS ---

def save_lead_to_backend(data):
    """Saves lead data into a persistent JSON file."""
    # Ensure directory exists
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    leads_file = os.path.join(data_dir, "leads.json")
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    leads = []
    if os.path.exists(leads_file):
        try:
            with open(leads_file, 'r', encoding='utf-8') as f:
                leads = json.load(f)
        except (json.JSONDecodeError, IOError):
            leads = []

    leads.append(data)
    
    with open(leads_file, 'w', encoding='utf-8') as f:
        json.dump(leads, f, indent=4)
    
    print(f"\n✅ DATA SECURELY SAVED TO: {leads_file}")

def check_intent(user_input, context_mission):
    """Uses LLM to classify if user is agreeing or interested."""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an intent classifier. Task: {context_mission}. If the user is agreeing, reply 'YES'. Otherwise reply 'NO'."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=2,
            temperature=0
        )
        return "YES" in res.choices[0].message.content.upper()
    except:
        return False

# --- 5. UPLOAD & SYNC ---

def handle_manual_upload():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    root.destroy()
    if file_path:
        filename = os.path.basename(file_path)
        text = ""
        with pymupdf.open(file_path) as doc:
            for page in doc: text += page.get_text() + "\n"
        chunks = chunk_text(text)
        sync_to_chroma(collection, chunks, filename)
        return True
    return False

def sync_initial_knowledge():
    data_path = os.path.join(root_dir, "data")
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.lower().endswith(".pdf"):
                existing = collection.get(where={"source": filename})
                if len(existing['ids']) == 0:
                    print(f"🆕 New file: {filename}. Indexing...")
                    text = ""
                    with pymupdf.open(os.path.join(data_path, filename)) as doc:
                        for page in doc: text += page.get_text() + "\n"
                    chunks = chunk_text(text)
                    sync_to_chroma(collection, chunks, filename)

# --- 6. THE MAIN CHAT LOOP ---

def start_chat_loop():
    sync_initial_knowledge()

    print("\n" + "="*50)
    print("🤖 BETOPIA EXECUTIVE CONSULTANT ACTIVE")
    print("👉 UPLOAD: Type 'upload' to add a PDF.")
    print("👉 VOICE: Press ENTER on an empty line.")
    print("="*50 + "\n")
    
    history = []
    booking_state = None
    temp_lead = {}
    last_response_had_offer = False 

    while True:
        if is_audio_playing():
            stop_audio()

        raw_input = input("You: ")
        user_input = raw_input.strip()
        is_voice_mode = False

        if not user_input:
            print("🎤 Listening...")
            user_input = record_and_transcribe()
            if not user_input: continue
            print(f"You (Voice): {user_input}")
            is_voice_mode = True

        if user_input.lower() == "exit": break

        if user_input.lower() == "upload":
            if handle_manual_upload(): print("✅ Knowledge Base Updated!")
            continue

        # --- STEP A: LEAD COLLECTION STATE MACHINE ---
        if booking_state:
            if not user_input: continue

            if booking_state == "NAME":
                temp_lead['name'] = user_input
                msg = f"Assistant: Thank you, {user_input}. Could you please provide your **Contact Number**?"
                booking_state = "PHONE"
            elif booking_state == "PHONE":
                temp_lead['phone'] = user_input
                msg = "Assistant: Understood. And what is your **Professional Email Address**?"
                booking_state = "EMAIL"
            elif booking_state == "EMAIL":
                temp_lead['email'] = user_input
                msg = "Assistant: What is your current **Designation or Job Title**?"
                booking_state = "POSITION"
            elif booking_state == "POSITION":
                temp_lead['position'] = user_input
                msg = "Assistant: What is the primary **Business Objective** for this meeting?"
                booking_state = "REASON"
            elif booking_state == "REASON":
                temp_lead['reason'] = user_input
                msg = (f"\n--- SUMMARY ---\nName: {temp_lead['name']}\nPhone: {temp_lead['phone']}\n"
                       f"Email: {temp_lead['email']}\nPosition: {temp_lead['position']}\n"
                       f"Reason: {temp_lead['reason']}\n----------------\nIs this information correct? (Yes/No)")
                booking_state = "VERIFY"
            elif booking_state == "VERIFY":
                # EXPERT FIX: Use local check + AI check for reliability
                is_confirmed = user_input.lower() in ["yes", "y", "correct", "yeah", "ok"] or \
                               check_intent(user_input, "Verify lead info")
                
                if is_confirmed:
                    save_lead_to_backend(temp_lead)
                    msg = "Assistant: Excellent. I have saved your details. A representative will reach out shortly."
                else:
                    msg = "Assistant: Understood. I have discarded that information. How else can I help?"
                booking_state = None; temp_lead = {}

            print(msg)
            if is_voice_mode: threading.Thread(target=speak_text, args=(msg,), daemon=True).start()
            continue

        # --- STEP B: HANDLE OFFER ACCEPTANCE ---
        if last_response_had_offer:
            if check_intent(user_input, "Accept consultation invitation"):
                msg = "Assistant: Excellent. To initiate coordination, what is your **Full Name**?"
                print(msg)
                if is_voice_mode: threading.Thread(target=speak_text, args=(msg,), daemon=True).start()
                booking_state = "NAME"; last_response_had_offer = False
                continue
            last_response_had_offer = False

        # --- STEP C: RAG PROCESSING WITH STREAMING ---
        try:
            local_retrieved = query_db(user_input, n_results=5)
            full_context = "--- DOCUMENT DATA ---\n" + "\n".join(local_retrieved)
            history_str = "\n".join([f"U: {h[0]}\nB: {h[1]}" for h in history[-3:]])
            
            response_stream = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "You are an Executive Business Consultant. Use DOCUMENT DATA for facts."},
                    {"role": "user", "content": f"History:\n{history_str}\n\nContext:\n{full_context}\n\nQuestion: {user_input}"}
                ],
                temperature=0.1,
                stream=True
            )

            print("\nAssistant: ", end="", flush=True)
            full_answer = ""
            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_answer += content

            # Lead Trigger
            biz_words = ["software", "product", "service", "betopia", "system"]
            if any(w in user_input.lower() for w in biz_words):
                invitation = "\n\nWould you like to schedule a formal consultation with a Betopia delegation?"
                print(invitation)
                full_answer += invitation
                last_response_had_offer = True

            if is_voice_mode: 
                threading.Thread(target=speak_text, args=(full_answer,), daemon=True).start()
            
            history.append((user_input, full_answer))
            print("\n")

        except Exception as e: 
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat_loop()