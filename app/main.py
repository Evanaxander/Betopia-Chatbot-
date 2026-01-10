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
except ModuleNotFoundError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

# --- 3. INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 4. BACKEND & INTENT HELPERS ---

def save_lead_to_backend(data):
    leads_file = os.path.join(root_dir, "data", "leads.json")
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    leads = []
    if not os.path.exists(os.path.dirname(leads_file)):
        os.makedirs(os.path.dirname(leads_file))
    if os.path.exists(leads_file):
        try:
            with open(leads_file, 'r', encoding='utf-8') as f: leads = json.load(f)
        except: leads = []
    leads.append(data)
    with open(leads_file, 'w', encoding='utf-8') as f: json.dump(leads, f, indent=4)
    print(f"\n📂 DATA SECURELY SAVED: {os.path.abspath(leads_file)}")

def check_intent(user_input, context_mission):
    """Uses LLM to classify if user is agreeing or interested."""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an intent classifier. Task: {context_mission}. If the user is agreeing (e.g., 'yeah', 'I would love to', 'proceed', 'sure', 'yes', 'okay'), reply 'YES'. Otherwise reply 'NO'."},
                {"role": "user", "content": user_input}
            ]
        )
        return "YES" in res.choices[0].message.content.upper()
    except:
        agree_words = ["yes", "yeah", "yep", "sure", "ok", "love", "would", "correct"]
        return any(word in user_input.lower() for word in agree_words)

# --- 5. UPLOAD & SYNC ---

def handle_manual_upload():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    root.destroy()
    if file_path:
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
                all_content.extend([f"[Source: {filename}] {c}" for c in chunks])
    return all_content

# --- 6. THE MAIN CHAT LOOP ---

def start_chat_loop():
    content = sync_initial_knowledge()
    index = create_faiss_index(embed_texts(content)) if content else None

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
        bot_is_talking = is_audio_playing()
        raw_input = input("You: ")
        
        if bot_is_talking:
            stop_audio()
            if not raw_input.strip(): continue 

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
            new_chunks = handle_manual_upload()
            if new_chunks:
                content.extend(new_chunks)
                index = create_faiss_index(embed_texts(content))
                print("✅ Knowledge Base Updated!")
            continue

        # --- STEP A: HANDLE ACTIVE BOOKING STATES ---
        if booking_state:
            # Topic switch detection
            if any(q in user_input.lower() for q in ["what", "who", "tell me", "how", "why"]) and booking_state != "VERIFY":
                print("Assistant: Switching back to your query.")
                booking_state = None
            else:
                if booking_state == "NAME":
                    temp_lead['name'] = user_input
                    msg = "Assistant: Thank you. Could you please provide your **Contact Number**?"
                elif booking_state == "PHONE":
                    temp_lead['phone'] = user_input
                    msg = "Assistant: Understood. And what is your **Professional Email Address**?"
                elif booking_state == "EMAIL":
                    temp_lead['email'] = user_input
                    msg = "Assistant: Could you please provide the designation and functional role of the primary attendee you would love to have a discussion with?"
                elif booking_state == "POSITION":
                    temp_lead['position'] = user_input
                    msg = "Assistant: What is the primary **Business Objective** for this meeting?"
                elif booking_state == "REASON":
                    temp_lead['reason'] = user_input
                    msg = f"\n--- SUMMARY ---\nName: {temp_lead['name']}\nPhone: {temp_lead['phone']}\nEmail: {temp_lead['email']}\nPosition: {temp_lead['position']}\nReason: {temp_lead['reason']}\n----------------\nIs this correct?"
                    booking_state = "VERIFY"
                    print(msg); continue
                elif booking_state == "VERIFY":
                    if check_intent(user_input, "Verify lead info"):
                        save_lead_to_backend(temp_lead)
                        msg = "Assistant: Verified. Our delegation will contact you shortly."
                    else:
                        msg = "Assistant: Information discarded. How else can I help?"
                    booking_state = None; temp_lead = {}
                
                # Progression logic
                if booking_state == "NAME": booking_state = "PHONE"
                elif booking_state == "PHONE": booking_state = "EMAIL"
                elif booking_state == "EMAIL": booking_state = "POSITION"
                elif booking_state == "POSITION": booking_state = "REASON"
                
                print(msg)
                if is_voice_mode: threading.Thread(target=speak_text, args=(msg,), daemon=True).start()
                continue

        # --- STEP B: HANDLE "YES" TO INITIAL OFFER ---
        if last_response_had_offer:
            if check_intent(user_input, "Accept consultation invitation"):
                msg = "Assistant: Excellent. To initiate coordination, what is your **Full Name**?"
                print(msg)
                if is_voice_mode: threading.Thread(target=speak_text, args=(msg,), daemon=True).start()
                booking_state = "NAME"
                last_response_had_offer = False
                continue
            else:
                last_response_had_offer = False # User ignored offer, process as RAG

        # --- STEP C: RAG PROCESSING ---
        try:
            local_retrieved = retrieve_chunks(user_input, content, index, lambda q: embed_texts([q]), k=10)
            full_context = "--- DOCUMENT DATA ---\n" + "\n".join(local_retrieved)
            history_str = "\n".join([f"You: {h[0]}\nBot: {h[1]}" for h in history])
            
            system_msg = (
                "You are an Executive Business Consultant. Use DOCUMENT DATA for facts. "
                "Tone: Professional, corporate, and authoritative. Avoid fluff. "
                "If info is missing, say: 'I do not have specific internal documentation on this matter.'"
            )
            
            res = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"History:\n{history_str}\n\nContext:\n{full_context}\n\nQuestion: {user_input}"}
                ],
                temperature=0.1
            )
            answer = res.choices[0].message.content

            # Offer logic: Only if query is about products or services
            biz_words = ["software", "product", "service", "system", "erp", "ai", "pos", "hrm", "betopia"]
            if any(w in user_input.lower() for w in biz_words) or any(w in answer.lower() for w in biz_words):
                if "do not have specific internal documentation" not in answer:
                    invitation = (
                        "\n\nBased on your interest in our solutions, I can facilitate a formal consultation "
                        "between you and a specialized Betopia delegation. Would you like to proceed with scheduling?"
                    )
                    answer += invitation
                    last_response_had_offer = True

            print(f"\nAssistant: {answer}")
            if is_voice_mode: threading.Thread(target=speak_text, args=(answer,), daemon=True).start()
            
            history.append((user_input, answer))
            if len(history) > 5: history.pop(0)

        except Exception as e: 
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chat_loop()