import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# --- PRE-LOAD CONFIGURATION ---
# This must happen before importing RAG modules to fix the Chroma API error
load_dotenv()
from openai import OpenAI

# --- 1. PATH RESOLUTION ---
current_file = os.path.abspath(__file__)
app_folder = os.path.dirname(current_file)
project_root = os.path.dirname(app_folder)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. MODULE IMPORTS ---
try:
    from rag.vector_store import collection, query_db 
    from voice.speaker import speak_text, stop_audio, is_audio_playing
    from voice.listener import record_and_transcribe
    print("✅ System: Neural Interface Online.")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 3. DATA PERSISTENCE ---
def save_lead_to_backend(data):
    """Saves lead info to data/leads.json in the project root."""
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    leads_file = os.path.join(data_dir, "leads.json")
    
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    leads = []
    if os.path.exists(leads_file):
        try:
            with open(leads_file, 'r', encoding='utf-8') as f:
                leads = json.load(f)
        except:
            leads = []
    
    leads.append(data)
    with open(leads_file, 'w', encoding='utf-8') as f:
        json.dump(leads, f, indent=4)
    print(f"\n📂 DATA SECURELY LOGGED: {os.path.abspath(leads_file)}")

def check_intent(user_input):
    """Refined intent check for confirmation flow."""
    clean = user_input.strip().lower()
    positive = ["yes", "yeah", "yep", "correct", "sure", "ok", "confirm", "please"]
    return any(word in clean for word in positive)

# --- 4. EXECUTIVE INTERACTION LOOP ---
def start_bot():
    print("\n" + "═"*60)
    print("  CHATOPIA!  ")
    print("  Your Thought, my response.  ")
    print("═"*60 + "\n")

    booking_state = None
    temp_lead = {}
    last_response_had_offer = False

    while True:
        if is_audio_playing():
            stop_audio()

        # Command-line interface
        raw_input = input("Your Query Here > ").strip()

        if raw_input.lower() in ["exit", "quit", "terminate"]: 
            print("Session terminated. Have a productive day."); break
            
        if not raw_input:
            # Voice Mode
            user_input = record_and_transcribe()
            if not user_input: continue
            print(f"Transcript > {user_input}")
        else:
            user_input = raw_input

        # --- A. STRATEGIC BOOKING WORKFLOW ---
        if booking_state:
            if booking_state == "NAME":
                temp_lead['name'] = user_input
                msg = f"Thank you, {user_input}. May I have a contact number for our records?"
                booking_state = "PHONE"
            elif booking_state == "PHONE":
                temp_lead['phone'] = user_input
                msg = "Splendid. Lastly, please provide your professional email address."
                booking_state = "EMAIL"
            elif booking_state == "EMAIL":
                temp_lead['email'] = user_input
                msg = (f"Please verify these credentials:\n"
                       f"• Representative: {temp_lead['name']}\n"
                       f"• Contact: {temp_lead['phone']}\n"
                       f"• ID: {temp_lead['email']}\n"
                       f"Is this information correct?")
                booking_state = "VERIFY"
            elif booking_state == "VERIFY":
                if check_intent(user_input):
                    save_lead_to_backend(temp_lead)
                    msg = "Credentials validated. A senior consultant will contact you shortly."
                else:
                    msg = "I have cleared the session data. How may I assist you otherwise?"
                booking_state = None; temp_lead = {}

            print(f"Executive Assistant > {msg}"); speak_text(msg)
            continue

        # --- B. CONSULTATION OFFER HANDLING ---
        if last_response_had_offer:
            if check_intent(user_input):
                msg = "Certainly. To initiate the briefing, please state your full name."
                print(f"Executive Assistant > {msg}"); speak_text(msg)
                booking_state = "NAME"
                last_response_had_offer = False
                continue
            last_response_had_offer = False

        # --- C. RAG-DRIVEN RESPONSE ---
        try:
            chunks = query_db(user_input, n_results=3)
            context = "\n".join(chunks)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are the Betopia Executive Consultant. Your style is professional, "
                        "authoritative, and concise. Use provided context to offer strategic "
                        "insights. Avoid flowery language; speak as a business partner."
                    )},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {user_input}"}
                ]
            )
            ans = response.choices[0].message.content
            
            # Smart offer logic for business conversion
            if any(k in user_input.lower() for k in ["how", "help", "solution", "service", "betopia"]):
                ans += "\n\nWould you like to authorize a formal strategic consultation?"
                last_response_had_offer = True

            print(f"Executive Assistant > {ans}"); speak_text(ans)
        except Exception as e:
            print(f"❌ System Fault: {e}")

if __name__ == "__main__":
    start_bot()