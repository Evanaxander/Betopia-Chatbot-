import sys
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
from openai import OpenAI
import pymupdf

# --- 1. BULLETPROOF PATH LOGIC ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. CUSTOM MODULE IMPORTS ---
try:
    from voice.listener import record_and_transcribe
    from voice.speaker import speak_text
    from rag.chunker import chunk_text
    from rag.embeddings import embed_texts, describe_image
    from rag.vector_store import create_faiss_index, get_chroma_collection, delete_file_from_db
    from rag.retriever import retrieve_chunks
    from rag.prompt import build_prompt
except ModuleNotFoundError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("👉 Ensure you have empty __init__.py files in /rag and /voice folders.")
    sys.exit(1)

# --- 3. INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LIB_DIR, IMG_DIR, UPLOAD_DIR = "data", "data/images", "uploads"
for f in [LIB_DIR, IMG_DIR, UPLOAD_DIR]: 
    os.makedirs(os.path.join(root_dir, f), exist_ok=True)

# --- 4. CORE UTILITIES ---

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
    return text

def sync_universal_knowledge():
    print("🔄 Syncing Universal Brain...")
    collection = get_chroma_collection()
    all_content = []
    targets = {
        os.path.join(root_dir, LIB_DIR): "library", 
        os.path.join(root_dir, IMG_DIR): "visual", 
        os.path.join(root_dir, UPLOAD_DIR): "user_upload"
    }

    for folder, category in targets.items():
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isdir(path): continue
            
            delete_file_from_db(collection, filename)
            
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(path)
                chunks = chunk_text(text) if text else []
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"🖼️ Analyzing image: {filename}")
                desc = describe_image(path)
                chunks = [desc] if desc else []
            else: continue

            if chunks:
                collection.add(
                    documents=chunks,
                    metadatas=[{"source": filename, "type": category} for _ in chunks],
                    ids=[f"{category}_{filename}_{i}" for i in range(len(chunks))]
                )
                all_content.extend([f"[Source: {filename}] {c}" for c in chunks])
    return all_content

def web_search_stream(query):
    print(f"🌐 Searching the web for: {query}...")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join([f"[Source: Internet] {r}" for r in results])
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return ""

def handle_file_explorer_upload():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Select file to upload", filetypes=[("PDF/Images", "*.pdf *.png *.jpg *.jpeg")])
    root.destroy()

    if file_path:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(root_dir, UPLOAD_DIR, filename))
        print(f"✅ Uploaded: {filename}")
        return True
    return False

# --- 5. THE MAIN CHAT LOOP ---

def start_chat_loop(all_content, index):
    print("\n" + "="*50)
    print("🤖 BETOPIA ASSISTANT LIVE")
    print("👉 Type your query (Text response only)")
    print("👉 Press [ENTER] on empty line for VOICE (Voice response)")
    print("👉 Commands: 'upload' to add files, 'exit' to quit.")
    print("="*50 + "\n")
    
    history = []
    
    while True:
        user_input = input("You: ")
        cmd = user_input.strip().lower()
        
        if cmd == "exit": break
        
        should_speak = False # Default to silent
            
        # 🟢 MODE 1: VOICE INPUT
        if user_input == "":
            should_speak = True
            print("🎤 [VOICE MODE]")
            print("👉 Press [ENTER] to START recording...")
            input() 
            question = record_and_transcribe() # Wait for manual stop
            if not question:
                print("⚠️ No speech detected.")
                continue
            print(f"🗨️ You said: {question}")
        
        # 🔵 MODE 2: COMMANDS
        elif cmd == "upload":
            if handle_file_explorer_upload():
                all_content = sync_universal_knowledge()
                if all_content:
                    index = create_faiss_index(embed_texts(all_content))
            continue

        elif cmd == "debug":
            print(f"🧠 Brain Size: {len(all_content)} chunks.")
            continue

        # 🟡 MODE 3: TEXT INPUT
        else:
            question = user_input
            should_speak = False

        # --- AI PROCESSING ---
        try:
            search_query = question
            if history:
                res = client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[{"role": "user", "content": f"Query: {question}\nRewrite as search term:"}]
                )
                search_query = res.choices[0].message.content.strip()

            local_retrieved = retrieve_chunks(search_query, all_content, index, lambda q: embed_texts([q]), k=5)
            web_context = web_search_stream(search_query)

            full_context = "--- DOCS ---\n" + "\n".join(local_retrieved) + f"\n\n--- WEB ---\n{web_context}"
            prompt = build_prompt(full_context, question, history=history)
            
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
            answer = res.choices[0].message.content
            
            # --- OUTPUT ---
            print(f"\n🤖 Bot: {answer}")
            
            if should_speak:
                speak_text(answer) 
            
            history.append((question, answer))
            if len(history) > 5: history.pop(0)
            
        except Exception as e:
            print(f"❌ Processing Error: {e}")

if __name__ == "__main__":
    content = sync_universal_knowledge()
    idx = None
    if content:
        idx = create_faiss_index(embed_texts(content))
    
    start_chat_loop(content, idx)