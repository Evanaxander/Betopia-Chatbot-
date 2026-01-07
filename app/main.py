import os
from dotenv import load_dotenv
from openai import OpenAI

# Custom RAG modules
from rag.pdf_loader import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, describe_image
from rag.vector_store import create_faiss_index
from rag.retriever import retrieve_chunks
from rag.prompt import build_prompt

# 1. SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = "data"
IMAGE_FOLDER = "data/images"

def run_system():
    print("🚀 Starting Betopia Sequential Ingestion...")
    all_content = []

    # --- Step A: Sequential PDF Processing ---
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.lower().endswith(".pdf"):
                path = os.path.join(DATA_DIR, filename)
                print(f"📄 Processing PDF: {filename}...")
                text = extract_text_from_pdf(path)
                chunks = chunk_text(text)
                # Labeling helps GPT identify the source
                all_content.extend([f"[Source: {filename}] {c}" for c in chunks])

    # --- Step B: Sequential Image Processing ---
    if os.path.exists(IMAGE_FOLDER):
        for filename in os.listdir(IMAGE_FOLDER):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(IMAGE_FOLDER, filename)
                print(f"🖼️  Analyzing Image: {filename}...")
                description = describe_image(path)
                if description:
                    all_content.append(f"[Source: Image {filename}] {description}")

    if not all_content:
        print("❌ No data found in /data. Please add PDFs or Images.")
        return

    # --- Step C: Create Brain ---
    print(f"🧠 Indexing {len(all_content)} chunks into Vector Store...")
    vectors = embed_texts(all_content)
    index = create_faiss_index(vectors)
    
    start_chat_loop(all_content, index)

def start_chat_loop(all_content, index):
    print("\n🤖 Betopia Terminal Chatbot Ready! (Type 'exit' to quit)\n")
    history = []
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "exit": break
        if not question: continue

        # Retrieval
        # We pass a lambda for embed_func to ensure the query is handled as a list
        retrieved = retrieve_chunks(
            question, 
            all_content, 
            index, 
            embed_func=lambda q: embed_texts([q] if isinstance(q, str) else q)
        )
        context = "\n\n".join(retrieved)

        # Build Prompt & Chat
        prompt = build_prompt(context, question, history=[(h["user"], h["assistant"]) for h in history])
        
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = res.choices[0].message.content
            print(f"\nBot: {answer}")
            history.append({"user": question, "assistant": answer})
            if len(history) > 5: history.pop(0)
        except Exception as e:
            print(f"❌ Chat Error: {e}")

if __name__ == "__main__":
    run_system()