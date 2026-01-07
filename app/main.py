import os
from dotenv import load_dotenv
from openai import OpenAI

# Custom RAG modules 
from rag.pdf_loader import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, describe_image
from rag.vector_store import create_faiss_index, get_chroma_collection, delete_file_from_db
from rag.retriever import retrieve_chunks
from rag.prompt import build_prompt

# 1. SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = "data"
IMAGE_FOLDER = "data/images"

def sync_and_get_content():
    """Syncs PDF and Image data with ChromaDB and returns consolidated text for FAISS."""
    print("🔄 Syncing knowledge base with latest files...")
    collection = get_chroma_collection()
    all_content = []

    # --- Step A: Dynamic PDF Processing ---
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.lower().endswith(".pdf"):
                path = os.path.join(DATA_DIR, filename)
                
                # DYNAMIC UPDATE: Delete old version to prevent confusion
                delete_file_from_db(collection, filename)
                
                print(f"📄 Indexing PDF: {filename}...")
                text = extract_text_from_pdf(path)
                chunks = chunk_text(text)
                
                # Labeling for LLM context
                labeled_chunks = [f"[Source: {filename}] {c}" for c in chunks]
                all_content.extend(labeled_chunks)
                
                # Persistent storage in ChromaDB
                collection.add(
                    documents=chunks,
                    metadatas=[{"source": filename} for _ in chunks],
                    ids=[f"{filename}_{i}" for i in range(len(chunks))]
                )

    # --- Step B: Sequential Image Processing ---
    if os.path.exists(IMAGE_FOLDER):
        for filename in os.listdir(IMAGE_FOLDER):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(IMAGE_FOLDER, filename)
                
                # DYNAMIC UPDATE: Delete old image description
                delete_file_from_db(collection, filename)
                
                print(f"🖼️ Analyzing Image: {filename}...")
                description = describe_image(path)
                if description:
                    image_text = f"[Source: Image {filename}] {description}"
                    all_content.append(image_text)
                    
                    # Add image description to Chroma
                    collection.add(
                        documents=[description],
                        metadatas=[{"source": filename}],
                        ids=[f"img_{filename}"]
                    )

    return all_content

def generate_standalone_query(question, history):
    """Rephrases follow-up questions using chat history to improve search results."""
    # Use only last 3 turns for context to keep it snappy
    history_context = "\n".join([f"User: {u}\nAI: {a}" for u, a in history[-3:]])
    msg = (f"Given the following chat history:\n{history_context}\n\n"
           f"Rephrase the follow-up question into a standalone search query. "
           f"If it's already standalone, return it as is.\nQuestion: {question}")
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": msg}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return question # Fallback to original question on error

def start_chat_loop(all_content, index):
    print("\n🤖 Knowledge Assistant Ready! (Type 'exit' to quit)\n")
    history = [] 
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "exit": break
        if not question: continue

        # 1. Contextual Query Rephrasing
        search_query = question
        if len(history) > 0:
            search_query = generate_standalone_query(question, history)
            print(f"🔍 Searching for: {search_query}...")
        else:
            print("🔍 Searching across all documents...")

        # 2. Retrieval
        retrieved = retrieve_chunks(
            search_query, 
            all_content, 
            index, 
            embed_func=lambda q: embed_texts([q] if isinstance(q, str) else q)
        )
        context = "\n\n".join(retrieved)

        # 3. Prompt Construction
        prompt = build_prompt(context, question, history=history)
        
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = res.choices[0].message.content
            print(f"\nBot: {answer}")
            
            # 4. History Management (Stores last 7 turns)
            history.append((question, answer))
            if len(history) > 7: 
                history.pop(0)
                
        except Exception as e:
            print(f"❌ Chat Error: {e}")

def run_system():
    # 1. Sync files and clean up ChromaDB
    all_content = sync_and_get_content()

    if not all_content:
        print("❌ No data found. Please add PDFs or Images.")
        return

    # 2. Create the Search Index (FAISS)
    print(f"🧠 Indexing {len(all_content)} total chunks into memory...")
    vectors = embed_texts(all_content)
    index = create_faiss_index(vectors)
    
    # 3. Start Chat
    start_chat_loop(all_content, index)

if __name__ == "__main__":
    run_system()