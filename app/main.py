import os
from dotenv import load_dotenv
from openai import OpenAI

# Custom RAG modules
from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import embed_texts, describe_image  # Added describe_image
from rag.vector_store import create_faiss_index
from rag.retriever import retrieve_chunks
from rag.prompt import build_prompt

# 1. SETUP & INITIALIZATION
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

PDF_PATH = "data/betopia.pdf"
IMAGE_FOLDER = "data/images"
MAX_MEMORY_TURNS = 5

print("🚀 Starting Betopia Knowledge Ingestion...")

# 2. DATA INGESTION (Process Once)

# --- Process PDF ---
print("📄 Loading PDF...")
pdf_text = load_pdf(PDF_PATH)
pdf_chunks = chunk_text(pdf_text)

# --- Process Images ---
image_chunks = []
if os.path.exists(IMAGE_FOLDER):
    print(f"🖼️  Scanning images in {IMAGE_FOLDER}...")
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(IMAGE_FOLDER, filename)
            print(f"   🔎 Analyzing: {filename}...")
            description = describe_image(path)
            if description:
                # We label the source so GPT knows it's describing an image
                image_chunks.append(f"[Source: Image {filename}] {description}")
else:
    print("⚠️  Image folder not found. Skipping image processing.")

# --- Combine and Create Vector Store ---
all_content = pdf_chunks + image_chunks
print(f"🧠 Creating embeddings for {len(all_content)} total chunks...")
vectors = embed_texts(all_content)
index = create_faiss_index(vectors)

print("\n🤖 Betopia Multimodal Chatbot is ready! Type 'exit' to quit.\n")

# 3. CHAT LOOP (Inference)
conversation_history = []

def embed_query(q):
    """
    Ensures the query is passed as a flat list of strings 
    to avoid the 'list has no attribute strip' error.
    """
    # If q is a list, use it; if it's a string, wrap it in a list
    query_list = q if isinstance(q, list) else [q]
    return embed_texts(query_list)

while True:
    # Use a newline to make the chat look cleaner
    question = input("\nYou: ").strip()
    
    if not question:
        continue
        
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break

    print("🔍 Searching PDF and Images...")

    # 1. Retrieve relevant chunks (searches PDF text + Image descriptions)
    # Note: embed_func now calls our fixed embed_query
    retrieved = retrieve_chunks(question, all_content, index, embed_func=embed_query)
    context = "\n\n".join(retrieved)

    # 2. Prepare memory for prompt
    history_tuples = [(turn["user"], turn["assistant"]) for turn in conversation_history]

    # 3. Build prompt using the retrieved context
    prompt = build_prompt(context, question, history=history_tuples)

    # 4. Ask GPT-4o-mini
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content
        print(f"\nBot: {answer}")
        print("-" * 50)

        # 5. Save to Conversation History
        conversation_history.append({
            "user": question,
            "assistant": answer
        })

        # Keep history from getting too long
        if len(conversation_history) > MAX_MEMORY_TURNS:
            conversation_history.pop(0)
            
    except Exception as e:
        print(f"❌ Error during chat: {e}")
    # Retrieve relevant chunks (now searches both PDF text AND Image descriptions)
    retrieved = retrieve_chunks(question, all_content, index, embed_func=embed_query)
    context = "\n\n".join(retrieved)

    # Prepare memory for prompt
    history_tuples = [(turn["user"], turn["assistant"]) for turn in conversation_history]

    # Build prompt
    prompt = build_prompt(context, question, history=history_tuples)

    # Ask GPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content
        print(f"\nBot: {answer}")
        print("-" * 50)

        # Save memory
        conversation_history.append({
            "user": question,
            "assistant": answer
        })

        # Keep memory clean
        if len(conversation_history) > MAX_MEMORY_TURNS:
            conversation_history.pop(0)
            
    except Exception as e:
        print(f"❌ Error during chat: {e}")