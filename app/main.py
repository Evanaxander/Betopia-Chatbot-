import os
from dotenv import load_dotenv
from openai import OpenAI

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import create_faiss_index
from rag.retriever import retrieve_chunks
from rag.prompt import build_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = "data/betopia.pdf"
MAX_MEMORY_TURNS = 5

# Load PDF
text = load_pdf(PDF_PATH)

# Split into chunks
chunks = chunk_text(text)

# Create embeddings
vectors = embed_texts(chunks)

# Build FAISS index
index = create_faiss_index(vectors)

conversation_history = []

def format_history(history):
    formatted = ""
    for turn in history:
        formatted += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    return formatted

def embed_query(q):
    from rag.embeddings import embed_texts
    return embed_texts([q])

print("🤖 Betopia PDF Chatbot is ready! Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved = retrieve_chunks(question, chunks, index, embed_func=embed_query)
    context = "\n\n".join(retrieved)

    # Prepare memory for prompt
    history_tuples = [(turn["user"], turn["assistant"]) for turn in conversation_history]

    # Build prompt
    prompt = build_prompt(context, question, history=history_tuples)

    # Ask GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content
    print("\nBot:", answer)
    print("-" * 50)

    # Save memory
    conversation_history.append({
        "user": question,
        "assistant": answer
    })

    # Keep only last N turns
    if len(conversation_history) > MAX_MEMORY_TURNS:
        conversation_history.pop(0)

    question = input("You: ").strip()
    if not question:
        print("Bot: Please type something 🙂")
        continue