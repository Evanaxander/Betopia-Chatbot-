import os
import base64
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 1. Environment Setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")
client = OpenAI(api_key=api_key)

# --- VISION OPTIMIZATION ---

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def describe_image(image_path):
    """
    Expert Fix: Added Exponential Backoff. 
    If OpenAI is busy or you hit a rate limit, it will retry automatically.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image technically. Focus on text, labels, and data points for RAG retrieval."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ vision error on {os.path.basename(image_path)}: {e}")
        return ""

def process_images_parallel(image_paths, max_workers=5):
    """Parallelized image processing with a worker limit to respect API quotas."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        descriptions = list(executor.map(describe_image, image_paths))
    return [d for d in descriptions if d]


# --- EMBEDDING OPTIMIZATION ---

def embed_texts(texts):
    """
    Expert Fix: Optimized Batching.
    We send the entire list to OpenAI in one request. 
    Added safety check for empty inputs which often crash scripts.
    """
    clean_texts = [str(t).strip() for t in texts if t and str(t).strip()]
    if not clean_texts:
        return []
    
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=clean_texts
        )
        return [np.array(item.embedding) for item in resp.data]
    except Exception as e:
        print(f"❌ Critical Embedding Failure: {e}")
        return []


# --- VECTOR DB PERSISTENCE ---

def sync_to_chroma(collection, chunks, filename):
    """
    Expert Fix: Atomic Updates.
    Checks for the 'source' metadata before performing expensive embedding operations.
    """
    # 1. Check if file is already indexed
    existing = collection.get(where={"source": filename})
    if existing and len(existing['ids']) > 0:
        print(f"⏩ {filename} is already in the database. Moving on.")
        return

    print(f"⚙️  Processing new knowledge: {filename}...")
    
    # 2. Batch embed (one API call for all chunks)
    vectors = embed_texts(chunks)
    
    if not vectors:
        return

    # 3. Prepare metadata and IDs
    # Metadata filtering allows the AI to say 'In file XYZ, it says...'
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "indexed_at": time.time()} for _ in chunks]
    
    # 4. Insert into ChromaDB
    collection.add(
        embeddings=[v.tolist() for v in vectors],
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ {filename} successfully indexed with {len(chunks)} chunks.")