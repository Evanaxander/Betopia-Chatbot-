import os
import base64
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_random_exponential

# =================================================================
# 1. INITIALIZATION & API SECURITY
# =================================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Safety check: Prevent the script from running if the key is missing
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

# =================================================================
# 2. VISION OPTIMIZATION (GPT-4o-mini Vision)
# =================================================================

# The @retry decorator handles temporary internet blips or API rate limits.
# Exponential backoff means it waits longer between each subsequent retry.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def describe_image(image_path):
    """
    Converts an image file into a technical text description.
    This allows the 'Chatbot' to search for visual data in the PDF.
    """
    try:
        # Convert binary image data to Base64 string for OpenAI's API
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Request a technical analysis focused on searchable labels and data
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
    """
    Performance Fix: Process multiple images at the same time using Threads.
    max_workers=5 keeps us within typical OpenAI 'Tier 1' rate limits.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        descriptions = list(executor.map(describe_image, image_paths))
    return [d for d in descriptions if d]

# =================================================================
# 3. EMBEDDING OPTIMIZATION (Text to Math)
# =================================================================

def embed_texts(texts):
    """
    Converts a list of text strings into numerical vectors (Arrays).
    Optimized for batching: One API call handles multiple text chunks.
    """
    # Clean input: Ignore empty strings or non-string data
    clean_texts = [str(t).strip() for t in texts if t and str(t).strip()]
    if not clean_texts:
        return []
    
    try:
        # Generate embeddings using the latest, most efficient OpenAI model
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=clean_texts
        )
        # Convert the results into a list of Numpy arrays for the database
        return [np.array(item.embedding) for item in resp.data]
    except Exception as e:
        print(f"❌ Critical Embedding Failure: {e}")
        return []

# =================================================================
# 4. VECTOR DATABASE PERSISTENCE (ChromaDB Sync)
# =================================================================

def sync_to_chroma(collection, chunks, filename):
    """
    The Bridge: Connects processed chunks to the ChromaDB Collection.
    Includes logic to prevent duplicate data from being indexed.
    """
    # --- STEP 1: DUPLICATE CHECK ---
    # We query the DB by the 'source' filename. If it exists, we skip processing
    # to save time and API costs.
    existing = collection.get(where={"source": filename})
    if existing and len(existing['ids']) > 0:
        print(f"⏩ {filename} is already in the database. Moving on.")
        return

    print(f"⚙️  Processing new knowledge: {filename}...")
    
    # --- STEP 2: BATCH EMBEDDING ---
    # Convert all text chunks for this file into vectors in one batch
    vectors = embed_texts(chunks)
    
    if not vectors:
        return

    # --- STEP 3: METADATA PREPARATION ---
    # metadata allows the AI to filter searches (e.g., only look in 'policy.pdf')
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "indexed_at": time.time()} for _ in chunks]
    
    # --- STEP 4: INSERTION ---
    # Add vectors, original text, and metadata to the persistent storage
    collection.add(
        embeddings=[v.tolist() for v in vectors],
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ {filename} successfully indexed with {len(chunks)} chunks.")