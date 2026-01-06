# app/rag/embeddings.py
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")

client = OpenAI(api_key=api_key)

def embed_texts(texts):
    """
    Generate embeddings for a list of texts using OpenAI embeddings API.

    Args:
        texts (list[str]): List of text strings to embed.

    Returns:
        list[np.ndarray]: List of embeddings as numpy arrays.
    """
    embeddings = []
    for t in texts:
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=t
            )
            # Convert embedding to numpy array
            embeddings.append(np.array(resp.data[0].embedding))
        except Exception as e:
            print(f"Error embedding text: {t[:50]}... | {e}")
    return embeddings
