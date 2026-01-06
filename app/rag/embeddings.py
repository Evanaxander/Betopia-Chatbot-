import os
import base64
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables first
load_dotenv()

# 2. Get API Key and validate
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env file at the project root.")

# 3. Initialize ONE client for both embeddings and image descriptions
client = OpenAI(api_key=api_key)

def describe_image(image_path):
    """
    Uses GPT-4o-mini Vision to turn an image into a searchable text description.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail for a technical database. Focus on facts, numbers, and key labels. This description will be used for RAG retrieval."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error describing image {image_path}: {e}")
        return ""

def embed_texts(texts):
    """
    Converts a list of strings into vectors.
    """
    embeddings = []
    
    # Ensure 'texts' is a list of strings, not a list of lists
    for t in texts:
        # If 't' is accidentally a list (e.g., [['hello']]), take the first element
        if isinstance(t, list):
            t = t[0]
            
        if not t or not str(t).strip():
            continue
            
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=str(t) # Ensure it's a string
            )
            embeddings.append(np.array(resp.data[0].embedding))
        except Exception as e:
            print(f"❌ Error embedding text: {str(t)[:50]}... | {e}")
            
    return embeddings