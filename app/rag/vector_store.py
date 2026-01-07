import faiss
import numpy as np

def create_faiss_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index

import chromadb

def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="betopia_knowledge")

def delete_file_from_db(collection, filename):
    """
    Removes all chunks associated with a specific file 
    to make room for updated information.
    """
    try:
        # We target the 'source' metadata we saved earlier
        collection.delete(where={"source": filename})
        print(f"🗑️  Old entries for {filename} removed from database.")
    except Exception as e:
        print(f"ℹ️  No existing entries found for {filename} (New file).")
