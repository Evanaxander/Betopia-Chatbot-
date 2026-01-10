import os
import chromadb
from chromadb.utils import embedding_functions

# Ensures the DB is created in the project root
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="betopia_knowledge", 
    embedding_function=openai_ef
)

def query_db(query_text, n_results=5):
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        return results['documents'][0] if results['documents'] else []
    except:
        return []