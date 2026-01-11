import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load variables from .env to ensure the key is available
load_dotenv()

# Ensures the DB is created in the project root
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# FIX: Explicitly pass the API key to the embedding function
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables.")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

# Initialize the Chroma client
client = chromadb.PersistentClient(path=DB_PATH)

# Get or create the collection with the explicitly defined embedding function
collection = client.get_or_create_collection(
    name="betopia_knowledge", 
    embedding_function=openai_ef
)

def query_db(query_text, n_results=5):
    """
    Queries the vector database for relevant context chunks.
    """
    try:
        results = collection.query(
            query_texts=[query_text], 
            n_results=n_results
        )
        # Returns the text documents found
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"❌ Database Query Error: {e}")
        return []