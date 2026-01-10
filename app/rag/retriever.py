import numpy as np

def retrieve_chunks(query, all_content, index, embed_func, k=5):
    # Ensure embed_func is actually callable
    if embed_func is None or not callable(embed_func):
        raise ValueError("The embedding function provided to retrieve_chunks is not valid.")
        
    # Generate the vector for the search query
    query_vector = embed_func([query])[0].astype('float32')
    
    # Search FAISS
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    
    # Return the corresponding text chunks
    return [all_content[i] for i in indices[0] if i < len(all_content)]