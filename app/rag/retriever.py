import numpy as np

def retrieve_chunks(query, chunks, index, top_k=3, embed_func=None):
    """
    query: user question
    chunks: all text chunks
    index: FAISS index
    embed_func: function to embed query
    """
    q_vec = embed_func([query])[0].astype('float32')
    D, I = index.search(np.array([q_vec]), top_k)
    results = [chunks[i] for i in I[0]]
    return results
