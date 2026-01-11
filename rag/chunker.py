import re

# =================================================================
# TEXT PROCESSING UTILITIES: CHUNKER
# =================================================================

def chunk_text(text, chunk_size=800, chunk_overlap=150):
    """
    Expert Fix: Splits by whitespace/sentences to preserve meaning.
    
    This function breaks down large PDF text into smaller, manageable pieces
    that fit within the LLM's context window.
    
    Parameters:
    - text (str): The raw string extracted from the document.
    - chunk_size (int): Max characters per chunk. 800 is a 'sweet spot' for RAG.
    - chunk_overlap (int): How many characters to repeat from the previous chunk.
                           This ensures context isn't lost at the cutting point.
    """

    # --- 1. TOKENIZATION ---
    # re.split(r'(\s+)', text) splits the text into words and spaces.
    # We keep the spaces so the final joined text looks natural.
    tokens = re.split(r'(\s+)', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    # --- 2. CHUNK ASSEMBLY ---
    for token in tokens:
        current_chunk.append(token)
        current_length += len(token)
        
        # Check if the current collection of tokens has reached the size limit
        if current_length >= chunk_size:
            # Join tokens back into a single string and add to our list
            chunks.append("".join(current_chunk))
            
            # --- 3. OVERLAP CALCULATION ---
            # To prevent losing context, we don't start the next chunk from scratch.
            # We calculate how many tokens represent our 'chunk_overlap' size.
            overlap_count = int(len(current_chunk) * (chunk_overlap / chunk_size))
            
            # Slide the window back: keep the tail end of the current chunk
            # to become the start of the next chunk.
            current_chunk = current_chunk[-max(1, overlap_count):]
            current_length = sum(len(t) for t in current_chunk)
            
    # --- 4. FINAL CLEANUP ---
    # If there is any remaining text that didn't reach 'chunk_size', 
    # add it as the final chunk.
    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return chunks