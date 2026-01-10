import re

def chunk_text(text, chunk_size=800, chunk_overlap=150):
    """Expert Fix: Splits by whitespace/sentences to preserve meaning."""
    # Split by any whitespace but keep the whitespace
    tokens = re.split(r'(\s+)', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_length += len(token)
        
        if current_length >= chunk_size:
            chunks.append("".join(current_chunk))
            # Create overlap by keeping the last portion of tokens
            overlap_count = int(len(current_chunk) * (chunk_overlap / chunk_size))
            current_chunk = current_chunk[-max(1, overlap_count):]
            current_length = sum(len(t) for t in current_chunk)
            
    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks