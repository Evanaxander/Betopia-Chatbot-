import pymupdf  # Modern way to import PyMuPDF (formerly fitz)

# =================================================================
# DOCUMENT PROCESSING: PDF TEXT EXTRACTION
# =================================================================

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts all readable text content.
    
    This is the first step in the RAG pipeline. It converts a static 
    binary PDF file into a raw string that our 'chunker' can then process.
    
    Parameters:
    - pdf_path (str): The system path to the PDF file.
    
    Returns:
    - str: A single string containing the combined text of all pages.
    """
    text = ""
    try:
        # 1. OPEN THE DOCUMENT
        # The 'with' statement ensures the file is closed automatically,
        # preventing memory leaks or file-lock errors.
        with pymupdf.open(pdf_path) as doc:
            
            # 2. PAGE ITERATION
            # PyMuPDF allows us to loop through the document page by page.
            for page in doc:
                # 3. TEXT EXTRACTION
                # page.get_text() extracts text in 'natural' reading order.
                # We add a newline ("\n") to ensure words at the end of 
                # a page don't merge with the first word of the next page.
                text += page.get_text() + "\n"
                
    except Exception as e:
        # Error handling for password-protected or corrupted PDF files.
        print(f"❌ Error reading {pdf_path}: {e}")
        
    return text