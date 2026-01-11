import pymupdf  # Modern way to import PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text sequentially from each page of a PDF."""
    text = ""
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text