from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import PyPDF2

# Load labor_law.pdf
DATA_DIR = Path(os.getenv("labor_law_DATA_DIR", Path(__file__).resolve().parent.parent))
labor_law = DATA_DIR / "labor_law.pdf"

# Read the PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Chunk the text
def chunk_text(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Embed the chunks
def embed_chunks(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Main process
if __name__ == "__main__":
    # Step 1: Read the PDF
    pdf_text = read_pdf(labor_law)

    # Step 2: Chunk the text
    chunks = chunk_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Step 3: Embed the chunks
    embeddings = embed_chunks(chunks, "intfloat/multilingual-e5-large")

    # Example: Print the first embedding
    print("First embedding:", embeddings[0])