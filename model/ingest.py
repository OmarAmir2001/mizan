from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # pymupdf
import chromadb

# Constants
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Load labor_law.pdf
labor_law = r'docs\labor_law.pdf'



def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
# Chunk the text


def chunk_text(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# Embed the chunks
def embed_chunks(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Store in ChromaDB
def store_in_chromadb(chunks, embeddings, collection_name="mizan_laws"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Stored {collection.count()} chunks in ChromaDB")
    return collection

# Main process
if __name__ == "__main__":
    # Step 1: Read the PDF
    pdf_text = read_pdf(labor_law)

    # Step 2: Chunk the text
    chunks = chunk_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Step 3: Embed the chunks
    embeddings = embed_chunks(chunks, "intfloat/multilingual-e5-large")
    
    # Step 4: Store in ChromaDB
    collection = store_in_chromadb(chunks, embeddings)
    print(f"Total chunks stored: {collection.count()}")