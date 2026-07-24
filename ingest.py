import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GUIDELINES_DIR = "./medical_guidelines/"
CHROMA_DIR = "./chroma_db"

def load_all_pdf_documents(directory_path):
    print(f"Scanning '{directory_path}' for all PDF guidelines...")
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} PDF pages/documents.")

    # Tag each chunk with its specialty subfolder name
    for doc in docs:
        file_path = doc.metadata.get("source", "")
        relative_path = os.path.relpath(file_path, directory_path)
        folder_parts = relative_path.split(os.sep)
        specialty = folder_parts[0] if len(folder_parts) > 1 else "General"
        doc.metadata["specialty"] = specialty

    return docs


def build_vector_db():
    docs = load_all_pdf_documents(GUIDELINES_DIR)

    if not docs:
        print(f"No PDF files found in '{GUIDELINES_DIR}'.")
        return

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    total_splits = len(splits)
    print(f"Generated {total_splits} total chunks across all specialties.")

    print("Initializing local HuggingFace Embedding Model ('all-MiniLM-L6-v2')...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating Chroma VectorDB and embedding chunks locally...")
    start_time = time.time()

    # Process documents into Chroma vector storage
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    elapsed_time = round(time.time() - start_time, 2)
    print(f"✅ Successfully embedded all {total_splits} chunks into Chroma storage in {elapsed_time}s!")


if __name__ == "__main__":
    build_vector_db()