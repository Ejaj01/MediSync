import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


def build_vector_db():
    loader = PyPDFDirectoryLoader("./medical_guidelines/")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    print(f"Total chunks to embed: {len(splits)}. Processing safely in batches...")

    # Initialize the database with the first small batch
    batch_size = 5
    db = Chroma.from_documents(documents=splits[:batch_size], embedding=embeddings, persist_directory="./chroma_db")

    # Process remaining chunks with a safety pause to prevent 429 errors
    for i in range(batch_size, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        db.add_documents(documents=batch)
        print(f"Embedded chunks {i} to {min(i + batch_size, len(splits))} successfully.")
        time.sleep(6)  # 6-second pause gives the API breathing room

    print("Vector storage successfully established.")


if __name__ == "__main__":
    build_vector_db()
