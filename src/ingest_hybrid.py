import os
import json
import pickle
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'lontar.json')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
KEYWORD_STORE_PATH = os.path.join(
    BASE_DIR, 'keyword_store', 'bm25_retriever.pkl')


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title", "Naskah Tanpa Judul")
    return metadata


def ingest_data():
    print(f"--- 1. Memulai Ingest Data Lontar ---")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File {DATA_PATH} tidak ditemukan.")
        return

    # 1. Load JSON
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema='.[]',
        content_key="content",
        metadata_func=metadata_func,
        text_content=False
    )
    documents = loader.load()
    print(f"Data dimuat: {len(documents)} dokumen dasar.")

    # 2. Splitting (Chunking)
    # Chunk Size dibuat agak besar (1000) untuk menangkap konteks cerita utuh
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = text_splitter.split_documents(documents)
    print(f"Total Chunks: {len(docs)}")

    # 3. Create Semantic Index (FAISS)
    print("--- 2. Membangun Index Vektor (FAISS) ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(VECTOR_STORE_PATH)
    print("✅ FAISS Index tersimpan.")

    # 4. Create Keyword Index (BM25)
    print("--- 3. Membangun Index Keyword (BM25) ---")
    bm25_retriever = BM25Retriever.from_documents(docs)
    os.makedirs(os.path.dirname(KEYWORD_STORE_PATH), exist_ok=True)
    with open(KEYWORD_STORE_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("✅ BM25 Index tersimpan.")


if __name__ == "__main__":
    ingest_data()
