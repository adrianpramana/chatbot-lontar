import os
import pickle
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# --- PATH CONFIG (ABSOLUTE) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'lontar.json')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
KEYWORD_STORE_PATH = os.path.join(
    BASE_DIR, 'keyword_store', 'bm25_retriever.pkl')


def metadata_func(record: dict, metadata: dict) -> dict:
    title = record.get("title", "Tanpa Judul")
    content = record.get("content", "")
    metadata["title"] = title

    # KATEGORISASI OTOMATIS (S2 FEATURE)
    text = (title + " " + content).lower()
    if any(x in text for x in ['filosofi', 'tattwa', 'hakikat', 'siwa', 'buda']):
        metadata["category"] = "Filosofi"
    elif any(x in text for x in ['upacara', 'yadnya', 'banten', 'sesajen']):
        metadata["category"] = "Ritual"
    elif any(x in text for x in ['usada', 'obat', 'tamba', 'sakit']):
        metadata["category"] = "Pengobatan"
    elif any(x in text for x in ['cerita', 'prabu', 'raja', 'babad']):
        metadata["category"] = "Sastra"
    else:
        metadata["category"] = "Umum"
    return metadata


def run_ingest():
    print("🚀 [1/3] Memulai Ingest Data Lontar...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return

    # 1. LOAD DATA
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema='.[]',
        content_key="content",
        metadata_func=metadata_func,
        text_content=False
    )
    docs = loader.load()
    print(f"   --> Berhasil memuat {len(docs)} dokumen.")

    # 2. CHUNKING (600 chars)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"   --> Dipecah menjadi {len(chunks)} chunks.")

    # 3. INDEXING (Multilingual Embedding)
    print("🧠 [2/3] Membangun Index Hybrid (Vector + Keyword)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # FAISS (Semantic)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_STORE_PATH)

    # BM25 (Keyword)
    bm25 = BM25Retriever.from_documents(chunks)
    os.makedirs(os.path.dirname(KEYWORD_STORE_PATH), exist_ok=True)
    with open(KEYWORD_STORE_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("🎉 [3/3] Selesai! Database siap.")


if __name__ == "__main__":
    run_ingest()
