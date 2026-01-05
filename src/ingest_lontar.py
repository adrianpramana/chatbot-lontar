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

    # Menggunakan himpunan kata kunci yang lebih spesifik untuk disambiguasi
    text_lower = (title + " " + content).lower()

    # Prioritas Kategori (Hierarchy)
    if any(x in text_lower for x in ['usada', 'tamba', 'boreh', 'loloh', 'cetik', 'obat', 'penawar']):
        metadata["category"] = "Pengobatan (Usada)"
        metadata["topic_tags"] = "kesehatan, herbal, penyembuhan"
    elif any(x in text_lower for x in ['wariga', 'dewasa', 'wuku', 'tilem', 'purnama', 'sasih', 'hari baik']):
        metadata["category"] = "Astronomi (Wariga)"
        metadata["topic_tags"] = "waktu, kalender, perbintangan"
    elif any(x in text_lower for x in ['banten', 'yadnya', 'caru', 'upacara', 'sesajen', 'odalan', 'pedudusan']):
        metadata["category"] = "Ritual (Yadnya)"
        metadata["topic_tags"] = "persembahan, upacara, suci"
    elif any(x in text_lower for x in ['tattwa', 'kadyatmikan', 'moksa', 'atma', 'buana agung', 'wrhaspati']):
        metadata["category"] = "Filosofi (Tattwa)"
        metadata["topic_tags"] = "ketuhanan, jiwa, semesta"
    elif any(x in text_lower for x in ['babad', 'silsilah', 'prasasti', 'raja', 'puri', 'dalem']):
        metadata["category"] = "Sejarah (Babad)"
        metadata["topic_tags"] = "asal-usul, leluhur, kerajaan"
    else:
        metadata["category"] = "Umum/Lainnya"
        metadata["topic_tags"] = "general"

    return metadata


def run_ingest():
    print("🚀 [1/3] Memulai Ingest Data Lontar (Mode: Advanced Metadata)...")

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

    # 2. CHUNKING (Optimasi untuk Gemma2:2b)
    # 512 tokens adalah sweet spot untuk model kecil agar tidak 'lupa' konteks awal
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"   --> Dipecah menjadi {len(chunks)} chunks semantik.")

    # 3. INDEXING
    print("🧠 [2/3] Membangun Index Hybrid...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # FAISS (Semantic)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_STORE_PATH)

    # BM25 (Keyword - Penting untuk istilah Bali Kuno yang jarang muncul di training data LLM)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5  # Default retrieval count

    os.makedirs(os.path.dirname(KEYWORD_STORE_PATH), exist_ok=True)
    with open(KEYWORD_STORE_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("🎉 [3/3] Selesai! Database Lontar siap digunakan.")


if __name__ == "__main__":
    run_ingest()
