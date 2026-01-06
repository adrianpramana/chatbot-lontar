# Melakukan import library yang diperlukan
import os
import time
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- ELASTICSEARCH CLIENT ---
from elasticsearch import Elasticsearch, helpers

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'lontar.json')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')

# KONFIGURASI ELASTICSEARCH (Sesuaikan dengan setup lokal Anda)
# Jika default tanpa password (mode dev):
ES_CLIENT = Elasticsearch("http://localhost:9200")

# NAMA INDEX ELASTICSEARCH
ES_INDEX_NAME = "lontar_knowledge_base"

# Fungsi untuk mengekstrak metadata dari setiap record


def metadata_func(record: dict, metadata: dict) -> dict:
    title = record.get("title", "Tanpa Judul")
    content = record.get("content", "")
    metadata["title"] = title
    text_lower = (title + " " + content).lower()

    # Logika Kategori (Tetap dipertahankan)
    if any(x in text_lower for x in ['usada', 'tamba', 'boreh', 'obat']):
        metadata["category"] = "Pengobatan (Usada)"
        metadata["topic_tags"] = "kesehatan, herbal"
    elif any(x in text_lower for x in ['wariga', 'dewasa', 'purnama']):
        metadata["category"] = "Astronomi (Wariga)"
        metadata["topic_tags"] = "waktu, kalender"
    elif any(x in text_lower for x in ['banten', 'yadnya', 'caru']):
        metadata["category"] = "Ritual (Yadnya)"
        metadata["topic_tags"] = "upacara, suci"
    else:
        metadata["category"] = "Umum/Lainnya"
        metadata["topic_tags"] = "general"
    return metadata


# Fungsi untuk membuat index Elasticsearch dengan mapping khusus
def setup_elastic_index():
    """
    Mendefinisikan Mapping dengan ANALYZER KHUSUS (Synonym Awareness).
    Fitur Canggih: Menambahkan pemahaman bahasa (Ontologi Sederhana).
    """
    if ES_CLIENT.indices.exists(index=ES_INDEX_NAME):
        print(f"⚠️ Menghapus index lama '{ES_INDEX_NAME}'...")
        ES_CLIENT.indices.delete(index=ES_INDEX_NAME)

    # Konfigurasi Analyzer dengan Sinonim Bali-Indonesia
    settings = {
        "settings": {
            "analysis": {
                "filter": {
                    "lontar_synonym": {
                        "type": "synonym",
                        "synonyms": [
                            # Sinonim Domain Lontar (Contoh)
                            "usada, tamba, obat, penyembuhan, herbal",
                            "wariga, dewasa, kalender, hari baik, padewasan",
                            "yadnya, banten, upacara, persembahan, caru",
                            "tattwa, filosofi, hakikat, ketuhanan",
                            "niskala, gaib, tak kasat mata",
                            "sakit, gerah, lara, roga"
                        ]
                    }
                },
                "analyzer": {
                    "balinese_analyzer": {
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "lontar_synonym"  # Filter sinonim diaktifkan
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "balinese_analyzer"  # Menggunakan analyzer pintar
                },
                "content": {
                    "type": "text",
                    "analyzer": "balinese_analyzer"
                },
                "category": {"type": "keyword"},
                "tags": {"type": "text"},
                "chunk_id": {"type": "keyword"}
            }
        }
    }

    # Membuat index dengan setting di atas dan mapping khusus
    ES_CLIENT.indices.create(index=ES_INDEX_NAME, body=settings)
    print(
        f"✅ Index '{ES_INDEX_NAME}' berhasil dibuat dengan Mapping (Standard Elastic Search).")


# Fungsi utama untuk menjalankan proses ingest data dan indexing
def run_ingest():
    print("🚀 [1/3] Memulai Ingest Data Lontar (Mode: Real Elasticsearch)...")

    # Cek Koneksi Elastic
    if not ES_CLIENT.ping():
        print("❌ GAGAL KONEKSI: Pastikan elasticsearch.bat sudah berjalan di background!")
        return

    # 1. LOAD DATA
    loader = JSONLoader(DATA_PATH, jq_schema='.[]', content_key="content",
                        metadata_func=metadata_func, text_content=False)
    docs = loader.load()

    # 2. CHUNKING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128)
    chunks = splitter.split_documents(docs)
    print(f"   --> Dipecah menjadi {len(chunks)} chunks.")

    # 3. VECTOR INDEXING (FAISS)
    print("🧠 [2/3] Membangun Semantic Layer (FAISS)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_STORE_PATH)

    # 4. ELASTICSEARCH INGESTION
    print("🔍 [3/3] Uploading ke Elasticsearch Local...")
    setup_elastic_index()

    # Persiapan Data untuk Bulk Insert ke Elasticsearch
    actions = []
    for i, chunk in enumerate(chunks):
        doc_body = {
            "_index": ES_INDEX_NAME,
            "_id": str(i),  # ID dokumen
            "_source": {
                "title": chunk.metadata.get("title"),
                "content": chunk.page_content,
                "category": chunk.metadata.get("category"),
                "tags": chunk.metadata.get("topic_tags"),
                "chunk_id": str(i)
            }
        }
        actions.append(doc_body)

    # Bulk Insert ke Elasticsearch dan cek hasilnya
    helpers.bulk(ES_CLIENT, actions)
    print(
        f"   --> Berhasil mengupload {len(actions)} dokumen ke Elasticsearch.")

    # Refresh index agar data langsung searchable
    ES_CLIENT.indices.refresh(index=ES_INDEX_NAME)
    print("🎉 Selesai! Database Hybrid (FAISS + Real Elasticsearch) siap.")


# Menjalankan fungsi utama jika file ini dieksekusi langsung
if __name__ == "__main__":
    run_ingest()
