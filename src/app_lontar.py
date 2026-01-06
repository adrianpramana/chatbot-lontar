# Melakukan import library yang dibutuhkan
import os
import gradio as gr
from dotenv import load_dotenv

# --- RAG LIBRARIES ---
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- ELASTICSEARCH CLIENT ---
from elasticsearch import Elasticsearch

# Memuat konfigurasi dari .env dan path dasar
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')

# CONFIG ELASTICSEARCH (Harus sama dengan Ingest)
ES_CLIENT = Elasticsearch("http://localhost:9200")
# Note: Tambahkan auth/password disini jika setup ES Anda menggunakan security.

# NAMA INDEX ELASTICSEARCH
ES_INDEX_NAME = "lontar_knowledge_base"
MODEL_NAME = "gemma2:2b"

print(f"🚀 Memulai Sistem Lontar AI (Model: {MODEL_NAME})...")

# Embedding Model & Cek Kesiapan Database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_db = None
ES_READY = False

# Cek FAISS Vector Store
if os.path.exists(VECTOR_STORE_PATH):
    try:
        vector_db = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Error FAISS: {e}")

# Cek Elasticsearch
try:
    if ES_CLIENT.ping():
        ES_READY = True
        print("✅ Terkoneksi ke Elasticsearch Local.")
    else:
        print("⚠️ Elasticsearch Service tidak terdeteksi (Pastikan elasticsearch.bat running).")
except Exception as e:
    print(f"❌ Error ES Connection: {e}")

# Setup LLM
llm = ChatOllama(model=MODEL_NAME, temperature=0.2, keep_alive="1h")

# Mendefinisikan fungsi untuk pemahaman query


def query_understanding(query):
    # (Logika sama: membersihkan query user)
    return query

# Fungsi untuk pencarian di Elasticsearch secara aktual untuk lexical search


def search_elastic_real(query_str, k=3):
    """
    Menggunakan Elasticsearch Query DSL (JSON).
    Ini adalah metode pencarian standar industri.
    """
    if not ES_READY:
        return []

    # CONSTRUKSI QUERY DSL
    # Mencari di title (boosted) dan content.
    # Fuzziness='AUTO' menangani typo secara otomatis.
    body_query = {
        "size": k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_str,
                            # ^2 artinya bobot judul dikali 2
                            "fields": ["title^2", "content", "tags"],
                            "fuzziness": "AUTO",
                            "operator": "or"
                        }
                    }
                ]
            }
        }
    }

    # Eksekusi Query ke Elasticsearch
    try:
        response = ES_CLIENT.search(index=ES_INDEX_NAME, body=body_query)
        hits = response['hits']['hits']

        results_docs = []
        for hit in hits:
            source = hit['_source']
            score = hit['_score']

            # Membuat Document dari hasil pencarian
            doc = Document(
                page_content=source.get('content', ''),
                metadata={
                    "title": source.get('title'),
                    "category": source.get('category'),
                    "topic_tags": source.get('tags'),
                    "source": "Elasticsearch Service",
                    "score": score
                }
            )
            results_docs.append(doc)

        return results_docs

    except Exception as e:
        print(f"Error Query ES: {e}")
        return []

# Fungsi untuk retrieval hybrid (Elastic Search + FAISS)


def retrieve_hybrid(query, k_semantic=4, k_lexical=3):
    if vector_db is None:
        return []

    docs_lexical = []
    docs_semantic = []

    # 1. Lexical Search
    docs_lexical = search_elastic_real(query, k=k_lexical)

    # 2. Semantic Search (FAISS)
    try:
        docs_semantic = vector_db.similarity_search(query, k=k_semantic)
    except:
        pass

    # 3. Fusion (Gabungan)
    final_docs = []
    seen = set()

    # Gabungkan: Prioritas ES (Lexical) lalu FAISS (Semantic)
    all_candidates = docs_lexical + docs_semantic

    for d in all_candidates:
        signature = d.page_content[:100]
        if signature not in seen:
            if d in docs_lexical:
                d.metadata['method'] = f"Elastic Match (Score: {d.metadata.get('score', 0):.2f})"
            else:
                d.metadata['method'] = "Semantic Match"
            final_docs.append(d)
            seen.add(signature)

    return final_docs[:k_semantic+1]


# fungsi pipeline yang menggabungkan semua langkah dan menghasilkan respons
def chat_pipeline(user_input, history):
    if not user_input:
        yield history, ""
        return

    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": ""})

    log_buffer = f"🔍 INPUT USER: {user_input}\n" + "-"*40 + "\n"
    yield history, log_buffer

    if not ES_READY and vector_db is None:
        history[-1]['content'] = "⚠️ Sistem DB Error. Cek koneksi Elasticsearch."
        yield history, log_buffer + "❌ CONNECTION ERROR"
        return

    log_buffer += "🧠 PROSES 1: QUERY UNDERSTANDING...\n"
    yield history, log_buffer
    expanded_query = query_understanding(user_input)
    log_buffer += f"   ↳ Query: '{expanded_query}'\n"
    yield history, log_buffer

    log_buffer += "\n📚 PROSES 2: HYBRID RETRIEVAL (ES + FAISS)...\n"
    yield history, log_buffer
    docs = retrieve_hybrid(expanded_query)

    if not docs:
        history[-1]['content'] = "Maaf, referensi tidak ditemukan."
        yield history, log_buffer + "❌ NIHIL"
        return

    context_str = ""
    log_buffer += f"✅ DITEMUKAN {len(docs)} REFERENSI:\n"
    for i, d in enumerate(docs):
        judul = d.metadata.get('title', '-')
        metode = d.metadata.get('method', 'Unknown')
        log_buffer += f"   • {i+1}. {judul} [{metode}]\n"
        context_str += f"SUMBER: {judul}\nISI: {d.page_content}\n\n"

    yield history, log_buffer

    log_buffer += "\n✍️ PROSES 3: GENERATING RESPONSE...\n"
    yield history, log_buffer

   # Menyusun system prompt dengan konteks dan instruksi penulisan yang ketat supaya jawaban sesuai gaya akademis dan filologis serta sesuai konteks lontar
    system_prompt = f"""PERAN:
Anda adalah 'Asisten Filologi Digital', pakar Lontar Bali yang menulis dengan gaya akademis murni layaknya penyusun jurnal ilmiah atau paper. Jawaban Anda harus analitis, kaya diksi, dan menghindari pola kalimat yang monoton.

TUGAS:
Sintesiskan data dari [KONTEKS] menjadi narasi ilmiah yang mengalir. Hindari format kaku "Item: Penjelasan". Sebaliknya, biarkan setiap poin menceritakan fungsi unik naskah tersebut.

ATURAN PENULISAN (STRICT ACADEMIC GUIDELINES):

1.  **Struktur Narasi Ilmiah (Flow)**:
    * **Pembuka:** Mulailah dengan paragraf sintesis yang menguraikan tema besar pertanyaan (misal: hakikat Puja dalam teologi Hindu Bali) sebelum masuk ke rincian naskah.
    * **Isi (Analytical List):**
        * Gunakan **bullet points** untuk merinci naskah/konsep.
        * **FORMAT:** Gunakan gaya **"**Judul Lontar/Konsep**: [Analisis langsung tentang isi, mantra, atau fungsi spesifik]."**
        * **LARANGAN KERAS (ANTI-REPETISI):** DILARANG memulai setiap poin dengan frasa yang sama berulang-ulang (seperti: *"Lontar ini berisi...", "Lontar ini menjelaskan...", "Penjelasan:..."*).
        * **VARIASI DIKSI:** Gunakan variasi kata kerja (misal: *"menguraikan", "menjabarkan", "menitikberatkan", "menjadi pedoman", "mengandung rapalan"*).
    * **Penutup:** Simpulkan dengan benang merah filosofis (Tri Hita Karana, Desa Kala Patra, dll).

2.  **Kedalaman Filologis**:
    * Jelaskan istilah **Kawi, Sanskerta, atau Bahasa Bali** dengan padanan maknanya dalam kurung. Contoh: *ngastawa (memuja/memuji)*.
    * Jangan hanya menyebut "untuk upacara", tapi sebutkan jenis upacaranya jika ada di data (misal: *Dewa Yadnya, Pitra Yadnya*).

3.  **Integritas & Anti-Halusinasi**:
    * Hanya gunakan fakta yang tersedia di [KONTEKS]. Jika konteks hanya memberikan judul tanpa deskripsi, sebutkan hanya judulnya atau abaikan jika tidak relevan.

4.  **TAWARAN EKSPLORASI (Call-to-Action)**:
    * Berisi kesimpulan jawaban dan pelajaran penting dari lontar.
    * Di bagian paling bawah, berikan jeda satu baris.
    * Tulis kalimat penutup yang elegan dan akademis dengan huruf miring (*italic*).
    * Contoh: *"Apakah terdapat perspektif lain dari jawaban ini yang ingin Anda diskusikan, atau Anda memerlukan penelusuran spesifik pada teks lontar tertentu?"*

DATA REFERENSI (CONTEXT):
{context_str}

PERTANYAAN PENGGUNA: 
{user_input}

JAWABAN ANALISIS ILMIAH:"""

    response_text = ""
    try:
        for chunk in llm.stream(system_prompt):
            token = chunk.content
            response_text += token
            history[-1]['content'] = response_text
            yield history, log_buffer
    except Exception as e:
        history[-1]['content'] = f"Error: {str(e)}"
        yield history, log_buffer

    log_buffer += "\n✨ SELESAI."
    yield history, log_buffer


# Membuat antarmuka Gradio dan menjalankannya
custom_css = """
#chatbot { height: 600px !important; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
#log_panel { background-color: #f8fafc; font-family: monospace; font-size: 0.85em; }
"""
theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate").set(
    body_background_fill="#ffffff")

with gr.Blocks(title="Lontar Chatbot | Made Adrian") as demo:
    gr.Markdown(
        "# 🌿 Lontar Chatbot Powered by Gemma2, FAISS, Elasticsearch | Created by I Made Adrian Astalina Pramana | S2 TI UNUD")
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Log Trace")
            log_view = gr.Textbox(
                label="Process Log", lines=24, interactive=False, elem_id="log_panel")
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Dialog Lontar", elem_id="chatbot", avatar_images=(
                None, "https://cdn-icons-png.flaticon.com/512/4712/4712009.png"))
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False, placeholder="Tanya tentang lontar...", scale=5, container=False, autofocus=True)
                btn_submit = gr.Button("Kirim", variant="primary", scale=1)

    txt_input.submit(chat_pipeline, [txt_input, chatbot], [
                     chatbot, log_view]).then(lambda: "", outputs=[txt_input])
    btn_submit.click(chat_pipeline, [txt_input, chatbot], [
                     chatbot, log_view]).then(lambda: "", outputs=[txt_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860,
                inbrowser=True, theme=theme, css=custom_css)
