import os
import pickle
import gradio as gr
from dotenv import load_dotenv

# --- RAG LIBRARIES ---
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. SETUP CONFIG
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
KEYWORD_STORE_PATH = os.path.join(
    BASE_DIR, 'keyword_store', 'bm25_retriever.pkl')

# Model (Sesuai request: Gemma2:2b untuk performa ringan di i5 Gen 11 Ram 8GB)
MODEL_NAME = "gemma2:2b"

print(f"🚀 Memulai Sistem Lontar AI (Model: {MODEL_NAME})...")

# 2. LOAD RESOURCES
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_db = None
bm25_retriever = None
DB_READY = False

if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(KEYWORD_STORE_PATH):
    try:
        vector_db = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        with open(KEYWORD_STORE_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        DB_READY = True
        print("✅ Database Siap.")
    except Exception as e:
        print(f"❌ Error DB: {e}")
else:
    print("⚠️ Database belum ada. Jalankan 'ingest_lontar.py'.")

# Setup LLM - Temperature rendah untuk akurasi data
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    keep_alive="1h"
)

# --- 3. LOGIKA RAG ---

# Fungsi Query Understanding dengan konsepnya menjelaskan alur pemahaman query LLM


def query_understanding(query):
    """
    Menggantikan query_expansion biasa.
    Melakukan analisis 'Intent' dan 'Entity' untuk pemahaman mendalam.
    """
    try:
        # Prompt Engineering: Few-Shot Learning untuk mengarahkan model
        prompt = f"""Analisis pertanyaan pengguna tentang Lontar Bali.
        Pertanyaan: "{query}"
        
        Tugas:
        1. Ekstrak topik inti (misal: "Caru Ayam" -> "Upacara/Caru").
        2. Perbaiki istilah Bali jika ada typo.
        3. Berikan output HANYA string kata kunci pencarian yang optimal.
        
        Output:"""

        response = llm.invoke(prompt)
        clean_query = response.content.strip().replace('"', '').replace("Output:", "")
        return clean_query
    except:
        return query


def retrieve_hybrid(query, k_semantic=4, k_lexical=3):
    if not DB_READY:
        return []

    docs_bm25 = []
    docs_faiss = []

    # 1. Lexical Search (BM25) - Menangkap istilah spesifik (misal: "manca wali krama")
    try:
        bm25_retriever.k = k_lexical
        docs_bm25 = bm25_retriever.invoke(query)
    except:
        pass

    # 2. Semantic Search (FAISS) - Menangkap makna (misal: "cara mengobati sakit kepala")
    try:
        docs_faiss = vector_db.similarity_search(query, k=k_semantic)
    except:
        pass

    # 3. Fusion & Deduplication (Reranking sederhana)
    final_docs = []
    seen = set()

    # Prioritaskan BM25 untuk istilah langka, FAISS untuk konsep umum
    all_candidates = docs_bm25 + docs_faiss

    for d in all_candidates:
        # Gunakan snippet konten sebagai signature unik
        signature = d.page_content[:100]
        if signature not in seen:
            if d in docs_bm25:
                d.metadata['method'] = "Lexical Match"
            else:
                d.metadata['method'] = "Semantic Match"

            final_docs.append(d)
            seen.add(signature)

    # Batasi output context agar tidak overload RAM
    return final_docs[:k_semantic+1]

# --- 4. PIPELINE CHAT (HULU KE HILIR LOGGING) ---


def chat_pipeline(user_input, history):
    """
    Pipeline RAG End-to-End dengan Logging Aktivitas Spesifik
    """
    if not user_input:
        yield history, ""
        return

    if history is None:
        history = []

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": ""})

    # [LOG 1: Input]
    log_buffer = f"🔍 INPUT USER: {user_input}\n" + "-"*40 + "\n"
    yield history, log_buffer

    if not DB_READY:
        history[-1]['content'] = "⚠️ Database belum siap. Mohon jalankan ingest data."
        yield history, log_buffer + "❌ DB ERROR"
        return

    # [LOG 2: Query Understanding]
    log_buffer += "🧠 PROSES 1: QUERY UNDERSTANDING...\n"
    yield history, log_buffer

    expanded_query = query_understanding(user_input)

    log_buffer += f"   ↳ Intent Terdeteksi: '{expanded_query}'\n"
    log_buffer += f"   ↳ Strategi: Hybrid Search (Semantic + Lexical)\n"
    yield history, log_buffer

    # [LOG 3: Retrieval]
    log_buffer += "\n📚 PROSES 2: RETRIEVAL DATA...\n"
    yield history, log_buffer

    docs = retrieve_hybrid(expanded_query)

    if not docs:
        history[-1]['content'] = "Maaf, tidak ditemukan referensi lontar yang relevan dengan pertanyaan Anda."
        yield history, log_buffer + "❌ NIHIL: Ambang batas relevansi tidak terpenuhi."
        return

    # [LOG 4: Context Assembly]
    context_str = ""
    log_buffer += f"✅ DITEMUKAN {len(docs)} REFERENSI:\n"

    for i, d in enumerate(docs):
        judul = d.metadata.get('title', '-')
        cat = d.metadata.get('category', '-')
        tags = d.metadata.get('topic_tags', '')

        # Log detail untuk user (Transparansi sistem)
        log_buffer += f"   • {i+1}. [{cat}] {judul}\n"
        log_buffer += f"       ↳ Tags: {tags}\n"

        # Context yang masuk ke LLM
        context_str += f"SUMBER: {judul} (Kategori: {cat})\nISI LONTAR: {d.page_content}\n\n"

    yield history, log_buffer

    # [LOG 5: Generation / Synthesis]
    log_buffer += "\n✍️ PROSES 3: GENERATING RESPONSE...\n"
    yield history, log_buffer

    # Prompt Engineering Akademis (Chain of Thought)
    system_prompt = f"""Anda adalah Asisten Peneliti Lontar Bali yang cerdas.
Tugas: Jawab pertanyaan pengguna berdasarkan KONTEKS yang diberikan.

INSTRUKSI KHUSUS:
1. Jawablah dengan nada akademis namun mudah dipahami.
2. JANGAN mengarang (halusinasi). Gunakan hanya fakta dari KONTEKS.
3. Sebutkan nama lontar (sumber) yang menjadi rujukan jawaban.
4. Berikan elaborasi singkat jika relevan. Juga hubungkan dengan konteks budaya Bali.
5. Jika tidak ada info di KONTEKS, katakan "Maaf, informasi tidak tersedia dalam sumber yang diberikan."
6. Jika pertanyaan tentang kategori/asal-usul, fokus pada aspek sejarah dan budaya. Berikan feedback singkat tentang pentingnya lontar tersebut.
7. Jawaban harus dalam bahasa Indonesia yang baik dan benar.
8. Gunakan format paragraf yang rapi dan terstruktur.


KONTEKS:
{context_str}

PERTANYAAN: {user_input}
JAWABAN:"""

    response_text = ""
    try:
        for chunk in llm.stream(system_prompt):
            token = chunk.content
            response_text += token
            history[-1]['content'] = response_text
            yield history, log_buffer
    except Exception as e:
        history[-1]['content'] = f"Error Generasi: {str(e)}"
        yield history, log_buffer

    # [LOG 6: Feedback]
    log_buffer += "\n✨ SELESAI. Siap untuk pertanyaan berikutnya."
    yield history, log_buffer


# --- 5. UI GRADIO (SESUAI REQUEST: KODE LAMA DIPERTAHANKAN) ---

custom_css = """
#chatbot { height: 600px !important; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
#log_panel { background-color: #f8fafc; font-family: monospace; font-size: 0.85em; }
"""

theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate"
).set(
    body_background_fill="#ffffff"
)

with gr.Blocks(title="Lontar AI Expert") as demo:
    gr.Markdown("# 🌿 Lontar AI Knowledge System")

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Log Trace")
            log_view = gr.Textbox(
                label="Process Log",
                lines=24,
                interactive=False,
                elem_id="log_panel"
            )

        with gr.Column(scale=6):
            # Komponen Chatbot yang dipertahankan
            chatbot = gr.Chatbot(
                label="Dialog Lontar",
                elem_id="chatbot",
                avatar_images=(
                    None, "https://cdn-icons-png.flaticon.com/512/4712/4712009.png")
            )

            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Tanya tentang lontar...",
                    scale=5,
                    container=False,
                    autofocus=True
                )
                btn_submit = gr.Button("Kirim", variant="primary", scale=1)

    # Wiring
    txt_input.submit(chat_pipeline, [txt_input, chatbot], [
                     chatbot, log_view]).then(lambda: "", outputs=[txt_input])
    btn_submit.click(chat_pipeline, [txt_input, chatbot], [
                     chatbot, log_view]).then(lambda: "", outputs=[txt_input])

if __name__ == "__main__":
    print("🚀 Meluncurkan Aplikasi...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=theme,
        css=custom_css
    )
