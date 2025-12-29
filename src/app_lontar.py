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

# Model
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

# Setup LLM
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.3,
    keep_alive="1h"
)

# --- 3. LOGIKA RAG ---


def query_expansion(query):
    try:
        prompt = f"Ubah query ini ke istilah baku lontar Bali. Output query saja.\nQuery: {query}"
        response = llm.invoke(prompt)
        return response.content.strip().replace('"', '')
    except:
        return query


def retrieve_hybrid(query, k=4):
    if not DB_READY:
        return []

    docs_bm25 = []
    docs_faiss = []

    try:
        docs_bm25 = bm25_retriever.invoke(query)
    except:
        pass
    try:
        docs_faiss = vector_db.similarity_search(query, k=k)
    except:
        pass

    final_docs = []
    seen = set()

    for d in docs_bm25[:2]:
        if d.page_content not in seen:
            d.metadata['method'] = "Lexical"
            final_docs.append(d)
            seen.add(d.page_content)

    for d in docs_faiss:
        if len(final_docs) >= k:
            break
        if d.page_content not in seen:
            d.metadata['method'] = "Semantic"
            final_docs.append(d)
            seen.add(d.page_content)

    return final_docs

# --- 4. PIPELINE CHAT (MESSAGES FORMAT) ---


def chat_pipeline(user_input, history):
    """
    Format: List of Dictionaries (Standard Gradio v5/v6)
    """
    if not user_input:
        yield history, ""
        return

    if history is None:
        history = []

    # Append User Input
    history.append({"role": "user", "content": user_input})

    # Append Placeholder Assistant
    history.append({"role": "assistant", "content": ""})

    log_buffer = f"🔍 INPUT: {user_input}\n" + "-"*30 + "\n"
    yield history, log_buffer

    if not DB_READY:
        history[-1]['content'] = "⚠️ Database belum siap."
        yield history, log_buffer + "❌ DB ERROR"
        return

    # A. Expansion
    log_buffer += "🧠 EXPANSION...\n"
    yield history, log_buffer
    expanded_query = query_expansion(user_input)
    log_buffer += f"   ↳ '{expanded_query}'\n"
    yield history, log_buffer

    # B. Retrieval
    log_buffer += "📚 RETRIEVAL...\n"
    yield history, log_buffer
    docs = retrieve_hybrid(expanded_query)

    if not docs:
        history[-1]['content'] = "Tidak ditemukan referensi relevan."
        yield history, log_buffer + "❌ NIHIL"
        return

    # C. Context
    context_str = ""
    log_buffer += f"✅ FOUND {len(docs)} DOCS:\n"
    for i, d in enumerate(docs):
        judul = d.metadata.get('title', '-')
        cat = d.metadata.get('category', '-')
        log_buffer += f"   • {i+1}. [{cat}] {judul}\n"
        context_str += f"SUMBER: {judul} ({cat})\nISI: {d.page_content}\n\n"

    yield history, log_buffer

    # D. Generation
    log_buffer += "\n✍️ GENERATING...\n"
    yield history, log_buffer

    system_prompt = f"""Anda Asisten Lontar Bali. Jawab berdasarkan KONTEKS:
    
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
        history[-1]['content'] = f"Error: {str(e)}"
        yield history, log_buffer

    log_buffer += "✨ SELESAI."
    yield history, log_buffer


# --- 5. UI GRADIO (CLEAN & STABLE) ---

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
            # [FINAL FIX] Hapus semua parameter berisiko
            # - Hapus 'bubble_full_width'
            # - Hapus 'show_copy_button'
            # - Hapus 'type'
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
