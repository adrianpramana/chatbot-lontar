import os
import pickle
import math
import gradio as gr
from dotenv import load_dotenv

# --- IMPORT OLLAMA
from langchain_ollama import ChatOllama

# --- IMPORTS LAINNYA ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate

# --- 1. SETUP CONFIG ---
load_dotenv()

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
KEYWORD_STORE_PATH = os.path.join(
    BASE_DIR, 'keyword_store', 'bm25_retriever.pkl')

# --- 2. SETUP MODEL LOKAL (OLLAMA) ---
# Pastikan Anda sudah install ollama dan run: 'ollama run llama3.1'
MODEL_NAME = "llama3.1"

print(f"🚀 Menggunakan Local LLM: {MODEL_NAME}")
print("   (Pastikan aplikasi Ollama sudah berjalan di background)")

# --- 3. LOAD RESOURCES with Embeddings ---
print("📂 Memuat Database Lontar...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = None
bm25_retriever = None

try:
    if os.path.exists(VECTOR_STORE_PATH):
        vector_db = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚠️ Vector Store tidak ditemukan. Jalankan ingest_hybrid.py")
except Exception as e:
    print(f"⚠️ Error load FAISS: {e}")

try:
    if os.path.exists(KEYWORD_STORE_PATH):
        with open(KEYWORD_STORE_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("⚠️ Keyword Store tidak ditemukan.")
except Exception as e:
    print(f"⚠️ Error load BM25: {e}")

# --- 4. DYNAMIC HYBRID SEARCH ---


def weighted_hybrid_search(query, k_total=5, alpha=0.5):
    if not vector_db or not bm25_retriever:
        return []

    k_bm25 = math.ceil(k_total * alpha)
    k_faiss = k_total - k_bm25

    combined_docs = []
    seen_contents = set()

    # A. BM25
    if k_bm25 > 0:
        try:
            raw_bm25 = bm25_retriever.invoke(query)
            for doc in raw_bm25[:k_bm25]:
                clean_content = doc.page_content.strip()
                if clean_content not in seen_contents:
                    doc.metadata['source_type'] = "Lexical (Teks Eksak)"
                    combined_docs.append(doc)
                    seen_contents.add(clean_content)
        except:
            pass

    # B. FAISS
    if k_faiss > 0:
        raw_faiss = vector_db.similarity_search(query, k=k_faiss)
        for doc in raw_faiss:
            clean_content = doc.page_content.strip()
            if clean_content not in seen_contents:
                doc.metadata['source_type'] = "Semantic (Makna)"
                combined_docs.append(doc)
                seen_contents.add(clean_content)

    return combined_docs


# --- 5. SETUP LLM (OLLAMA) ---
# Ganti ChatGoogleGenerativeAI dengan ChatOllama
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.3,
    # keep_alive menahan model di RAM agar chat berikutnya cepat
    keep_alive="5m"
)

system_template = """Anda adalah Asisten Peneliti Filologi Lontar Bali.
Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan KONTEKS yang diberikan.

INSTRUKSI:
1. Gunakan Bahasa Indonesia yang baik dan akademis.
2. Jawab HANYA berdasarkan informasi di bagian KONTEKS.
3. Jika informasi tidak ada di konteks, katakan "Maaf, informasi tidak ditemukan dalam naskah."
4. Jika ada istilah Bali Kuno/Kawi, jelaskan artinya.

KONTEKS:
{context}

PERTANYAAN: 
{question}

JAWABAN:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"], template=system_template)

# --- 6. CORE LOGIC ---


def process_query_with_params(message, k_slider, alpha_slider):
    if not message:
        return "Silakan ketik pertanyaan Anda."

    try:
        # A. Retrieval
        docs = weighted_hybrid_search(message, k_total=int(
            k_slider), alpha=float(alpha_slider))

        if not docs:
            return "Maaf, tidak ditemukan informasi relevan. Pastikan database sudah di-ingest."

        # B. Format Context
        context_text = "\n\n".join(
            [f"[{d.metadata.get('title', 'Naskah')}]\n{d.page_content}" for d in docs])

        # C. Generate (OLLAMA)
        final_prompt = prompt_template.format(
            context=context_text, question=message)

        # Invoke Ollama
        response = llm.invoke(final_prompt)

        # D. Output Formatting
        ref_text = ""
        for i, doc in enumerate(docs):
            title = doc.metadata.get('title', 'Naskah')
            src_type = doc.metadata.get('source_type', 'Unknown')
            icon = "🔤" if "Lexical" in src_type else "🧠"
            snippet = doc.page_content[:150].replace('\n', ' ')
            ref_text += f"**{i+1}. {title}** {icon} *({src_type})*\n> \"{snippet}...\"\n\n"

        # Response content dari Ollama bisa langsung string atau object, kita handle aman
        answer_content = response.content if hasattr(
            response, 'content') else str(response)

        return f"{answer_content}\n\n---\n#### 📚 Referensi Naskah Terpilih:\n{ref_text}"

    except Exception as e:
        return f"System Error (Ollama): {str(e)}"

# --- 7. UI EVENT HANDLER ---


def respond(message, history_list, k_val, alpha_val):
    if history_list is None:
        history_list = []

    # Format Dictionary (Universal untuk Gradio 3, 4, 5)
    history_list.append({"role": "user", "content": str(message)})

    # Proses Jawaban
    bot_message = process_query_with_params(message, k_val, alpha_val)

    history_list.append({"role": "assistant", "content": str(bot_message)})

    return "", history_list, history_list


# --- 8. UI GRADIO BLOCKS ---
academic_css = """
body { background-color: #f4f6f9; }
.gradio-container { max-width: 1200px !important; margin: auto; }
#chatbot { 
    height: 600px !important; 
    background: white; 
    border-radius: 10px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1 { color: #2c3e50; font-family: 'Georgia', serif; text-align: center; }
"""

if __name__ == "__main__":
    with gr.Blocks(title="Lontar Knowledge System (Local)") as demo:

        history_state = gr.State([])

        gr.Markdown(f"""
        # 🏛️ Lontar Knowledge System (LLM Berbasis Lokal)
        Running on **Ollama ({MODEL_NAME})** - Created with ❤️ by Made Adrian.
        """)

        with gr.Row():
            # KIRI: Chat Area
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Dialog Naskah",
                    show_label=False
                )

                with gr.Row():
                    txt_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ketik pertanyaan (Contoh: Apa makna filosofis Panca Warna?)...",
                        scale=8,
                        container=False
                    )
                    btn_submit = gr.Button(
                        "🔍 Analisis", variant="primary", scale=1)

            # KANAN: Control Panel
            with gr.Column(scale=3):
                with gr.Accordion("⚙️ Parameter RAG", open=True):
                    slider_k = gr.Slider(
                        minimum=1, maximum=10, step=1, value=4,
                        label="📚 Jumlah Referensi (Top-K)",
                        info="Banyak dokumen yang dibaca AI"
                    )
                    slider_alpha = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.1, value=0.5,
                        label="⚖️ Bobot Pencarian (Alpha)",
                        info="0.0 (Makna) <---> 1.0 (Kata Kunci)"
                    )

                gr.Markdown("### 💡 Contoh Pertanyaan")
                gr.Examples(
                    examples=[
                        "Bagaimana wujud Bhatara Kala?",
                        "Jelaskan filosofi 'Haywa Kita Tan Tutur'!",
                        "Apa saja isi naskah leksikografi dan tata bahasa (Adiswara, Ekalavya, Kretabasa, Suksmabasa, Cantakaparwa, Dasanama)",
                        "Jelaskan Tata cara berperilaku di Pura (Krama Pura) menurut naskah lontar",
                        "Bagaimana Upacara penyucian bangunan (Pamlaspas)"
                    ],
                    inputs=txt_input
                )

        # --- WIRING EVENT ---
        txt_input.submit(
            fn=respond,
            inputs=[txt_input, history_state, slider_k, slider_alpha],
            outputs=[txt_input, chatbot, history_state]
        )

        btn_submit.click(
            fn=respond,
            inputs=[txt_input, history_state, slider_k, slider_alpha],
            outputs=[txt_input, chatbot, history_state]
        )

    print(f"🚀 Aplikasi berjalan dengan LOCAL LLM: {MODEL_NAME}")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=academic_css
    )
