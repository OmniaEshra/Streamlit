import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch
import faiss
import json
import os
from uuid import uuid4

# --- Streamlit Page Config ---
st.set_page_config(page_title="رفيق", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri&display=swap');
    .title-custom {
        font-family: 'Amiri', serif;
        font-size: 3.5rem;
        font-weight: bold;
        background: -webkit-linear-gradient(lightgreen, lightblue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        direction: rtl;
        text-align: center;
    }
    .chat-bubble {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 1rem;
        max-width: 80%;
        direction: rtl;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .chat-user {
        background-color: #e8f5e9;
        text-align: right;
        margin-left: auto;
    }
    .chat-bot {
        background-color: #e3f2fd;
        text-align: left;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Load Resources with Caching ---
@st.cache_resource
def load_faiss_and_docs():
    index_path = hf_hub_download(
        repo_id="OmniaSh/faiss_data",
        filename="index.faiss",
        repo_type="dataset",
        token=HF_TOKEN
    )
    docs_path = hf_hub_download(
        repo_id="OmniaSh/faiss_data",
        filename="docs.json",
        repo_type="dataset",
        token=HF_TOKEN
    )
    index = faiss.read_index(index_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs

@st.cache_resource
def load_embedder():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "OmniaSh/mistral",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "OmniaSh/mistral",
        use_auth_token=HF_TOKEN
    )
    return model, tokenizer

# --- Ask Function ---
def ask(question, index, docs, embedder, model, tokenizer, top_k=5, max_new_tokens=200):
    query_embedding = embedder.encode(question, normalize_embeddings=True)
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    retrieved_chunks = [docs[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)
    prompt = f"السؤال: {question}\n\nالمحتوى:\n{context}\n\nالإجابة:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("الإجابة:")[-1].strip()

# --- Chat Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    chat_id = str(uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" width="40"/>
        </div>
    """, unsafe_allow_html=True)

    if st.button("➕ محادثة جديدة", use_container_width=True):
        new_chat_id = str(uuid4())
        st.session_state.chats[new_chat_id] = []
        st.session_state.current_chat = new_chat_id
        st.experimental_rerun()

    st.markdown("### 💬 سجل المحادثات")
    for cid in st.session_state.chats:
        label = f"محادثة {list(st.session_state.chats).index(cid) + 1}"
        if st.button(label, key=cid):
            st.session_state.current_chat = cid
            st.experimental_rerun()

# --- Title ---
st.markdown('<div class="title-custom">رفيق</div>', unsafe_allow_html=True)

# --- Show Chat History ---
chat_history = st.session_state.chats[st.session_state.current_chat]
for msg in chat_history:
    role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
    st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# --- Input ---
user_input = st.chat_input("✍️ اكتب سؤالك هنا...")

if user_input:
    # Append and show user message
    st.session_state.chats[st.session_state.current_chat].append({"role": "user", "content": user_input})
    st.markdown(f"<div class='chat-bubble chat-user'>{user_input}</div>", unsafe_allow_html=True)

    # Answer generation
    with st.spinner("🔍 جاري توليد الإجابة..."):
        index, docs = load_faiss_and_docs()
        embedder = load_embedder()
        model, tokenizer = load_model()
        answer = ask(user_input, index, docs, embedder, model, tokenizer)

    # Append and show assistant message
    st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": answer})
    st.markdown(f"<div class='chat-bubble chat-bot'>{answer}</div>", unsafe_allow_html=True)
