
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

st.set_page_config(page_title="Arabic-English Chatbot", layout="wide")
st.markdown("""<style>.block-container{padding-top:1rem;}</style>""", unsafe_allow_html=True)

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Save/load chat history
HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Initialize
if "chat_id" not in st.session_state:
    st.session_state.chat_id = "Default Chat"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Sidebar - Chat management
with st.sidebar:
    st.title("🧠 Your Chats")
    new_chat = st.button("➕ New Chat")
    if new_chat:
        new_id = f"Chat {len(st.session_state.chat_history) + 1}"
        st.session_state.chat_id = new_id
        st.session_state.chat_history[new_id] = []
        save_chat_history(st.session_state.chat_history)

    delete_chat = st.button("🗑️ Delete Current Chat")
    if delete_chat and st.session_state.chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[st.session_state.chat_id]
        st.session_state.chat_id = "Default Chat"
        save_chat_history(st.session_state.chat_history)

    st.markdown("---")
    for cid in st.session_state.chat_history.keys():
        if st.button(cid):
            st.session_state.chat_id = cid

# Message display
st.title("🤖 Arabic-English Chatbot")
st.markdown("Current Chat: **{}**".format(st.session_state.chat_id))
st.markdown("---")

if st.session_state.chat_id not in st.session_state.chat_history:
    st.session_state.chat_history[st.session_state.chat_id] = []

for msg in st.session_state.chat_history[st.session_state.chat_id]:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])

# Prompt box
prompt = st.chat_input("اكتب سؤالك هنا / Ask your question here")

# Encoding wrapper
def encode_prompt(prompt):
    chat_prompt = f"[INST] {prompt.strip()} [/INST]"
    return tokenizer(chat_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        inputs = encode_prompt(prompt)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.95, temperature=0.7)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_response = decoded.split("[/INST]")[-1].strip()

        message_placeholder.markdown(full_response)

    st.session_state.chat_history[st.session_state.chat_id].append({
        "user": prompt,
        "assistant": full_response
    })
    save_chat_history(st.session_state.chat_history)
