import streamlit as st
import os
import time
import pickle
import base64

from modules.splitter import split_documents
from modules.vectorstore import get_vectorstore, add_documents_to_db
from modules.retriever import get_retriever
from modules.chatbot import create_llm, generate_response
from langchain_community.document_loaders import PyPDFLoader


# ------------------ CONFIG ------------------

st.set_page_config(page_title="PDF AI Assistant", layout="wide")

DATA_FOLDER = "data"
CACHE_FOLDER = "cache"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)


# ------------------ SESSION STATE ------------------

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = create_llm()

if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

if "show_suggested_questions" not in st.session_state:
    st.session_state.show_suggested_questions = False


# ------------------ HELPER ------------------

def shorten_name(name, length=18):
    if len(name) > length:
        return name[:length] + "..."
    return name


def apply_rename(pdf_name, rename_key):
    new_name = st.session_state.get(rename_key, "").strip()
    if new_name and new_name != pdf_name and pdf_name in st.session_state.pdf_files:
        old_file = os.path.join(DATA_FOLDER, pdf_name)
        new_file = os.path.join(DATA_FOLDER, new_name)
        if os.path.exists(old_file):
            os.rename(old_file, new_file)

        st.session_state.pdf_files[new_name] = st.session_state.pdf_files.pop(pdf_name)
        st.session_state.chat_store[new_name] = st.session_state.chat_store.pop(pdf_name)
        if st.session_state.current_pdf == pdf_name:
            st.session_state.current_pdf = new_name

        st.session_state[f"show_menu_{pdf_name}"] = False
        st.session_state[rename_key] = new_name


def delete_pdf(pdf_name):
    if pdf_name in st.session_state.pdf_files:
        del st.session_state.pdf_files[pdf_name]
    if pdf_name in st.session_state.chat_store:
        del st.session_state.chat_store[pdf_name]
    file_path = os.path.join(DATA_FOLDER, pdf_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    if st.session_state.current_pdf == pdf_name:
        st.session_state.current_pdf = None
    st.session_state[f"show_menu_{pdf_name}"] = False


# ------------------ SIDEBAR ------------------

if st.sidebar.button("➕ New Chat", key="new_chat"):
    st.session_state.current_pdf = None
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.show_suggested_questions = False

st.sidebar.markdown("## 📂 Document History")

for pdf_name in list(st.session_state.pdf_files.keys()):

    col1, col2 = st.sidebar.columns([5,1])

    display_name = shorten_name(pdf_name)

    if col1.button(display_name, help=pdf_name, use_container_width=True, key=f"pdf_{pdf_name}"):
        st.session_state.current_pdf = pdf_name
        st.session_state.vectorstore = st.session_state.pdf_files[pdf_name]
        st.session_state.retriever = get_retriever(st.session_state.vectorstore)

    if col2.button("⋯", key=f"menu_{pdf_name}", help="File actions", use_container_width=True):
        key = f"show_menu_{pdf_name}"
        st.session_state[key] = not st.session_state.get(key, False)

    if st.session_state.get(f"show_menu_{pdf_name}", False):
        rename_input_key = f"rename_input_{pdf_name}"
        if rename_input_key not in st.session_state:
            st.session_state[rename_input_key] = pdf_name

        new_name = st.sidebar.text_input(
            f"Rename {pdf_name}",
            value=st.session_state.get(rename_input_key, pdf_name),
            key=rename_input_key,
            on_change=apply_rename,
            args=(pdf_name, rename_input_key),
        )

        if st.sidebar.button("Delete", key=f"delete_{pdf_name}"):
            delete_pdf(pdf_name)




st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear All PDFs", use_container_width=True):

    st.session_state.pdf_files.clear()
    st.session_state.chat_store.clear()
    st.session_state.current_pdf = None
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.show_suggested_questions = False

    for file in os.listdir(DATA_FOLDER):
        os.remove(os.path.join(DATA_FOLDER, file))



# ------------------ MAIN ------------------

st.title("📄 AI PDF Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    if uploaded_file.name not in st.session_state.pdf_files:

        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = split_documents(documents)

            from config import DB_DIR
            pdf_db_path = os.path.join(DB_DIR, uploaded_file.name)

            vectorstore = get_vectorstore(pdf_db_path)
            add_documents_to_db(vectorstore, chunks)

        st.session_state.pdf_files[uploaded_file.name] = vectorstore
        st.session_state.chat_store[uploaded_file.name] = []
        st.session_state.current_pdf = uploaded_file.name

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = get_retriever(vectorstore)

        st.success("PDF uploaded successfully!")

    else:
        st.session_state.current_pdf = uploaded_file.name


# ---------------- ACTION BUTTONS ----------------

if st.session_state.current_pdf:

    # Suggested Questions (moved to chat section for first time only)
    st.session_state.show_suggested_questions = True


# ---------------- PDF PREVIEW ----------------

if st.session_state.current_pdf:

    st.subheader("📄 Document Preview")

    pdf_path = os.path.join(DATA_FOLDER, st.session_state.current_pdf)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
    <iframe
    src="data:application/pdf;base64,{base64_pdf}"
    width="100%"
    height="500">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)


# ---------------- CHAT ----------------

if st.session_state.current_pdf:

    pdf_name = st.session_state.current_pdf

    if pdf_name not in st.session_state.chat_store:
        st.session_state.chat_store[pdf_name] = []

    st.subheader(f"Chatting with: {pdf_name}")

    if st.session_state.show_suggested_questions and len(st.session_state.chat_store[pdf_name]) == 0:
        st.markdown("### 💡 Suggested Questions")
        col1, col2, col3, col4 = st.columns(4)

        if col1.button("📄 Summarize PDF", key="q1"):
            st.session_state.auto_prompt = "Summarize this document."

        if col2.button("🧠 Key Points", key="q2"):
            st.session_state.auto_prompt = "Give key points of this document."

        if col3.button("📊 Main Topics", key="q3"):
            st.session_state.auto_prompt = "Explain the main topics."

        if col4.button("🔍 Important Details", key="q4"):
            st.session_state.auto_prompt = "Give important details from the document."

        st.session_state.show_suggested_questions = False

    for user_msg, bot_msg in st.session_state.chat_store[pdf_name]:

        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            st.markdown(bot_msg)


# ---------------- INPUT ----------------

query = st.chat_input("Ask something about your PDF...")


if "auto_prompt" in st.session_state:
    query = st.session_state.auto_prompt
    del st.session_state.auto_prompt


if query and st.session_state.retriever:

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        loading = st.empty()

        # Animated generating dots
        for i in range(8):
            dot_count = i % 4
            loading.info("🤖 Generating" + "." * dot_count)
            time.sleep(0.2)

        response = generate_response(
            st.session_state.llm,
            st.session_state.retriever,
            query
        )

        loading.empty()
        st.markdown(response)

    st.session_state.chat_store.setdefault(st.session_state.current_pdf, []).append((query, response))