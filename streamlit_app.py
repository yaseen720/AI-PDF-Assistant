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


def shorten_name(name, length=18):
    if len(name) > length:
        return name[:length] + "..."
    return name



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


# ------------------ HELPER ------------------

def save_chat(pdf_name):
    with open(f"{CACHE_FOLDER}/{pdf_name}_chat.pkl", "wb") as f:
        pickle.dump(st.session_state.chat_store[pdf_name], f)


# ------------------ SIDEBAR ------------------

st.sidebar.markdown("## 📂 Document History")

sidebar_container = st.sidebar.container()

with sidebar_container:

    for pdf_name in list(st.session_state.pdf_files.keys()):

        col1, col2 = st.columns([5,1])

        display_name = shorten_name(pdf_name)

        if col1.button(display_name, help=pdf_name, use_container_width=True):

            st.session_state.current_pdf = pdf_name
            st.session_state.vectorstore = st.session_state.pdf_files[pdf_name]
            st.session_state.retriever = get_retriever(
                st.session_state.vectorstore
            )


        with col2:
            with st.popover("⋮"):

                new_name = st.text_input(
                    "Rename PDF",
                    key=f"rename_{pdf_name}"
                )

                if st.button("Save Name", key=f"save_{pdf_name}"):

                    if new_name and new_name != pdf_name:

                        st.session_state.pdf_files[new_name] = \
                            st.session_state.pdf_files[pdf_name]

                        st.session_state.chat_store[new_name] = \
                            st.session_state.chat_store.get(pdf_name, [])

                        del st.session_state.pdf_files[pdf_name]

                        if pdf_name in st.session_state.chat_store:
                            del st.session_state.chat_store[pdf_name]

                        if st.session_state.current_pdf == pdf_name:
                            st.session_state.current_pdf = new_name

                        st.rerun()

                if st.button("🗑 Delete PDF", key=f"delete_{pdf_name}"):

                    del st.session_state.pdf_files[pdf_name]

                    if pdf_name in st.session_state.chat_store:
                        del st.session_state.chat_store[pdf_name]

                    file_path = os.path.join(DATA_FOLDER, pdf_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    if st.session_state.current_pdf == pdf_name:
                        st.session_state.current_pdf = None

                    st.rerun()


st.sidebar.markdown("---")

# Clear button
if st.sidebar.button(
    "🗑 Clear All PDFs",
    use_container_width=True,
    key="clear_all_btn"
):

    st.session_state.pdf_files.clear()
    st.session_state.chat_store.clear()
    st.session_state.current_pdf = None

    for file in os.listdir(DATA_FOLDER):
        os.remove(os.path.join(DATA_FOLDER, file))

    st.rerun()


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


# ------------------ SPLIT SCREEN LAYOUT ------------------

import base64

col_chat, col_pdf = st.columns([2,1])

# ---------------- PDF PREVIEW ----------------

with col_pdf:

    st.subheader("📄 Document Preview")

    if st.session_state.current_pdf:

        pdf_path = os.path.join(DATA_FOLDER, st.session_state.current_pdf)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        page = st.session_state.get("jump_page", 1)

        pdf_display = f"""
        <iframe
        src="data:application/pdf;base64,{base64_pdf}#page={page}"
        width="100%"
        height="500"
        style="border-radius:10px;border:1px solid #ddd;">
        </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)


# ---------------- CHAT ----------------

with col_chat:

    if st.session_state.current_pdf:

        pdf_name = st.session_state.current_pdf

        if pdf_name not in st.session_state.chat_store:
            st.session_state.chat_store[pdf_name] = []

        st.subheader(f"Chatting with: {pdf_name}")

        # CHAT HISTORY
        for user_msg, bot_msg in st.session_state.chat_store[pdf_name]:

            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                st.markdown(bot_msg)

# ---------------- SUGGESTED QUESTIONS ----------------

if st.session_state.current_pdf:

    st.markdown("### 💡 Suggested Questions")

    # Default buttons
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📄 Summarize PDF"):
        st.session_state.auto_prompt = "Summarize this document."

    if col2.button("🧠 Key Points"):
        st.session_state.auto_prompt = "What are the key points in this document?"

    if col3.button("📊 Main Topics"):
        st.session_state.auto_prompt = "What are the main topics discussed?"

    if col4.button("🔍 Important Details"):
        st.session_state.auto_prompt = "What are the most important details in this document?"

# ---------------- INPUT (ALWAYS LAST) ----------------

prompt = st.chat_input("Ask something about the document...")

# check if suggested button was clicked
if "auto_prompt" in st.session_state:
    prompt = st.session_state.auto_prompt
    del st.session_state.auto_prompt

if prompt and st.session_state.retriever:

    pdf_name = st.session_state.current_pdf

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Generating..."):

            response = generate_response(
                st.session_state.llm,
                st.session_state.retriever,
                prompt
            )

            message_placeholder = st.empty()
            full_text = ""

            for char in response:

                full_text += char
                message_placeholder.markdown(full_text)

                time.sleep(0.002)

            message_placeholder.markdown(full_text)

         