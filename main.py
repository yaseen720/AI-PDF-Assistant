import streamlit as st
import os
import time
import pickle
import base64

from modules.splitter import split_documents
from modules.vectorstore import get_vectorstore, add_documents_to_db
from modules.retriever import get_retriever
from modules.chatbot import create_llm, generate_response
from modules.loader import load_file, is_supported_file

SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]


# ------------------ CONFIG ------------------

st.set_page_config(page_title="Document AI Assistant", layout="wide")

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

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "compare_report" not in st.session_state:
    st.session_state.compare_report = None

if "selected_candidates" not in st.session_state:
    st.session_state.selected_candidates = []


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


def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path


def get_text_preview(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".docx", ".png", ".jpg", ".jpeg"]:
        documents = load_file(file_path)
        if documents:
            return "\n\n".join(doc.page_content for doc in documents)
    return None


def render_preview(file_name):
    file_path = os.path.join(DATA_FOLDER, file_name)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        with open(file_path, "rb") as f:
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

    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            from PIL import Image
            image = Image.open(file_path)
            st.image(image, caption=file_name, use_column_width=True)
            text_preview = get_text_preview(file_path)
            if text_preview:
                st.markdown("### Extracted Text")
                st.text_area("Image OCR text", text_preview, height=250)
        except Exception as e:
            st.warning(f"Unable to preview image: {e}")

    elif ext == ".docx":
        text_preview = get_text_preview(file_path)
        st.markdown("### Document Preview")
        st.text_area("DOCX text", text_preview or "No text extracted from this document.", height=300)

    else:
        st.write("No preview available for this file type.")


def call_llm_direct(llm, prompt):
    if llm is None:
        return "LLM not initialized."

    try:
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        return f"Error generating response: {e}"


def analyze_resume(file_name):
    if file_name not in st.session_state.pdf_files:
        return "Resume not available."

    retriever = get_retriever(st.session_state.pdf_files[file_name])
    prompt = (
        "You are a resume analyst. Use only the information from the resume content retrieved below.\n"
        "Extract structured data: Name, Skills, Experience, Education, Projects, LinkedIn, GitHub.\n"
        "Score the resume from 0 to 100 using the following breakdown:\n"
        "- Skills: 30 points\n"
        "- Experience: 25 points\n"
        "- Projects: 25 points\n"
        "- Education & Completeness: 20 points\n"
        "Decision must be one of: Hire, Consider, Reject.\n"
        "Return output exactly in this format:\n\n"
        "Candidate Score: X/100\n\n"
        "Breakdown:\n"
        "- Skills: X/30\n"
        "- Experience: X/25\n"
        "- Projects: X/25\n"
        "- Education: X/20\n\n"
        "Decision:\n"
        "- Hire / Consider / Reject\n\n"
        "Explanation:\n"
        "- Give clear reasoning for the score and mention strengths and weaknesses.\n"
        "Use headings and bullet points only if they help clarity. Do not add extra sections."
    )

    history_backup = st.session_state.get("chat_history", []).copy()
    result = generate_response(st.session_state.llm, retriever, prompt)
    st.session_state.chat_history = history_backup
    return result


def compare_candidates(candidate_names):
    if not candidate_names or len(candidate_names) < 2:
        return "Select at least two candidates to compare."

    summaries = []
    for name in candidate_names:
        if name not in st.session_state.pdf_files:
            continue
        summary = analyze_resume(name)
        summaries.append(f"### Candidate: {name}\n{summary}")

    if len(summaries) < 2:
        return "Need at least two valid candidate resumes to compare."

    combined_prompt = (
        "You are comparing these candidates based on their resume analysis.\n"
        "Rank all candidates and choose the best candidate.\n"
        "Explain the reason for selection and provide a clear ranking.\n"
        "Use headings, bullet points, and include the best candidate at the top.\n\n"
    )
    combined_prompt += "\n\n".join(summaries)
    return call_llm_direct(st.session_state.llm, combined_prompt)


def extract_score_from_report(report):
    import re
    if not report:
        return None
    match = re.search(r"Score[:\s]+(\d{1,3})", report, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 100)
    return None


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

st.title("📄 AI Document Assistant")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=SUPPORTED_UPLOAD_TYPES,
    accept_multiple_files=True,
    help="Upload PDF, DOCX, PNG, JPG, or JPEG files.",
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.pdf_files:
            file_path = save_uploaded_file(uploaded_file)
            with st.spinner(f"Processing {uploaded_file.name}..."):
                documents = load_file(file_path)
                if not documents:
                    st.error(f"Unable to extract text from {uploaded_file.name}.")
                    continue

                chunks = split_documents(documents)
                if not chunks:
                    st.error(f"Failed to split {uploaded_file.name} into chunks.")
                    continue

                from config import DB_DIR
                pdf_db_path = os.path.join(DB_DIR, uploaded_file.name)

                vectorstore = get_vectorstore(pdf_db_path)
                add_documents_to_db(vectorstore, chunks)

            st.session_state.pdf_files[uploaded_file.name] = vectorstore
            st.session_state.chat_store[uploaded_file.name] = []
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = get_retriever(vectorstore)

            st.success(f"{uploaded_file.name} uploaded successfully!")
        else:
            st.session_state.current_pdf = uploaded_file.name


# ---------------- ACTION BUTTONS ----------------

if st.session_state.current_pdf:

    # Suggested Questions (moved to chat section for first time only)
    st.session_state.show_suggested_questions = True


# ---------------- PDF PREVIEW ----------------

if st.session_state.current_pdf:
    pdf_name = st.session_state.current_pdf
    st.subheader("📄 Document Preview")
    render_preview(pdf_name)

    st.markdown("---")
    st.subheader("⚙️ Resume Tools")

    analyze_col, compare_col = st.columns([2, 3])
    if analyze_col.button("Analyze Resume", key="analyze_resume"):
        st.session_state.analysis_results[pdf_name] = analyze_resume(pdf_name)

    candidate_options = list(st.session_state.pdf_files.keys())
    selected = compare_col.multiselect(
        "Select candidates to compare",
        options=candidate_options,
        default=st.session_state.selected_candidates or candidate_options[:min(5, len(candidate_options))],
        key="candidate_selector",
    )
    st.session_state.selected_candidates = selected

    if compare_col.button("Compare Candidates", key="compare_candidates_btn"):
        if len(selected) < 2:
            st.warning("Upload and select at least two candidates to compare.")
        else:
            st.session_state.compare_report = compare_candidates(selected)

    report = st.session_state.analysis_results.get(pdf_name)
    if report:
        score = extract_score_from_report(report)
        if score is not None:
            st.metric("Resume Score", f"{score}/100")

        st.subheader("🔎 Resume Analysis")
        st.markdown(report)

    if st.session_state.compare_report:
        st.subheader("🧾 Candidate Comparison")
        st.markdown(st.session_state.compare_report)


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

query = st.chat_input("Ask something about your document...")


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