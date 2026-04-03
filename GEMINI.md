# PDF AI Assistant

A comprehensive AI-powered PDF assistant built with LangChain, Streamlit, and ChromaDB. This project allows users to upload, process, and interact with PDF documents through both a web-based UI and a CLI.

## Project Overview

The PDF AI Assistant leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on the content of uploaded PDF files. It is specifically optimized for resume analysis but can handle any PDF document.

### Main Technologies
- **Language:** Python
- **UI Framework:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Default LLM:** [Ollama](https://ollama.com/) (model: `phi3`)
- **Optional Gemini support:** `GoogleGemini` via LangChain

### Gemini setup
1. Install Gemini support:
   - `pip install langchain[google-gemini]` or `pip install google-ai`
2. Set provider in `config.py`:
   - `MODEL_PROVIDER = "gemini"`
   - `GEMINI_MODEL = "gemini-1.0"` (or your preferred model)
3. Ensure `GOOGLE_API_KEY` / auth credentials are set if required by your environment.

### Switching providers
- `MODEL_PROVIDER = "ollama"` for local Ollama
- `MODEL_PROVIDER = "gemini"` for Google Gemini
- `MODEL_PROVIDER = "openai"` for OpenAI

## Project Structure

- `streamlit_app.py`: The main entry point for the Streamlit web application.
- `app.py`: A CLI-based interface for interacting with the assistant.
- `config.py`: Centralized configuration for models, paths, and hyperparameters.
- `modules/`: Core logic components:
    - `chatbot.py`: LLM initialization and response generation logic.
    - `loader.py`: PDF document loading utilities.
    - `retriever.py`: Vector store retrieval configuration.
    - `splitter.py`: Text splitting and chunking strategies.
    - `vectorstore.py`: ChromaDB integration and document persistence.
- `data/`: Directory for storing uploaded PDF files.
- `db/`: Directory for persistent ChromaDB storage.
- `cache/`: Directory for temporary cache files.

## Building and Running

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- Pull the default model: `ollama pull phi3`

### Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Web Interface (Streamlit)
```bash
streamlit run streamlit_app.py
```

#### CLI Interface
```bash
python app.py
```

## Development Conventions

- **Modular Logic:** Keep core logic within the `modules/` directory to ensure reusability across UI and CLI interfaces.
- **Configuration:** Use `config.py` for all adjustable parameters like model names and chunk sizes.
- **State Management:** In the Streamlit app, use `st.session_state` to manage chat history and document context across re-runs.
- **Error Handling:** Implement try-except blocks in module functions to handle LLM or Vector Store failures gracefully.
