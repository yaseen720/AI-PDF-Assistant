# config.py

# ---- LLM CONFIG ----
# Provider: "ollama", "gemini" or "openai"
MODEL_PROVIDER = "ollama"
OLLAMA_MODEL = "phi3"
GEMINI_MODEL = "gemini-1.0"
OPENAI_MODEL = "gpt-4o-mini"

MODEL_NAME = OLLAMA_MODEL
TEMPERATURE = 0

# ---- EMBEDDING MODEL ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- TEXT SPLITTING ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- PATHS ----
DATA_PATH = "data/sample.pdf"
DB_DIR = "db"