# config.py

# ---- LLM CONFIG ----
# Provider: "groq" (for deployment)
MODEL_PROVIDER = "groq"
OLLAMA_MODEL = "phi3"
GEMINI_MODEL = "gemini-1.0"
GROQ_MODEL = "llama3-8b-8192"

MODEL_NAME = GROQ_MODEL
TEMPERATURE = 0

# ---- EMBEDDING MODEL ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- TEXT SPLITTING ----
CHUNK_SIZE = 300  # Reduced for lower memory usage
CHUNK_OVERLAP = 50

# ---- PATHS ----
DATA_PATH = "data/sample.pdf"
DB_DIR = "db"