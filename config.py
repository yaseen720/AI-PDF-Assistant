# config.py

# ---- LLM CONFIG ----
MODEL_NAME = "phi3"
TEMPERATURE = 0

# ---- EMBEDDING MODEL ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- TEXT SPLITTING ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- PATHS ----
DATA_PATH = "data/sample.pdf"
DB_DIR = "db"