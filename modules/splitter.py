# modules/splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """
    Splits loaded documents into smaller chunks
    for better embedding and retrieval performance.
    """

    if not documents:
        print("[ERROR] No documents received for splitting.")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        splits = splitter.split_documents(documents)

        print(f"[INFO] Split into {len(splits)} chunks.")
        return splits

    except Exception as e:
        print(f"[ERROR] Failed during splitting: {e}")
        return None