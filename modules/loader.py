# modules/loader.py

import os
from langchain_community.document_loaders import PyPDFLoader


DATA_FOLDER = "data"


def load_all_pdfs():
    """
    Loads all PDFs from data folder.
    Adds filename as metadata for each chunk.
    """

    documents = []

    try:
        for file in os.listdir(DATA_FOLDER):
            if file.endswith(".pdf"):
                file_path = os.path.join(DATA_FOLDER, file)
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = file

                documents.extend(docs)

        print(f"[INFO] Loaded {len(documents)} pages from all PDFs.")
        return documents

    except Exception as e:
        print(f"[ERROR] Failed to load PDFs: {e}")
        return None