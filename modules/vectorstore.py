# modules/vectorstore.py

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, DB_DIR


def get_vectorstore(db_path):

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    print("[INFO] Vector DB Ready.")
    return vectorstore


def add_documents_to_db(vectorstore, chunks):

    if not chunks:
        return

    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print("[INFO] Documents stored successfully.")