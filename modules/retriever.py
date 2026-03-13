# modules/retriever.py

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 2})

    if not vectorstore:
        print("[ERROR] Vectorstore not available.")
        return None

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # number of relevant chunks to retrieve
    )

    print("[INFO] Retriever ready.")
    return retriever