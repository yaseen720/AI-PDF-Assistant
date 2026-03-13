# app.py

from modules.loader import load_all_pdfs
from modules.splitter import split_documents
from modules.vectorstore import get_vectorstore, add_documents_to_db
from modules.retriever import get_retriever
from modules.chatbot import create_llm, generate_response


def main():

    print("🚀 Starting Multi-PDF AI Assistant...\n")

    # 1️⃣ Load ALL PDFs from data folder
    documents = load_all_pdfs()
    if not documents:
        return

    # 2️⃣ Split into chunks
    chunks = split_documents(documents)
    if not chunks:
        return

    # 3️⃣ Load Vector DB
    vectorstore = get_vectorstore()
    if not vectorstore:
        return

    # 4️⃣ Add only NEW PDFs to DB
    add_documents_to_db(vectorstore, chunks)

    # 5️⃣ Create Retriever
    retriever = get_retriever(vectorstore)
    if not retriever:
        return

    # 6️⃣ Load LLM
    llm = create_llm()
    if not llm:
        return

    print("\n✅ Multi-PDF Chatbot Ready! Type 'exit' to quit.\n")

    # 7️⃣ Chat Loop
    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        response = generate_response(llm, retriever, query)

        print("\nBot:", response)
        print("-" * 60)


if __name__ == "__main__":
    main()