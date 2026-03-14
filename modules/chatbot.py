from langchain_community.chat_models import ChatOllama
from config import MODEL_NAME, TEMPERATURE
import streamlit as st


def create_llm():
    """
    Initializes the local LLM using Ollama.
    """

    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )
        print(f"[INFO] LLM '{MODEL_NAME}' loaded successfully.")
        return llm

    except Exception as e:
        print(f"[ERROR] Failed to load LLM: {e}")
        return None


def generate_response(llm, retriever, query):
    """
    Generates answer using retrieved context + conversation memory.
    """

    if llm is None:
        return "LLM not initialized."

    if retriever is None:
        return "No PDF loaded."

    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant content found in this PDF."

        # Limit retrieved docs
        docs = docs[:4]

        # Build context
        context = "\n\n".join(
            doc.page_content for doc in docs if hasattr(doc, "page_content")
        )

        # Ensure chat history exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Use last 2 messages for memory
        recent_history = st.session_state.chat_history[-2:]

        history_text = ""
        for q, a in recent_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        # Prompt
        prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not in the context, say:
"I couldn't find that in the document."

Conversation History:
{history_text}

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        # Extract response text
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        # Save memory
        st.session_state.chat_history.append((query, answer))

        return answer

    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")
        return "Error generating response."
