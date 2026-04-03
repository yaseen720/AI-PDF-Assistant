from langchain_community.chat_models import ChatOllama

# heat-proof for LangChain version mismatches
try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain.chat_models import GoogleGemini
except ImportError:
    GoogleGemini = None

try:
    from langchain_openai import OpenAI
except ImportError:
    OpenAI = None

from config import MODEL_PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, OPENAI_MODEL, TEMPERATURE
import streamlit as st


def create_llm():
    """Create and return an LLM object based on config MODEL_PROVIDER."""
    try:
        provider = MODEL_PROVIDER.lower().strip()

        if provider == "ollama":
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=TEMPERATURE,
                num_predict=256,
                top_k=40,
                top_p=0.9,
            )

        elif provider == "gemini":
            if GoogleGemini is None:
                raise ImportError("GoogleGemini is not installed. Run pip install langchain[google-gemini] or google-ai.")
            llm = GoogleGemini(model=GEMINI_MODEL, temperature=TEMPERATURE)

        elif provider == "openai":
            if ChatOpenAI is not None:
                llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE)
            elif OpenAI is not None:
                llm = OpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE)
            else:
                raise ImportError("OpenAI chat model class not found. Install langchain-openai or upgrade langchain.")

        else:
            raise ValueError(f"Unknown MODEL_PROVIDER '{MODEL_PROVIDER}'. Use 'ollama', 'gemini', or 'openai'.")

        print(f"[INFO] {provider.capitalize()} LLM loaded ({MODEL_PROVIDER}).")
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
        try:
            docs = retriever.invoke(query)
        except AttributeError:
            # Fallback if retriever doesn't have invoke method
            docs = retriever.similarity_search(query) if hasattr(retriever, 'similarity_search') else retriever.get_relevant_documents(query)

        if not docs:
            return "No relevant content found in this PDF."

        # Limit retrieved docs
        docs = docs[:4]

        # Build context
        context = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs
        )

        if not context.strip():
            return "No relevant content found in this PDF."

        # Force detect links
        context += "\n\nAlso check for LinkedIn and GitHub links in the resume even if hidden."


        # Ensure chat history exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Use last 2 messages for memory
        recent_history = st.session_state.chat_history[-2:]

        history_text = ""
        for q, a in recent_history:
            history_text += f"User: {q}\nAssistant: {a}\n"

        # Detect resume mode
        resume_mode = any(word in query.lower() for word in ["resume", "cv", "candidate"])

        if "summarize" in query.lower():
            system_instruction = "Summarize the document clearly in short."

        elif "key points" in query.lower():
            system_instruction = "Extract important key bullet points from the document."

        elif "topics" in query.lower():
            system_instruction = "List main topics discussed in the document."

        elif "details" in query.lower():
            system_instruction = "Extract important detailed information from the document."

        elif "skills" in query.lower():
            system_instruction = "Extract all technical and soft skills from the resume."

        elif "summary" in query.lower():
            system_instruction = "Give a professional short summary of the candidate."

        elif "analyze" in query.lower() or "resume" in query.lower():
            system_instruction = """Analyze the resume and extract:
- Name
- Contact Info
- Skills
- Education
- Experience
- Projects
- LinkedIn (if available)
- GitHub (if available)

Return the answer in clean, human-readable format.
Use headings and bullet points.
Do NOT return JSON or code format."""

        else:
            system_instruction = """You are a helpful AI assistant.
Use ONLY the provided context to answer.
If not found, say: "I couldn't find that in the document.\""""


        
        # Prompt
        prompt = f"""{system_instruction}

Conversation History:
{history_text}

Context:
{context}

Question:
{query}"""

        response = llm.invoke(prompt)

        # Extract response text
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        if not answer or answer.strip() == "":
            return "Unable to generate a response. Please try again."

        # Save memory
        st.session_state.chat_history.append((query, answer))

        return answer

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to generate response: {e}")
        print(f"[TRACEBACK] {error_details}")
        return f"Error generating response: {str(e)}"
