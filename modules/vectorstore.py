# modules/vectorstore.py

import os

from config import EMBEDDING_MODEL, DB_DIR

try:
    from langchain_community.vectorstores import Chroma, FAISS
except ImportError:
    Chroma = None
    FAISS = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


def get_vectorstore(db_path):

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if HuggingFaceEmbeddings is not None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            print("[INFO] Using HuggingFace embeddings.")
        except Exception as ex:
            print(f"[WARNING] HuggingFaceEmbeddings init failed: {ex}")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print("[WARNING] Falling back to DummyEmbeddings. This will work but results may be low-quality.")

        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.0] * 1536 for _ in texts]

            def embed_query(self, text):
                return [0.0] * 1536

        embeddings = DummyEmbeddings()

    vectorstore = None

    if Chroma is not None:
        try:
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
            )
            print("[INFO] Chroma vectorstore initialized.")
        except Exception as ex:
            print(f"[WARNING] Chroma init failed: {ex}")
            vectorstore = None

    if vectorstore is None and FAISS is not None:
        try:
            vectorstore = FAISS.from_texts([], embeddings)
            print("[INFO] FAISS vectorstore initialized (empty).")
        except Exception as ex:
            print(f"[WARNING] FAISS init failed: {ex}")
            vectorstore = None

    if vectorstore is None:
        print("[WARNING] Chroma/FAISS not available; using InMemoryVectorStore fallback.")

        class InMemoryRetriever:
            def __init__(self, store, k=4):
                self.store = store
                self.k = k

            def get_relevant_documents(self, query):
                if not self.store.docs:
                    return []
                
                # If we have vectors, use them
                if self.store.vectors:
                    q_vec = self.store.embed_query(query)

                    def cos_sim(v1, v2):
                        dot = sum(x * y for x, y in zip(v1, v2))
                        norm1 = sum(x * x for x in v1) ** 0.5
                        norm2 = sum(x * x for x in v2) ** 0.5
                        return dot / (norm1 * norm2 + 1e-12)

                    scores = [cos_sim(q_vec, v) for v in self.store.vectors]
                    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    top = ranked[:self.k]
                    return [self.store.docs[i] for i in top if i < len(self.store.docs)]
                else:
                    # Fallback: return all docs if no vectors
                    return self.store.docs[:self.k]

            def invoke(self, query):
                """LangChain compatibility method"""
                return self.get_relevant_documents(query)

            def as_retriever(self, search_kwargs=None):
                return self

        class InMemoryVectorStore:
            def __init__(self, embedding_function):
                self.embedding_function = embedding_function
                self.docs = []
                self.vectors = []

            def embed_query(self, query):
                if hasattr(self.embedding_function, 'embed_query'):
                    return self.embedding_function.embed_query(query)
                if hasattr(self.embedding_function, 'embed_documents'):
                    # approximate with same method; this may be slow
                    return self.embedding_function.embed_documents([query])[0]
                raise AttributeError("Embedding backend does not support embed_query")

            def embed_documents(self, texts):
                if hasattr(self.embedding_function, 'embed_documents'):
                    return self.embedding_function.embed_documents(texts)
                raise AttributeError("Embedding backend does not support embed_documents")

            def add_documents(self, docs):
                if not docs:
                    return

                texts = [getattr(d, 'page_content', str(d)) for d in docs]
                try:
                    new_vectors = self.embed_documents(texts)
                    self.vectors.extend(new_vectors)
                    self.docs.extend(docs)
                except Exception as e:
                    print(f"[WARNING] Error embedding documents: {e}")
                    # Fall back: store without vectors
                    self.docs.extend(docs)

            def persist(self):
                pass

            def as_retriever(self, search_kwargs=None):
                k = (search_kwargs or {}).get('k', 4)
                return InMemoryRetriever(self, k=k)

            def similarity_search(self, query, k=4):
                return self.as_retriever({'k': k}).get_relevant_documents(query)

        vectorstore = InMemoryVectorStore(embeddings)

    print("[INFO] Vector DB Ready.")
    return vectorstore


def add_documents_to_db(vectorstore, chunks):

    if not chunks:
        return

    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print("[INFO] Documents stored successfully.")