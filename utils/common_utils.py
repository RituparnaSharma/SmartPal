
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

def load_vectors():
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    loaded_vector_store = Chroma(persist_directory="models/chroma_db", embedding_function=gemini_embeddings)
    return loaded_vector_store