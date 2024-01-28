
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
# import pickle
def load_vectors():
    #     # To load the vector store later
    # with open('models/modeldb.pkl', 'rb') as f:
    #     loaded_vector_store = pickle.load(f)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    loaded_vector_store = Chroma(persist_directory="models/chroma_db", embedding_function=gemini_embeddings)
    return loaded_vector_store