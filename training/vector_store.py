
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from utils.get_connection import GoogleApi
# import pickle
import os
import warnings

warnings.filterwarnings("ignore")

GoogleApi.create_connection()


class StoreModel():
    def __init__(self,Data_path:str,metadata:list,chunk_size:int,chunk_overlap:int):
        self.PATH = Data_path
        self.metadata = metadata
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    # Load Data
    def load_data(self):
        data_loader = DirectoryLoader(self.PATH, show_progress=True).load()
        if self.metadata and len(data_loader) == len(self.metadata):
            for out,meta in zip(data_loader,metadata):
                out.metadata = meta
            return data_loader
        elif self.metadata:
            raise ValueError("Number of documents does not match number of metadata entries.")
        return data_loader
    
    # split documents to fit token limit 
    def split_docs(self):
        loaded_data = self.load_data()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(loaded_data)
        return docs
    
    def load_embedding_model(self):
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        return gemini_embeddings
    
    def create_vectors(self,MODEL_PATH):
        gemini_embeddings = self.load_embedding_model()
        documents = self.split_docs()
        vector_store = Chroma.from_documents(documents, gemini_embeddings, persist_directory=f"{MODEL_PATH}/chroma_db")
        return vector_store
    
    # def store_index(self,MODEL_PATH):
    #     model_name = os.path.join(MODEL_PATH,"parser_model10")
    #     vector_store = self.create_vectors()

    #     with open(f'{model_name}.pkl', 'wb') as f:
    #         pickle.dump(vector_store, f)

if __name__=="__main__":
    metadata = [{"source": "Github repositries and github projects along with each projects create date"},
                    {"source": "Personal information, education, work details and certifications"},
                    {"source": "resume related deatils"}]
    st_obj = StoreModel(Data_path = 'Data',metadata = metadata,chunk_size = 1500,chunk_overlap = 128)
    st_obj.create_vectors(MODEL_PATH = 'Models')