
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.get_connection import GoogleApi
import warnings

warnings.filterwarnings("ignore")
GoogleApi.create_connection()


def load_question_generation_model():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash" ,  
                                    temperature=0.7,
                                    max_tokens=256,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0,
                                    convert_system_message_to_human=True)
    return model

def load_document_model(temperature:float):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                            temperature=temperature,
                            convert_system_message_to_human=True)
    return model


