from dotenv import load_dotenv
import google.generativeai as genai
import os
import streamlit as st

load_dotenv()

class GoogleApi:

    def create_connection():
        api_key = st.secrets["GOOGLE_API_KEY"]
        try:
            genai.configure(api_key=api_key)
            print("Connection established")
        except Exception as e:
            print(f"Conection Error : {e}")

if __name__=="__main__":
    GoogleApi.create_connection()