from chain.RAG_chain import greet_chain,selection_chain,conversation_join_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
import time
import google
import subprocess
import sys


# Function to set the stop flag
def stop():
    st.session_state.stop_pressed = True
def deploy():
    subprocess.run([f"{sys.executable}",  "training/vector_store.py"])
if __name__ == "__main__" :
    st.button("deploy", on_click=deploy)


    # JavaScript to add a unique class to the 'Stop' button
    st.markdown("""
        <script>
        const addButtonClass = () => {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(btn => {
                if (btn.innerText === 'Stop') {
                    btn.classList.add('stop-button');
                }
            });
        }
        addButtonClass();
        </script>
    """, unsafe_allow_html=True)

    # Custom CSS for the 'stop-button' class
    st.markdown("""
        <style>
        .stop-button {
            width: 100%;
            color: white;
            background-color: #FF4136; /* Red background for stop button */
            border: 2px solid #FF4136; /* Red border for stop button */
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    # Title with custom class for styling
    st.markdown("<h1 class='chat-title'>SmartPal</h1>", unsafe_allow_html=True)


    if 'CHAIN_MEMORY' not in st.session_state:
        st.session_state.CHAIN_MEMORY=ConversationBufferWindowMemory(k=5,return_messages=True,memory_key="chat_history")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'stop_pressed' not in st.session_state:
        st.session_state.stop_pressed = False
    
    qa_chain = conversation_join_chain(memory = st.session_state.CHAIN_MEMORY)
    selection_chain_ = selection_chain()
    greet_chain_ = greet_chain()

    question = st.chat_input("ask me about my profession")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.button("Stop", on_click=stop)
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner('Please wait...'):  # Start of spinner context
            result = {}
            try:
                selection_chain_result = selection_chain_.invoke({"question":question})
                print(selection_chain_result)
                if selection_chain_result['text'] == "yes":
                    result = greet_chain_.invoke({"question":question})
                elif selection_chain_result['text'] == "no":
                    result =qa_chain.invoke({"question":question})
                else:
                    result["answer"] = "Certainly! I apologize if your previous question did not align with Rituparna's interests. Could I kindly request more details to tailor my question more appropriately?"
            except google.generativeai.types.generation_types.BlockedPromptException as e:
                result['answer'] = "please ask your question properly"
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            print(result)
            try:
                split_res = result['answer'].split()
            except:
                split_res = result['text'].split()
            for chunk in split_res:
                # Continue with the typing effect
                full_response += chunk + " "
                time.sleep(0.11)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.stop_pressed = False