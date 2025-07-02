from chain.RAG_chain import greet_chain, selection_chain, conversation_join_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
import time
import google
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import subprocess
import sys


# Function to set the stop flag
def stop():
    st.session_state.stop_pressed = True


# def deploy():
#     subprocess.run([f"{sys.executable}",  "training/vector_store.py"])

if __name__ == "__main__":
    # st.button("deploy", on_click=deploy)
    
    # Page configuration
    st.set_page_config(
        page_title="SmartPal",
        page_icon="🤖",
        layout="centered"
    )
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .chat-title {
            color: #4CAF50; 
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-box {
            background-color: #f2f2f2;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .stButton > button {
            width: 100%;
            color: white;
            background-color: #ff4b4b;
            border: 2px solid #ff4b4b;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #ff6b6b;
            border-color: #ff6b6b;
        }
        /* Hide the default chat input styling if needed */
        .stChatInput {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title with custom class for styling
    st.markdown("<h1 class='chat-title'>SmartPal</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'CHAIN_MEMORY' not in st.session_state:
        st.session_state.CHAIN_MEMORY = ConversationBufferWindowMemory(
            k=2, 
            return_messages=True, 
            memory_key="chat_history"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'stop_pressed' not in st.session_state:
        st.session_state.stop_pressed = False
        
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = True  # Set to False once you've debugged
    
    # Initialize chains
    qa_chain = conversation_join_chain(memory=st.session_state.CHAIN_MEMORY)
    selection_chain_ = selection_chain()
    greet_chain_ = greet_chain()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about my profession"):
        # Add stop button in sidebar when processing
        if not st.session_state.stop_pressed:
            with st.sidebar:
                st.button("🛑 Stop", on_click=stop, type="secondary")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner('Thinking...'):
                result = {}
                try:
                    # Debug: Show what we're sending to selection chain
                    if st.session_state.debug_mode:
                        st.write(f"DEBUG: Sending to selection chain: {prompt}")
                    
                    selection_chain_result = selection_chain_.invoke({"question": prompt})
                    
                    # Debug: Show selection chain result
                    if st.session_state.debug_mode:
                        st.write(f"DEBUG: Selection chain result: {selection_chain_result}")
                        st.write(f"DEBUG: Selection chain result type: {type(selection_chain_result)}")
                    
                    # Handle different possible result formats
                    selection_text = ""
                    if isinstance(selection_chain_result, dict):
                        selection_text = selection_chain_result.get('text', '').lower().strip()
                        if not selection_text:
                            # Try other possible keys
                            selection_text = selection_chain_result.get('output', '').lower().strip()
                            if not selection_text:
                                selection_text = str(selection_chain_result).lower().strip()
                    else:
                        selection_text = str(selection_chain_result).lower().strip()
                    
                    if st.session_state.debug_mode:
                        st.write(f"DEBUG: Processed selection text: '{selection_text}'")
                    
                    # More flexible routing logic
                    if "yes" in selection_text:
                        if st.session_state.debug_mode:
                            st.write("DEBUG: Routing to greet_chain")
                        result = greet_chain_.invoke({"question": prompt})
                        if st.session_state.debug_mode:
                            st.write(f"DEBUG: Greet chain result: {result}")
                    elif "no" in selection_text:
                        if st.session_state.debug_mode:
                            st.write("DEBUG: Routing to qa_chain")
                        result = qa_chain.invoke({"question": prompt})
                        if st.session_state.debug_mode:
                            st.write(f"DEBUG: QA chain result: {result}")
                    else:
                        if st.session_state.debug_mode:
                            st.write(f"DEBUG: No routing match found for: '{selection_text}'")
                        result["answer"] = ("Certainly! I apologize if your previous question "
                                         "did not align with Rituparna's interests. Could I kindly "
                                         "request more details to tailor my question more appropriately?")
                        
                except google.generativeai.types.generation_types.BlockedPromptException as e:
                    result['answer'] = "Please ask your question properly"
                    print(f"Blocked prompt exception: {e}")
                    st.error(f"Blocked prompt: {e}")
                except Exception as e:
                    result['answer'] = "I'm sorry, I encountered an error. Please try again."
                    print(f"Unexpected error: {e}")
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Streaming response effect
            full_response = ""
            try:
                response_text = result.get('answer', result.get('text', ''))
                split_res = response_text.split()
            except Exception as e:
                split_res = ["Sorry,", "I", "couldn't", "process", "your", "request."]
                print(f"Error processing response: {e}")
            
            # Check for stop condition during streaming
            for i, chunk in enumerate(split_res):
                if st.session_state.stop_pressed:
                    full_response += " [Response stopped by user]"
                    break
                    
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")  # Cursor effect
                time.sleep(0.05)  # Reduced sleep time for better UX
            
            # Final response without cursor
            message_placeholder.markdown(full_response.strip())
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response.strip()
            })
            
        # Reset stop flag
        st.session_state.stop_pressed = False

    # Clear chat history button in sidebar
    with st.sidebar:
        st.markdown("---")
        
        # Debug mode toggle
        debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.CHAIN_MEMORY.clear()
            st.rerun()
        
        # Display chat statistics
        st.markdown("---")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        st.markdown(f"**Memory Window:** {st.session_state.CHAIN_MEMORY.k} exchanges")
