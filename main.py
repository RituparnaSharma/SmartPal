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


def has_special_formatting(text):
    """Check if text contains bullet points, numbered lists, or multiple paragraphs"""
    formatting_indicators = [
        '•', '*', '-', '1.', '2.', '3.', '\n\n', 
        '**', '__', '#', '###', '>', '|'
    ]
    return any(indicator in text for indicator in formatting_indicators)


def render_formatted_response(placeholder, text):
    """Render formatted text with better visual effects"""
    import re
    
    # Split text into logical sections
    sections = re.split(r'(\n\n|\n(?=\d+\.|\n(?=•|\*|-)))', text)
    full_content = ""
    
    for i, section in enumerate(sections):
        if st.session_state.stop_pressed:
            full_content += "\n\n[Response stopped by user]"
            break
            
        section = section.strip()
        if not section:
            continue
            
        full_content += section + "\n\n" if i < len(sections) - 1 else section
        
        # Add animated content wrapper for better visual appeal
        placeholder.markdown(f'<div class="animated-content">{full_content}</div>', 
                           unsafe_allow_html=True)
        time.sleep(0.3)  # Slower for formatted content
    
    # Final render without wrapper
    placeholder.markdown(full_content)
    return full_content


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
        /* Enhanced markdown styling for better readability */
        .stMarkdown {
            line-height: 1.6;
        }
        .stMarkdown ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }
        .stMarkdown li {
            margin: 0.3rem 0;
            line-height: 1.5;
        }
        .stMarkdown p {
            margin: 0.8rem 0;
            line-height: 1.6;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            margin: 1rem 0 0.5rem 0;
            color: #4CAF50;
        }
        .stMarkdown strong {
            color: #2E7D32;
            font-weight: 600;
        }
        .stMarkdown blockquote {
            border-left: 4px solid #4CAF50;
            padding-left: 1rem;
            margin: 1rem 0;
            font-style: italic;
            background-color: #f8f9fa;
        }
        /* Animate bullet points appearing */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .animated-content {
            animation: fadeInUp 0.3s ease-out;
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
                    selection_chain_result = selection_chain_.invoke({"question": prompt})
                    
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
                    
                    # More flexible routing logic
                    if "yes" in selection_text:
                        result = greet_chain_.invoke({"question": prompt})
                    elif "no" in selection_text:
                        result = qa_chain.invoke({"question": prompt})
                    else:
                        result["answer"] = ("Certainly! I apologize if your previous question "
                                         "did not align with Rituparna's interests. Could I kindly "
                                         "request more details to tailor my question more appropriately?")
                        
                except google.generativeai.types.generation_types.BlockedPromptException as e:
                    result['answer'] = "Please ask your question properly"
                    print(f"Blocked prompt exception: {e}")
                except Exception as e:
                    result['answer'] = "I'm sorry, I encountered an error. Please try again."
                    print(f"Unexpected error: {e}")

            # Streaming response effect with better formatting
            full_response = ""
            try:
                response_text = result.get('answer', result.get('text', ''))
                
                # Handle different types of content formatting
                if has_special_formatting(response_text):
                    # For content with bullet points, paragraphs, etc.
                    full_response = render_formatted_response(message_placeholder, response_text)
                else:
                    # For simple text, use word-by-word streaming
                    split_res = response_text.split()
                    
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
                    
            except Exception as e:
                full_response = "Sorry, I couldn't process your request."
                message_placeholder.markdown(full_response)
                print(f"Error processing response: {e}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response.strip() if full_response else response_text
            })
            
        # Reset stop flag
        st.session_state.stop_pressed = False

    # Clear chat history button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.CHAIN_MEMORY.clear()
            st.rerun()
        
        # Display chat statistics
        st.markdown("---")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        st.markdown(f"**Memory Window:** {st.session_state.CHAIN_MEMORY.k} exchanges")
