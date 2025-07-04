
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import StreamlitCallbackHandler
from utils.gen_models import load_document_model,load_question_generation_model
from utils.common_utils import load_vectors

def question_genrator_chain():
    model = load_question_generation_model()
    human_prompt = """
        "question: {question}
    """
    contextualize_q_system_prompt =contextualize_q_system_prompt = """Given a chat history and the latest user question \
 instructs the AI to reformulate a user's latest question into a standalone question.\
    This reformulated question should be clear and understandable without needing the context of the previous chat history. \
        It should also be relevant for retrieving answers from a knowledge base. The AI should not answer the question but only reformulate it.\
              If the user's question contains any words matching those in the chat history, include them in the reformulated question; otherwise, \
                keep the question as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", human_prompt),
        ]
    )
    chain = LLMChain(llm=model, prompt=contextualize_q_prompt)
    return chain

def document_chain():
    model = load_document_model(temperature=0.0)
    human_prompt = """
        Resume Details:\n {context}\n
        Aspect to Highlight: \n{question}\n

        Narrative:
    """

    system_msg_template =  """You are Rituparna Sharma, a highly skilled and analytical male assistant with advanced capabilities in resume parsing. Your primary objective is to generate concise, accurate, and professional responses using only the provided context.

**Workflow and Guidelines:**

1.  **Greeting:** Begin every response with a professional greeting.
2.  **Persona Consistency:** You must strictly maintain your assigned persona as Rituparna Sharma, a **male** assistant. All self-references and descriptions must use male pronouns (he, him, his). **Never use female pronouns (she, her, hers).**
3.  **Acknowledge the Query:** Briefly state your understanding of the user's inquiry.
4.  **Strict Context Adherence:** Formulate all answers using *only* the provided context. If the required information is not available, you must respond with "I don't know."
5.  **Conciseness:** Your entire response must be between 50 and 70 words.
6.  **Directness:** Answer the question as truthfully and directly as possible. Do not include summaries like "Main point:" or restate the answer unless explicitly asked. Avoid unnecessary elaboration.
7.  **Handling Personal Information:** If asked for personal details like age or address, extract them exactly as they appear in the context.
8.  **Certificates:** When mentioning certificates, you must:
    * State their exact names.
    * Provide their full URLs.
    * Clarify that you will not access the links.
9.  **GitHub Details:** If the question involves GitHub, your response must link to the specific project names and dates mentioned in the provided context.
10. **Conclusion:** End your response by briefly summarizing the main points you have addressed. You may also offer additional relevant information if it is helpful to the user.
11. **Formatting:** Use Markdown for richer presentation (like bullet points) only when it is appropriate and enhances clarity.
"""

    human_msg_template = HumanMessagePromptTemplate.from_template(template=human_prompt,input_variables=["context","question"])
    prompt_template = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_msg_template),human_msg_template])

    chain = load_qa_chain(llm=model, chain_type="stuff",prompt = prompt_template )
    return chain


def greet_chain():
    model = load_document_model(temperature=0.3)
    prompt = PromptTemplate(template="""" 
                            You are a male virtual assistant who speaks on behalf of Rituparna.\n
                            You greet users warmly and professionally, acknowledge their input naturally, and invite them to continue the conversation or ask questions.Avoid robotic phrases like “I understand you are asking about...”. Use varied, human-friendly responses instead.\n
                             Here's the input: {question}"
                            
""", input_variables=["question"],output_variables=["answer"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def selection_chain():
    model = load_document_model(temperature=0.0)
    default_prompt = PromptTemplate(template="""you are a virtual assistant with advanced analysis capabilities, your task is to accurately analyze user inputs\
                                    and determine its nature. If the input is a query related rituparna/ritu career-related inquiry, involving topics such as resumes,\
                                     LinkedIn, GitHub, or other professional matters respond with 'no'. If the input is\
                                     a greeting, either direct or indirect, respond with 'yes'. Direct greetings include phrases like 'hello',\
                                     'hi', 'good morning', 'good afternoon', 'good evening', 'hey', and 'greetings'. Indirect greetings are subtler,\
                                     often in the form of questions or statements like 'how are you?', 'how's it going?', 'what's up?',\
                                     'how have you been?', 'having fun?', 'are you okay?', 'it's nice to see you', 'long time no see',\
                                     'how's your day going?', 'hope you're doing well', 'is everything alright?'. If the user input is part of a general conversation or an expression of emotions \
                                    (e.g., "I am sad", "I'm feeling happy today", etc.)  respond with 'yes' .If any question ask to you which can be extracted\
                                    from provided context,answer with "no"

Here's the user's input: {question}"""

, input_variables=["question"],output_key="answer")
    default_chain = LLMChain(llm=model,prompt=default_prompt,llm_kwargs = {"max_token":50})
    return default_chain

def conversation_join_chain(memory):
    st_callback = StreamlitCallbackHandler(st.container())
    loaded_vector_store = load_vectors()
    qa_chain = ConversationalRetrievalChain(
        combine_docs_chain=document_chain(),
        retriever=loaded_vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 8}),  
        memory=memory,
        callbacks=[st_callback],
        question_generator = question_genrator_chain(),
        get_chat_history=lambda chat : chat,
        rephrase_question = True,
        response_if_no_docs_found = "No relavent document found related to question"
    )
    return qa_chain
