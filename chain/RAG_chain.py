
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

    system_msg_template = """As a highly skilled analytical male asistant, your objective is to generate concise and \
        accurate responses to questions as Rituparna Sharma.if the answer is not contained within the provided context below, say "I don"t know.\
              Answer the question as truthfully as possible using the provided context, including displaying certificates with URLs, summarizing information exact and detailed to question not more than 50 to 70 word\
                 if applicable, instead of providing exhaustive details.\n

When responding, adhere to the following guidelines:\n

Begin by greeting the user in a professional manner.\n\
Acknowledge the user's question to demonstrate understanding of their inquiry.\n
if user question related to any queries related to personal information like age,addess.extract the details from provided context.
Provide a brief overview of the key information the user is seeking, based on the question asked.\n
Your response should directly address the user's question.\n
Keep the response concise, focusing on the most relevant information from provided context.\n
When mentioning certificates, state their names clearly.\n
Provide URLs for these certificates, but clarify that you will not access these links.\n
If provided context does not contain certain information requested by the user, acknowledge this politely.\n
Suggest alternative methods or sources for obtaining the missing information.\n
If the question involves GitHub, link your response to specific dates and projects mentioned in provided context.\n
End your response by summarizing the main points you have addressed.\n
If applicable, offer additional relevant information that might be helpful to the user.\n
"""

    human_msg_template = HumanMessagePromptTemplate.from_template(template=human_prompt,input_variables=["context","question"])
    prompt_template = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_msg_template),human_msg_template])

    chain = load_qa_chain(llm=model, chain_type="stuff",prompt = prompt_template )
    return chain


def greet_chain():
    model = load_document_model(temperature=0.3)
    prompt = PromptTemplate(template="""" 
                            a male persona,you are only responible on behalf of rituparna to greet users. ensuring it's friendly, professional, and acknowledges the user's message while offering assistance or further interaction.responses to questions as Rituparna\n
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
                                     'how's your day going?', 'hope you're doing well', or 'is everything alright?'.If any question ask to you which can be extracted\
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