import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from streamlit_chat import message


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
 
def get_conversation_chain(vectorstore):
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True)
    return qa_chain
   
def handle_userinput(user_question):
    response = st.session_state.qa({"query": user_question})
    return response["result"]

def main():
    load_dotenv()
    
    st.set_page_config(page_title = "Chat with multiple pdfs", page_icon=":books:")
    
    if "qa" not in st.session_state:
        st.session_state.qa = None

    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your document:")  
    if user_question:
        message(user_question, is_user = True)
        response = handle_userinput(user_question) 
        message(response)
        
    
    with st.sidebar:
        st.subheader("Your docs")
        pdf_docs = st.file_uploader("upload your pdfs here and click process", accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing"):
                #getting the pdf text from pdfs
                raw_text = get_pdf_text(pdf_docs)
    
                #getting the text chunks
                text_chunks = get_text_chunks(raw_text)

                #creating vector store // creating embeddings
                vectorstore = get_vectorestore(text_chunks)
                
                #conv chain
                st.session_state.qa = get_conversation_chain(vectorstore)
    
         
                
if __name__ == '__main__':
    main()