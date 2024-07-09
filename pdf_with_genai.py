import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Configure generative AI with Google API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))  # Use io.BytesIO to read bytes-like object
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_spliiter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    chunks=text_spliiter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')




def get_conversational_chain():
    prompt_template= '''
    Anser the question as deatailed as possible from the provided context,make sure to provide all the details,if the answer is not in the
    provided context just say,'answer is not available in the context',don't provide the wrong answer\n
    context:\n {context}?\n
    Question:\n{question}\n 
    
    Answer:
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        # Pass the documents and question to the chain and get the response
        response = chain({
            'input_documents': docs, 
            'question': user_question
        })
        
        # Extract the output from the response
        answer = response['output_text']
        
        # Print and display the response
        print(answer)
        st.write('Reply:', answer)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        st.write(f"An error occurred: {e}")


def main():
    st.set_page_config('Chat with MUltiple PDF')
    st.header('Chat with Multiple PDF using GeminiAI')
    
    user_question = st.text_input('Ask a Question fromthe PDF Files')
    
    if user_question:
        user_input(user_question)
        
    with  st.sidebar:
        st.title("menu")
        pdf_docs = st.file_uploader('Upload your PDF Files and Click on the Submit & Process', type=['pdf'], accept_multiple_files=True)
        if st.button('Submit & Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success('Done')


if __name__ == '__main__':
    main()