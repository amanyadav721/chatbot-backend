import os
import logging
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_token = os.getenv("GEMINI_KEY")

if not groq_api_key and google_api_token:
    logging.error("GROQ_API_KEY, GEMINI_KEY,sarvamai_api_key must be set in the .env file")
    raise ValueError("GROQ_API_KEY, GEMINI_KEY,must be set in the .env file")

os.environ["GROQ_API_KEY"] = groq_api_key

try:
    model = ChatGroq(temperature=0.6, model_name="llama-3.1-70b-versatile")
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_token)
    llm = model
    logging.info("Language models initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize language models: {e}")
    raise e

pdf_path = os.path.join(os.path.dirname(__file__), 'dataset', 'data.pdf')


def get_pdf_text(pdf_path):
    """Extract text from a single PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        logging.info("PDF text extraction successful.")
        return text  # Add this line to return the extracted text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise e
    

def get_text_chunks(text):
    """Split text into manageable chunks."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        raise e
    return chunks

def get_vectorstore(text_chunks):
    """Create a FAISS vector store from text chunks."""
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=emb)
        logging.info("FAISS vector store created successfully.")
    except Exception as e:
        logging.error(f"Error creating FAISS vector store: {e}")
        raise e
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initialize the conversational retrieval chain with memory."""
    try:
        memory =  ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        logging.info("Conversational retrieval chain initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing conversational retrieval chain: {e}")
        raise e
    return conversation_chain

def chat_bot(prompt):
    try:
        text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(text)
        
        vectorstore = get_vectorstore(text_chunks)
        
        conversation_chain = get_conversation_chain(vectorstore)
    
        query = prompt
        response = conversation_chain.run(query)
        return response
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


