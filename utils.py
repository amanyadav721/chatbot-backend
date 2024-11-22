import os
import logging
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vectordata import retriever


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



def get_vectorstore():
    """Create a FAISS vector store from text chunks."""
    try:
        vectorstore = retriever()
        logging.info("Pinceone vector store created successfully.")
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
            retriever=vectorstore,
            memory=memory
        )
        logging.info("Conversational retrieval chain initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing conversational retrieval chain: {e}")
        raise e
    return conversation_chain


def chat_bot(user_prompt):
    try:
        prompt=f"""You are a health coach based on andrew huberman who will help you understand how to optimize your health,
        Talk like you are a health expert level health coach,
        For you help you have provided with andrew huberman knowlege base
        You need to answers question
        Here is user's question:{user_prompt}
        """
        vectorstore = get_vectorstore()
        conversation_chain = get_conversation_chain(vectorstore)
        query = prompt
        response = conversation_chain.run(query)
        return response
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")




