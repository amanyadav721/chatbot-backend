import os
import logging
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

index_name = "chatbot"



load_dotenv()
logging.basicConfig(level=logging.INFO)
load_dotenv()

#intializing pinecone client
api_key=os.getenv("PINECONE_API_KEY")
google_api_token = os.getenv("GEMINI_KEY")
pc = Pinecone(api_key=api_key)

try:
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_token)
    logging.info("Language models initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize language models: {e}")
    raise e

pdf_path = os.path.join(os.path.dirname(__file__), 'dataset', 'data.pdf')



#creating the index 
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )

index=pc.Index(index_name)





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

#sparse matrix
def sparsematrix():
     bm25_encoder = BM25Encoder().load('bm25_values.json')
     logging.info(f"bm25_encoder created sparse matrix")
     return bm25_encoder

def retriever():
     bm25_encoder=sparsematrix()
     retriever = PineconeHybridSearchRetriever(embeddings=emb,sparse_encoder=bm25_encoder,index=index)
     return retriever
   



