"""
This script contains the core logic for the chatbot's backend.
It includes:
1. Initialization of the Llama LLM and embeddings using IBM WatsonX and HuggingFace.
2. Processing of uploaded PDF documents to generate embeddings and prepare them for question answering.
3. Handling user queries and retrieving responses based on processed documents.
"""

import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables for model, embeddings, and chat history
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

def init_llm():
    """
    Initialize the language model (LLM) and embeddings for document processing.
    """
    global llm_hub, embeddings

    # Define IBM WatsonX credentials
    my_credentials = {
        #Enter credentials here
    }

    # Model parameters for WatsonX
    params = {
        GenParams.MAX_NEW_TOKENS: 256,  # Maximum number of tokens generated in a single run
        GenParams.TEMPERATURE: 0.1,    # Controls randomness: lower value = more deterministic
    }

    # Initialize the WatsonX model
    LLAMA2_model = Model(
        model_id='meta-llama/llama-3-8b-instruct',
        credentials=my_credentials,
        params=params,
        project_id= #Enter project ID here
    )

    llm_hub = WatsonxLLM(model=LLAMA2_model)  # Create the LLM instance

    # Initialize embeddings using HuggingFace's pre-trained model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

def process_document(document_path):
    """
    Process a PDF document to prepare for question-answering.
    Steps:
        1. Load the document.
        2. Split text into chunks.
        3. Generate embeddings.
        4. Set up a conversational retrieval chain.
    """
    global conversation_retrieval_chain

    loader = PyPDFLoader(document_path)  # Load the document
    documents = loader.load()

    # Split the document into manageable text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings and create a vector store
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Build the conversational retrieval chain
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )

def process_prompt(prompt):
    """
    Handle user queries and retrieve responses based on processed documents.
    Args:
        prompt (str): User's question.
    Returns:
        str: Model's response.
    """
    global conversation_retrieval_chain, chat_history

    # Query the LLM with the user's prompt and retrieve the response
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]

    # Update chat history
    chat_history.append((prompt, answer))

    return answer

# Initialize the LLM and embeddings
init_llm()
