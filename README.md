## Knowledge Management Chatbot </br>
This project builds a custom chatbot that can process and analyze user-provided documents (e.g., PDFs) and respond to queries based on their content. It utilizes LangChain for handling language models, Llama 3 for intelligent response generation, IBM WatsonX for cloud computing, and Flask for the backend.

### Features </br>
Custom Data Integration: Upload a document (PDF) for analysis, enabling the chatbot to respond to questions about the document's content.</br>
Conversational Interface: Engage with the chatbot in a natural, question-answer style format.</br>
LangChain Integration: Leverages LangChain to manage LLM-based processing and embeddings for efficient text retrieval.</br>
Flask Backend: Manages routes and connects the chatbot's logic to a user-friendly interface.</br>
Vector-Based Retrieval: Implements Chroma for similarity searches, enabling precise document-based question answering.</br>

### Technologies Used </br>
Python: Core programming language.</br>
Flask: Web framework for backend development.</br>
LangChain: Framework for LLM-based applications.</br>
Llama 3: For embeddings processing and intelligent response generation.</br>
IBM WatsonX: For running large language models like Llama.</br>
Chroma: Vector store for efficient document retrieval.</br>
PyPDFLoader: For loading and splitting PDF files.</br>
JavaScript: To enable front-end interactivity.</br>

### Prerequisites </br>
Python: Core programming language.</br>
Flask: Web framework for backend development.</br>
LangChain: Framework for LLM-based applications.</br>
IBM WatsonX: For running large language models like LLAMA2.</br>
Chroma: Vector store for efficient document retrieval.</br>
PyPDFLoader: For loading and splitting PDF files.</br>
JavaScript: To enable front-end interactivity.</br>


## Set Up Environment

### Install required dependencies:
pip install -r requirements.txt


### Install FFmpeg

Linux: </br>
sudo apt update</br>
sudo apt install ffmpeg -y

macOS: </br>
brew install ffmpeg

Windows: </br>
Download from FFmpeg official website.</br>
Add the bin directory to your system's PATH.</br>

### Set Up Credentials</br>
Obtain IBM WatsonX credentials for input in 'worker.py' file </br>

## Running the Application </br>
Start the server with:</br>
python server.py</br>

Access the web interface by launching application.</br>


## Key Functions
### worker.py </br>
init_llm(): Initializes the LLM via WatsonX and embeddings for document processing.</br>
process_document(document_path): Loads and splits a PDF into manageable chunks, then creates a vector store for retrieval.</br>
process_prompt(prompt): Processes user queries and retrieves relevant answers from the document.</br>

### server.py </br>
Handles Flask routes for:</br>
GET /: Loads the main interface.</br>
POST /process-message: Processes user messages.</br>
POST /process-document: Processes uploaded documents.</br>



