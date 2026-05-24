from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated, Any, Dict, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from langchain_core.messages import SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import tempfile


load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free"
)

#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# for each thread id you can upload doc which would be only specific to that thread id
# Stores retriever objects for each chat thread
# Format:
# {
#    "thread-id": retriever_object
# }
# This helps each conversation use its own uploaded PDF/vector database
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


#Function for RAG
#1. Function receives:
 #  - PDF bytes
  # - thread_id
 #  - optional filename

#2. We create a temporary real PDF file from those bytes

#3. Store the temporary path in temp_path

#4. Pass that path to PyPDFLoader
def retreiver_doc(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # load the document
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        #split the documnts
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        #performing embedding and storing in a vectorstore
        vector_store = FAISS.from_documents(chunks, embeddings)

        #retrieving relevant documents based on a query
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}  
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass      

#tools
#1st tool
search_tool = DuckDuckGoSearchRun(region="us-en")
#second tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div

    Perform arithmetic operations.

    Examples:
    - add → addition
    - sub → subtraction
    - mul → multiplication
    - div → division
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)
        }
#3rd tool
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"    
    response = requests.get(url)
    return response.json()

#adding RAG tool
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for a thread.

    If no retriever is found for the thread, returns an error message.
    """
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


#adding all the tools in a list
tools = [search_tool, calculator, get_stock_price, rag_tool]
#making llm aware of the available tools
llm_with_tools = llm.bind_tools(tools)

#define a state for the graph of typeDict
class ChatState(TypedDict):
    #creating a var message that is a list of BaseMessage objects(user,si,system msg) and using a reducer to combine the messages into a single message
    messages: Annotated[list[BaseMessage], add_messages]

#creating a graph using stategraph and passing the staetype as a parameter
graph = StateGraph(ChatState)

def chatbot(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

#db connection for persistence using sqlite
CONN = sqlite3.connect(database="chat_history.db", check_same_thread=False)

#adding checkpointer to save the state of the graph in memory : Persistance
memory = SqliteSaver(conn=CONN)
#creating the node
tool_node = ToolNode(tools)
graph.add_node('chatbot',chatbot)
graph.add_node('tools', tool_node)
#creating the edges
graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot',tools_condition)
#creating a loop
graph.add_edge('tools', 'chatbot')

#compiling the graph
chat = graph.compile(checkpointer=memory)

#get all the threads currently stored in the database
# get all thread ids
def get_all_threads():

    all_threads = set()

    for checkpoint in memory.list(None):

        thread_id = checkpoint.config[
            'configurable'
        ]['thread_id']

        all_threads.add(thread_id)

    return list(all_threads)
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})