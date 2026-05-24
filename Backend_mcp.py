from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import TypedDict, Annotated
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
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import sys

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free"
)
async def setup_mcp():

    client = MultiServerMCPClient(
        {
            "expense_tracker": {
                "command": sys.executable,
                "args": [
                    "/Users/priyamvadasingh/Desktop/Expense-tracker/main.py"
                ],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    return tools

 
mcp_tools = asyncio.run(setup_mcp())
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

#adding all the tools in a list
tools = [search_tool, calculator, get_stock_price, *mcp_tools]
#making llm aware of the available tools
llm_with_tools = llm.bind_tools(tools)

#define a state for the graph of typeDict

class ChatState(TypedDict):
    #creating a var message that is a list of BaseMessage objects(user,si,system msg) and using a reducer to combine the messages into a single message
    messages: Annotated[list[BaseMessage], add_messages]

#creating a graph using stategraph and passing the staetype as a parameter

graph = StateGraph(ChatState)
#making it async for mcp integration
async def chatbot(state: ChatState):
    system_prompt = SystemMessage(
    content="""
You are a helpful AI assistant.

You MUST use available tools for:
- real-time information
- stock prices
- calculations
- internet searches
- expense tracking
- expense summaries
- updating expenses

Do not answer from memory when tools are available.
"""
)
    messages = [system_prompt] + state['messages']
    response = await llm_with_tools.ainvoke(messages)
    print(response)
    print("TOOL CALLS:", response.tool_calls)
    #adding the response to the message list and returning the updated message list as a dictionary with the key "message"
    return {"messages":[response]}

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