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

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free"
)
#define a state for the graph of typeDict
class ChatState(TypedDict):
    #creating a var message that is a list of BaseMessage objects(user,si,system msg) and using a reducer to combine the messages into a single message
    messages: Annotated[list[BaseMessage], add_messages]

#creating a graph using stategraph and passing the staetype as a parameter
graph = StateGraph(ChatState)

def chatbot(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    #adding the response to the message list and returning the updated message list as a dictionary with the key "message"
    return {"messages":[response]}

#db connection for persistence using sqlite
CONN = sqlite3.connect(database="chat_history.db", check_same_thread=False)

#adding checkpointer to save the state of the graph in memory : Persistance
memory = SqliteSaver(conn=CONN)
#creating the node
graph.add_node('chatbot',chatbot)
#creating the edges
graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot',END)

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