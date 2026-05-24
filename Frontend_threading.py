import streamlit as st
from LangGraph_backend import chat, get_all_threads
from langchain_core.messages import HumanMessage,AIMessage
import uuid


# **************************************** utility functions *************************
#LangGraph thread ids should preferably be strings.
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id
#crete a new thread id
#reset the conversation history
#save it in session
#update the mgs history
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
#loading the convo for a paticular thread id
def load_conversation(thread_id):
    state = chat.get_state(
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    )

 # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])
# **************************************** Session Setup ******************************
 #LG internally expects dict with special key called configurable. using it you assign key -alue for thread-id to store the conversation history in the session state of streamlit. This allows us to maintain the conversation history across different interactions with the chatbot.
 #storing the conversation history in session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
#chat_thread is the name of the list which conatin the list of all thread id.
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()
add_thread(st.session_state['thread_id'])   
    

# **************************************** Sidebar UI *********************************
st.sidebar.title("Chatbot")
st.sidebar.button('New Chat', on_click=reset_chat)

#st.sidebar.button('Clear Conversations')
st.sidebar.header('My Conversations')
for  thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]},
            'run_name' : 'chat-run'
          }

# **************************************** MAIN UI *********************************
#display the converstaion history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
        
user_input = st.chat_input("Type here")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    #invoking the llm for displaying the llm response in the chat interface
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
         message_chunk.content for message_chunk, metadata  in chat.stream(
        {'messages': [HumanMessage(content=user_input)]},
         config= CONFIG,
         stream_mode = 'messages'
    )
    )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    