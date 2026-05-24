import streamlit as st
from Backend_tools import chat, get_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import asyncio


# **************************************** utility functions *************************

# LangGraph thread ids should preferably be strings.
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id


# create a new thread id
# reset the conversation history
# save it in session
# update the msg history
def reset_chat():
    thread_id = generate_thread_id()

    st.session_state['thread_id'] = thread_id

    add_thread(st.session_state['thread_id'])

    st.session_state['message_history'] = []


def add_thread(thread_id):

    if thread_id not in st.session_state['chat_threads']:

        st.session_state['chat_threads'].append(thread_id)


# loading the convo for a particular thread id
def load_conversation(thread_id):

    state = chat.get_state(
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    )

    # Check if messages key exists in state values
    return state.values.get('messages', [])


# ******************************* ASYNC RESPONSE FUNCTION *******************************

def get_ai_response(user_input):

    response = asyncio.run(
        chat.ainvoke(
            {
                'messages': [HumanMessage(content=user_input)]
            },
            config=CONFIG
        )
    )

    ai_message = response['messages'][-1].content

    return ai_message


# **************************************** Session Setup ******************************

# storing the conversation history in session state
if 'message_history' not in st.session_state:

    st.session_state['message_history'] = []


if 'thread_id' not in st.session_state:

    st.session_state['thread_id'] = generate_thread_id()


# chat_threads contains all thread ids
if 'chat_threads' not in st.session_state:

    st.session_state['chat_threads'] = get_all_threads()


add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title("Chatbot")

st.sidebar.button(
    'New Chat',
    on_click=reset_chat
)

st.sidebar.header('My Conversations')


for thread_id in st.session_state['chat_threads'][::-1]:

    if st.sidebar.button(thread_id):

        st.session_state['thread_id'] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                role = 'user'

            elif isinstance(msg, AIMessage):

                role = 'assistant'

            else:
                continue

            temp_messages.append(
                {
                    'role': role,
                    'content': msg.content
                }
            )

        st.session_state['message_history'] = temp_messages


CONFIG = {
    'configurable': {
        'thread_id': st.session_state["thread_id"]
    },
    'run_name': 'chat-run'
}


# **************************************** MAIN UI *********************************

# display the conversation history
for message in st.session_state['message_history']:

    with st.chat_message(message['role']):

        st.text(message['content'])


user_input = st.chat_input("Type here")


# **************************************** USER INPUT *********************************

if user_input:

    # save user message
    st.session_state['message_history'].append(
        {
            'role': 'user',
            'content': user_input
        }
    )

    # display user message
    with st.chat_message('user'):

        st.text(user_input)

    # generate AI response
    with st.chat_message('assistant'):

        with st.spinner("Thinking..."):

            ai_message = get_ai_response(user_input)

            st.write(ai_message)

    # save assistant message
    st.session_state['message_history'].append(
        {
            'role': 'assistant',
            'content': ai_message
        }
    )