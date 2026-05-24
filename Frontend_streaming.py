import streamlit as st
from LangGraph_backend import chat
from langchain_core.messages import HumanMessage
 #LG internally expects dict with special key called configurable. using it you assign key -alue for thread-id to store the conversation history in the session state of streamlit. This allows us to maintain the conversation history across different interactions with the chatbot.
CONFIG = {'configurable': {'thread_id': 'thread_1'}}
 #storing the conversation history in session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


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
    