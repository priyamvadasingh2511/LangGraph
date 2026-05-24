import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from Backend_rag import (
    chat,
    get_all_threads,
    retreiver_doc,
    thread_document_metadata,
)


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

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}

add_thread(st.session_state['thread_id'])   

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# **************************************** Sidebar UI *********************************
st.sidebar.title("Chatbot")
if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

#st.sidebar.button('Clear Conversations')
st.sidebar.header('My Conversations')
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)
if uploaded_pdf:

    if uploaded_pdf.name not in thread_docs:

        with st.sidebar.status("Indexing PDF…", expanded=True):

            summary = retreiver_doc(
                uploaded_pdf.getvalue(),
                thread_id=st.session_state['thread_id'],
                filename=uploaded_pdf.name
            )

        thread_docs[uploaded_pdf.name] = summary


    else:
        st.sidebar.info(
            f"{uploaded_pdf.name} already indexed"
        )
doc_meta = thread_document_metadata(thread_key)

if doc_meta:
        st.sidebar.success(
            f"Using: {doc_meta['filename']} "
            f"({doc_meta['chunks']} chunks)"
        )


for thread_id in st.session_state['chat_threads'][::-1]:

    if st.sidebar.button(
        thread_id,
        key=f"thread-{thread_id}"
    ):
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

            temp_messages.append({
                'role': role,
                'content': msg.content
            })

        st.session_state['message_history'] = temp_messages
        st.session_state["ingested_docs"].setdefault(
            str(thread_id),
            {}
)
        st.rerun()

CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
}

# **************************************** MAIN UI *********************************

#display the converstaion history
for message in st.session_state['message_history']:

    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type here")


# *********************** STREAM FUNCTION ADDED ************************
def stream_response(user_input):

    status_holder = {"box": None}

    for message_chunk, metadata in chat.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode='messages'
    ):

        # TOOL MESSAGE
        if isinstance(message_chunk, ToolMessage):

            tool_name = getattr(message_chunk, "name", "tool")

            if status_holder["box"] is None:

                status_holder["box"] = st.status(
                    f"🔧 Using `{tool_name}` …",
                    expanded=True
                )

            else:

                status_holder["box"].update(
                    label=f"🔧 Using `{tool_name}` …",
                    state="running",
                    expanded=True,
                )

        # AI MESSAGE
        if isinstance(message_chunk, AIMessage):
            yield message_chunk.content

    # Finalize only if a tool was actually used
    if status_holder["box"] is not None:

        status_holder["box"].update(
            label="✅ Tool finished",
            state="complete",
            expanded=False
        )

# *********************** USER INPUT ************************
if user_input:

    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.text(user_input)

    #invoking the llm for displaying the llm response in the chat interface
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            stream_response(user_input)
        )

    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message
    })