import queue
import streamlit as st
from langgraph_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    ingest_pdf,
    get_thread_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


# utility functions
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


async def _load_conversation_async(thread_id):
    state = await chatbot.aget_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def load_conversation(thread_id):
    return submit_async_task(
        _load_conversation_async(thread_id)
    ).result()  # return empty list if no message history


# SET UP A SESSION
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# sidebar UI
st.sidebar.title("GraphNexus")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# --- Document Context Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Document Context")

# 1. Check if the current thread already has a file
current_thread_id = str(st.session_state["thread_id"])
metadata = get_thread_metadata(current_thread_id)

if metadata:
    st.sidebar.success(
        f"📄 **Active:** {metadata.get('filename')}\n"
        f"Analyzed {metadata.get('documents')} pages ({metadata.get('chunks')} chunks)"
    )
else:
    st.sidebar.info("No document uploaded for this chat.")

# 2. File Uploader with Dynamic Key (resets when thread changes)
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF", type=["pdf"], key=f"uploader_{current_thread_id}"
)

if uploaded_file:
    # Only process if we haven't already (or if it's a new file)
    # Note: ingest_pdf is fast enough to run directly; for larger files, a spinner helps.
    if not metadata or metadata.get("filename") != uploaded_file.name:
        with st.sidebar.status("Indexing document...", expanded=True):
            ingest_pdf(uploaded_file.getvalue(), current_thread_id, uploaded_file.name)
        st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    # Highlight the current thread in the list or just show buttons
    if st.sidebar.button(str(thread_id), key=f"thread_btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(
                message, HumanMessage
            ):  # if curr message instance is human message
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state["message_history"] = temp_messages

# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if user_input := st.chat_input("Type here"):
    # 1. Display user message immediately
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # 2. Prepare Config
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # 3. Stream Assistant Response
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    # Stream both content and metadata to handle ToolMessages
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    st.error(f"Error: {metadata}")
                    break

                # Handling Tool Updates
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` ...",
                            state="running",
                            expanded=True,
                        )

                # Yielding Content Tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Close status box if it was opened
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # 4. Save Assistant message to history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
