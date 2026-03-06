# GraphNexus

A multi-session AI chatbot built with LangGraph and Streamlit. Conversations are persisted to a local SQLite database, so chat history survives page refreshes and app restarts.

## Features

- **Multi-turn conversations** — full message history is passed to the LLM on every turn
- **Multiple chat sessions** — start new chats at any time, each with its own isolated thread
- **Persistent storage** — all conversations are saved to `chatbot.db` via SQLite; nothing is lost on refresh
- **Session switching** — click any past conversation in the sidebar to reload it
- **Streaming responses** — AI replies stream token-by-token in real time

## Architecture

```
streamlit_frontend.py   — UI layer (Streamlit)
langgraph_backend.py    — LangGraph graph + SQLite checkpointer
chatbot.db              — SQLite database (auto-created on first run)
```

**Backend** (`langgraph_backend.py`):
- Defines a `ChatState` TypedDict holding the message list
- Single `chat_node` that calls `ChatOpenAI` and appends the response
- `SqliteSaver` checkpointer persists every graph state to `chatbot.db` keyed by `thread_id`
- `retrieve_all_threads()` queries all distinct thread IDs from the checkpoint store

**Frontend** (`streamlit_frontend.py`):
- Each new chat generates a UUID `thread_id`
- `st.session_state['message_history']` is the UI rendering cache for the current session
- On sidebar thread click, `chatbot.get_state()` reads the full history from SQLite and restores it
- Messages are streamed using `chatbot.stream(..., stream_mode='messages')`

## Setup

1. **Clone the repo and install dependencies**

   ```bash
   uv sync
   ```

2. **Set your OpenAI API key**

   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Run the app**

   ```bash
   uv run streamlit run streamlit_frontend.py
   ```

   The app opens at `http://localhost:8501`. The `chatbot.db` file is created automatically on first run.

## Requirements

- Python >= 3.12
- See `pyproject.toml` for full dependency list (LangGraph, LangChain, Streamlit, etc.)
