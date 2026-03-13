# GraphNexus

An AI-powered multi-tool chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Streamlit](https://streamlit.io). GraphNexus gives you a persistent, multi-session conversational assistant that can search the web, query your PDFs, look up stock prices, do math, and fetch web pages — all from a clean chat interface.

---

## Features

- **Multi-session chat** — create multiple independent conversation threads, each persisted across restarts via SQLite
- **RAG over PDFs** — upload a PDF per thread and ask questions about its contents; answers are grounded in the actual document
- **Web search** — live DuckDuckGo search for up-to-date information the model wasn't trained on
- **Stock prices** — real-time stock quotes via Alpha Vantage (e.g. `What's the price of AAPL?`)
- **Calculator** — reliable arithmetic via a dedicated tool so the LLM never guesses
- **Web page fetching** — fetch and read arbitrary URLs via a remote MCP server
- **Streaming responses** — token-by-token streaming with live tool-use status indicators
- **LangSmith tracing** — optional observability for every agent run

---

## Tech Stack

| Layer                    | Technology                              |
| ------------------------ | --------------------------------------- |
| LLM                      | OpenAI `gpt-5`                          |
| Embeddings               | OpenAI `text-embedding-3-small`         |
| Agent orchestration      | LangGraph (`StateGraph`)                |
| Conversation persistence | LangGraph + `AsyncSqliteSaver` (SQLite) |
| Vector search (RAG)      | FAISS (in-memory, per-thread)           |
| Web search               | DuckDuckGo (`langchain-community`)      |
| Stock data               | Alpha Vantage REST API                  |
| Web fetching             | MCP via `langchain-mcp-adapters`        |
| Frontend                 | Streamlit                               |
| Package manager          | [`uv`](https://github.com/astral-sh/uv) |
| Python                   | 3.12                                    |

---

## Architecture

```
streamlit_frontend.py
  └── langgraph_backend.py
        ├── StateGraph
        │     ├── chat_node  — GPT-5 with all tools bound
        │     ├── tools node — ToolNode executing tool calls
        │     └── AsyncSqliteSaver — persists state to chatbot.db
        │
        ├── Tools
        │     ├── rag_tool          — per-thread FAISS retriever
        │     ├── DuckDuckGoSearch  — live web search
        │     ├── get_stock_price   — Alpha Vantage REST
        │     ├── calculator        — arithmetic operations
        │     └── MCP fetch tools   — remote.mcpservers.org
        │
        └── RAG Pipeline
              ├── PyPDFLoader
              ├── RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
              └── FAISS + OpenAIEmbeddings
```

The agent runs on a dedicated background asyncio event loop in a daemon thread, bridged to Streamlit's synchronous execution context via a thread-safe `queue.Queue`. This keeps the UI responsive while the agent processes tool calls asynchronously.

---

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

---

## Installation

```bash
git clone https://github.com/your-username/GraphNexus.git
cd GraphNexus

# Install all dependencies
uv sync
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-...
ALPHAVANTAGE_API_KEY=...

# Optional — enables LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGSMITH_PROJECT=GraphNexus
```

Get your API keys:

- **OpenAI** — [platform.openai.com](https://platform.openai.com)
- **Alpha Vantage** — [alphavantage.co](https://www.alphavantage.co/support/#api-key) (free tier available)
- **LangSmith** — [smith.langchain.com](https://smith.langchain.com) (optional)

---

## Running the App

```bash
uv run streamlit run streamlit_frontend.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Chat

Type any message in the chat input. The agent automatically decides which tools to use based on your question. A status indicator shows when a tool is active.

### PDF Question Answering

1. Open the **sidebar** and expand the PDF section for your current thread
2. Upload a PDF file — it will be parsed, chunked, and indexed automatically
3. Ask questions about it in the chat; the agent's `rag_tool` retrieves the most relevant passages

### Thread Management

- Click **New Chat** in the sidebar to start a fresh conversation thread
- Previous threads are listed by creation time; click any to resume it
- All threads and their full message histories persist in `chatbot.db`

### Example Prompts

```
What's the current price of Tesla stock?
Search the web for the latest news on AI regulation.
What does the uploaded document say about X?
What is 1234 * 5678?
Fetch and summarize https://example.com
```

---

## Project Structure

```
GraphNexus/
├── langgraph_backend.py   # Agent graph, tools, RAG pipeline, async event loop
├── streamlit_frontend.py  # Streamlit UI, streaming, thread management
├── pyproject.toml         # Dependencies and project metadata
├── uv.lock                # Pinned dependency tree
├── .env                   # API keys (not committed)
├── .python-version        # Pins Python 3.12
└── chatbot.db             # SQLite conversation store (auto-created, not committed)
```
