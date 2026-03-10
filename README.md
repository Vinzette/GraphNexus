# GraphNexus

A multi-session AI chatbot built with **LangGraph** and **Streamlit**. GraphNexus provides persistent, tool-augmented conversations with real-time streaming — all running locally.

---

## Features

- **Persistent sessions** — every conversation is stored in a local SQLite database and survives app restarts
- **Multi-thread management** — each chat gets a unique UUID thread ID; switch between past conversations from the sidebar
- **Live streaming** — assistant tokens appear word-by-word via an async queue bridge
- **Tool use** — the model can call web search, a calculator, stock price lookup, and a remote MCP fetch tool mid-conversation
- **Async-safe architecture** — a dedicated background event loop decouples LangGraph's async runtime from Streamlit's synchronous rendering

---

## Tools

| Tool                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `duckduckgo_search` | General web search via DuckDuckGo                      |
| `calculator`        | Arithmetic operations: add, subtract, multiply, divide |
| `get_stock_price`   | Live stock quotes via Alpha Vantage                    |
| `fetch` _(MCP)_     | Fetch any webpage and return its content as Markdown   |

The `fetch` tool is loaded dynamically from a remote MCP server (`streamable_http` transport) using `langchain-mcp-adapters`.

---

## Architecture

```
┌─────────────────────────────────┐
│         Streamlit Frontend       │
│                                  │
│  Sidebar ──► Thread selector     │
│  Chat UI ──► st.write_stream     │
│                  │               │
│           Queue bridge           │
└──────────────────┼───────────────┘
                   │  submit_async_task()
┌──────────────────▼───────────────┐
│        Background Async Loop      │
│                                  │
│  ┌──────────────────────────┐    │
│  │      LangGraph Graph      │    │
│  │                          │    │
│  │  START ──► chat_node      │    │
│  │              │            │    │
│  │         tools_condition   │    │
│  │           ↙       ↘      │    │
│  │     [END]      tools node │    │
│  │                  │        │    │
│  │              chat_node ◄──┘    │
│  └──────────────────────────┘    │
│                                  │
│  AsyncSqliteSaver (chatbot.db)   │
└──────────────────────────────────┘
```

### Backend (`langgraph_backend.py`)

The LangGraph graph has two nodes:

- **`chat_node`** — calls the model asynchronously with full message history; the model may respond directly or emit tool calls
- **`tools`** — a `ToolNode` that executes any requested tools and appends results back into the message state

State is checkpointed to `chatbot.db` via `AsyncSqliteSaver`, keyed by `thread_id`. A daemon thread runs a persistent `asyncio` event loop, allowing the synchronous Streamlit process to dispatch coroutines safely.

### Frontend (`streamlit_frontend.py`)

The Streamlit UI handles session management and rendering:

- New chats generate a UUID thread ID
- Past threads are fetched from the checkpoint store on startup and listed in the sidebar
- When the model invokes a tool, a live `st.status` widget is shown and updated in real time
- Assistant tokens are yielded through a `queue.Queue` and consumed by `st.write_stream`

---

## Project Structure

```
graphnexus/
├── langgraph_backend.py   # Graph definition, tools, async loop, SQLite checkpointer
├── streamlit_frontend.py  # Streamlit UI and async streaming bridge
├── pyproject.toml         # Project metadata and dependencies
├── chatbot.db             # SQLite database (created on first run)
└── .env                   # API keys (not committed)
```

---

## Quick Start

### 1. Install dependencies

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key
```

| Variable               | Required | Notes                                                                                                                         |
| ---------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`       | Yes      | Powers the `gpt-5` chat model                                                                                                 |
| `ALPHAVANTAGE_API_KEY` | No       | Required only for the `get_stock_price` tool. Get one free at [alphavantage.co](https://www.alphavantage.co/support/#api-key) |

### 3. Run the app

```bash
uv run streamlit run streamlit_frontend.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Tech Stack

| Layer              | Technology                                                                         |
| ------------------ | ---------------------------------------------------------------------------------- |
| Orchestration      | [LangGraph](https://github.com/langchain-ai/langgraph)                             |
| LLM & tool binding | [LangChain](https://github.com/langchain-ai/langchain) + OpenAI (`gpt-5`)          |
| UI                 | [Streamlit](https://streamlit.io)                                                  |
| Persistence        | SQLite via `aiosqlite` + `AsyncSqliteSaver`                                        |
| MCP integration    | [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) |
| Package management | [`uv`](https://docs.astral.sh/uv/)                                                 |

---

## Example Prompts

- `Search for the latest news on quantum computing`
- `What is AAPL trading at right now?`
- `Calculate 847 divided by 13`
- `Fetch https://example.com and summarize it`
- `What's 15% of 2340, and then search for the best savings accounts?`
- `Search for the latest news about Nvidia.`
- `Calculate 42 * 19.`
- `Get the latest stock price for AAPL.`
- `Use the fetch tool to fetch https://www.anthropic.com/engineering/code-execution-with-mcp and summarize it.`

## Async Design Notes

This project uses async components in the backend and a synchronous UI in the frontend. The important pieces are:

- `chat_node` uses `ainvoke(...)`
- SQLite checkpointing uses `AsyncSqliteSaver`
- thread restoration uses `aget_state(...)`
- streamed messages use `astream(...)`
- the frontend bridges async work through `submit_async_task(...)` and a queue

That design avoids blocking the Streamlit app while still supporting async LangGraph execution.

## Persistence Model

Every conversation is stored in `chatbot.db` using LangGraph checkpoints.

- Starting a new chat creates a new thread ID
- Reopening the app preserves old conversations
- Selecting a thread from the sidebar restores its message history

## MCP Fetch Support

The project connects to a remote MCP server for the `fetch` tool:

- endpoint: `https://remote.mcpservers.org/fetch/mcp`
- transport: `streamable_http`

This tool is useful when you want the assistant to retrieve webpage content as structured markdown instead of relying only on normal chat knowledge.

## Requirements

- Python 3.12+
- Internet access for OpenAI, DuckDuckGo search, Alpha Vantage, and the remote MCP fetch server

## Troubleshooting

### `ModuleNotFoundError` for MCP adapters

Install project dependencies again:

```bash
uv sync
```

### Fetch tool is not available

Check the following:

- `langchain-mcp-adapters` is installed
- the remote MCP endpoint is reachable from your network
- the backend is using `streamable_http` for the fetch transport
- you restarted the app after changing backend tool configuration

### Stock tool returns an error

Make sure `ALPHAVANTAGE_API_KEY` is present in `.env`.

## Roadmap Ideas

- Add better tool result rendering in the UI
- Add structured logging for MCP and tool-loading failures
- Add tests around thread restoration and streaming behavior
- Add deployment instructions for hosting the app beyond local development
