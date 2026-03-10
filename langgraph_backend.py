from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import sqlite3
import os
import asyncio
import aiosqlite
import threading
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
import requests # is used for making HTTP requests (e.g., GET, POST) to web services and APIs.

load_dotenv()

#async
# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

llm = ChatOpenAI(model='gpt-5')
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
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
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key loaded from the environment.
    """
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "Missing ALPHAVANTAGE_API_KEY in environment variables"}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    r = requests.get(url)
    return r.json()

#MCP
# Initialize the client
client = MultiServerMCPClient(
    {
        "fetch": {
            "transport": "streamable_http",
            "url": "https://remote.mcpservers.org/fetch/mcp"
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, calculator, *mcp_tools]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# node def
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    # take user query
    messages = state["messages"]
    # seed to llm
    response = await llm_with_tools.ainvoke(messages)
    # append back to history
    return {"messages": [response]}

tool_node =  ToolNode(tools) 

async def _init_checkpointer():
    conn = await aiosqlite.connect(database='chatbot.db')
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

graph = StateGraph(ChatState)
# adding nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
# edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpointer)

#helper
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())

# initial_state = {"messages": [HumanMessage(content="What is the capital of Japan")]}

# (chatbot.invoke(initial_state)['messages'][-1].content)
if __name__ == "__main__":
    thread_id = 1
    while True:
        user_message = input("Type here: ")
        print("User:", user_message)

        if user_message.strip().lower() in ["quit", "exit", "bye"]:
            break
        config = {"configurable": {"thread_id": thread_id}}
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_message)]}, config=config
        )
        print(
            "AI:", response["messages"][-1].content
        )  # response has msg history --> grab the last one
