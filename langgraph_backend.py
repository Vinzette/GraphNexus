from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests # is used for making HTTP requests (e.g., GET, POST) to web services and APIs.

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI()
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



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)


# node def
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    # take user query
    messages = state["messages"]
    # seed to llm
    response = llm_with_tools.invoke(messages)
    # append back to history
    return {"messages": [response]}

tool_node =  ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) #same db will be used in diff threads
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
# adding nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
# edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads= set()
    for checkpoint in checkpointer.list(None): #give checkpoints for not a specific id
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

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
