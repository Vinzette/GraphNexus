from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI()


# node def
def chat_node(state: ChatState):
    # take user query
    messages = state["messages"]
    # seed to llm
    response = llm.invoke(messages)
    # append back to history
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) #same db will be used in diff threads
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
# adding nodes
graph.add_node("chat_node", chat_node)
# eges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads= set()
    for checkpoint in checkpointer.list(None): #give checkpoints for not a specific id
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

initial_state = {"messages": [HumanMessage(content="What is the capital of Japan")]}

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
