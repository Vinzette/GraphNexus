from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

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


checkpointer = MemorySaver()
graph = StateGraph(ChatState)
# adding nodes
graph.add_node("chat_node", chat_node)
# eges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

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
