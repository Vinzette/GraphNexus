from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI()
#node def
def chat_node(state: ChatState):
    #take user query
    messages = state['messages']
    #seed to llm
    response = llm.invoke(messages)
    #append back to history
    return {'messages': [response]}


graph = StateGraph(ChatState)
#adding nodes
graph.add_node('chat_node', chat_node)
#eges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile()

initial_state={
    'messages': [HumanMessage(content="What is the capital of Japan")]
}

print(chatbot.invoke(initial_state))