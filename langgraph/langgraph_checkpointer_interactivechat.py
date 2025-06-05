from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import Tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Persistent memory imports
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# This script demonstrates a LangGraph workflow with persistent memory integration.
# The agent can use tools (e.g., web search) and maintain chat history across sessions using SQLite or in-memory storage.

load_dotenv()

# Setup Tools
# Instantiate a Google Serper API wrapper for web search
serper = GoogleSerperAPIWrapper()
# Define a LangChain Tool for online search
tool_search = Tool(
        name="search",
        func=serper.run,
        description="Useful for when you need more information from an online search"
    )

# List of available tools for the agent
tools = [tool_search]

# Helper function to get an LLM instance by model name
def get_llm(model_name: str):
    if model_name == "gpt":
        return ChatOpenAI(model="gpt-4o-mini")  # Use OpenAI GPT model
    elif model_name == "llama":
        return ChatOllama(model="llama3.2:1b")  # Use local Ollama Llama model

# Helper function to get a memory saver (in-memory or SQLite for persistence)
def get_memory(memory_type: str = "in-memory"):
    if memory_type == "in-memory":
        return MemorySaver()  # Volatile, in-memory storage
    elif memory_type == "sqlite":
        db_path = "memory.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        sql_memory = SqliteSaver(conn)  # Persistent SQLite storage
        return sql_memory


## Step 1. Define State
class State(BaseModel):
    # The state holds a list of messages. The add_messages reducer will append new messages to this list.
    messages: Annotated[list, add_messages]


def main():
    # Setup LLM and bind tools to it
    llm = get_llm("gpt")
    llm_with_tools = llm.bind_tools(tools)
    
    # Setup memory - save chat history (choose between in-memory or SQLite)
    memory = get_memory("sqlite")

    ## Step 2 -> start graph builder
    graph_builder = StateGraph(State)  # Initialize a stateful graph with the State schema

    ## Step 3 -> Define Nodes
    # The chatbot node: generates a response using the LLM (with tool access)
    def chatbot(state: State) -> State:
        new_messages = [llm_with_tools.invoke(state.messages)]
        return State(messages=new_messages)

    # Add nodes to the graph
    graph_builder.add_node("chatbot", chatbot)  # Main conversational node
    graph_builder.add_node("tools", ToolNode(tools=tools))  # Tool node for executing tool calls


    ## Step 4 -> create edge
    # Conditional edge: if the chatbot decides a tool is needed, transition to the tool node
    graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
    # After tool execution, return to the chatbot node
    graph_builder.add_edge("tools", "chatbot")
    # Start the conversation at the chatbot node
    graph_builder.add_edge(START, "chatbot")
    # Allow the chatbot to end the conversation
    graph_builder.add_edge("chatbot", END)


    ## Step 5 -> compile graph with memory
    graph = graph_builder.compile(checkpointer=memory)  # Compile the graph with persistent memory

    ## Step 6 -> create entry for chat and invoke graph
    config = {"configurable": {"thread_id": "1"}}  # Use a thread ID for persistent chat sessions
    def chat(user_input: str, history):
        # Helper function for single-turn chat (not used in main loop)
        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
        return result["messages"][-1].content


    ## Step 7 -> Accept user input and invoke graph
    # Interactive chat loop: accept user input, run the graph, and print the AI's response
    while True:
        user_input = input("Enter your message: ")
        if user_input.strip().lower() in ["exit", "bye"]:
            print("Goodbye!")
            break
        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
        print(f"AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()

