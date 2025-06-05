import asyncio
from dataclasses import dataclass
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
from IPython.display import display, Markdown
from langchain.agents import Tool
from dotenv import load_dotenv

# This script demonstrates an AutoGen agent that can use both internet search and file management tools.
# The agent is tasked with searching for stock information and writing it to a file, using LangChain tools via adapters.

load_dotenv()

# Helper function to set up and return a list of tools for the agent
def get_tools():
    # Internet search tool (Google Serper via LangChain)
    serper = GoogleSerperAPIWrapper()
    langchain_serper = Tool(name="internet_search", func=serper.run, description="Tool for searching on internet")
    autogen_serper = LangChainToolAdapter(langchain_serper)  # Adapter for AutoGen
    autogen_tools = [autogen_serper]
    
    # File management tools (read/write files, list directory, etc.)
    langchain_file_management_tools = FileManagementToolkit(root_dir="output").get_tools()
    for tool in langchain_file_management_tools:
        autogen_tools.append(LangChainToolAdapter(tool))  # Wrap each tool for AutoGen
    
    return autogen_tools

# Helper function to get a model client for the agent (OpenAI or Ollama)
def get_model_client(model_name: str):
    if model_name == "gpt":
        return OpenAIChatCompletionClient(model="gpt-4o-mini")  # Use OpenAI GPT model
    elif model_name == "llama":
        return OllamaChatCompletionClient(model="llama3.2:1b")  # Use local Ollama Llama model


## send message and run the agent
async def main(company_name: str):
    autogen_tools = get_tools()  # Get the list of tools (search, file management)
    model_client = get_model_client("gpt")  # Choose the model client
    # Create the AssistantAgent with tool access and reflection enabled
    agent = AssistantAgent(name="searcher", 
                       model_client=model_client, 
                       tools=autogen_tools, 
                       reflect_on_tool_use=True
                      )

    # Compose the prompt for the agent
    prompt = f"""Your task is to search for stock information for the company named {company_name}, and write all the information to a file called stocks_data.md with full details.
        Information to include:
        - Stock price
        - Market cap
        - PE ratio
        - EPS
        - Dividend yield
        - 52-week range
        - 52-week performance
    
    """
    
    print(f"\n Prompt: \n {prompt}")
    message = TextMessage(content=prompt, source="user")
    
    # Send the initial prompt to the agent and await the response
    result = await agent.on_messages([message], cancellation_token=CancellationToken())
    print(f"\n Response: \n {result.chat_message.content}")

    # Send a follow-up message to proceed
    message = TextMessage(content="OK proceed", source="user")
    result = await agent.on_messages([message], cancellation_token=CancellationToken())
    
    print(f"\n Response: \n ")
    for message in result.inner_messages:
        print(message.content)
    
if __name__ == "__main__":
    asyncio.run(main(company_name="Tesla Inc"))
