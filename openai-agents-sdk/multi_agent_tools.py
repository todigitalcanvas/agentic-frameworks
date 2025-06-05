# Import necessary classes and functions from the agents package
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    trace,
    WebSearchTool,
)
import asyncio  # For running asynchronous code
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Load environment variables from .env file
load_dotenv()

# Get Model - Local LLM Ollama, LLM GPT
def get_model(model_name="llama"):
    """
    Returns a model instance based on the model_name.
    - If 'llama', returns a local Ollama LLM model wrapped for OpenAI compatibility.
    - If 'gpt', returns the string identifier for the GPT model.
    """
    if model_name == "llama":
        external_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1", api_key="not-needed"
        )
        model = OpenAIChatCompletionsModel(
            openai_client=external_client, model="llama3.2:1b"
        )
    elif model_name == "gpt":
        model = "gpt-4o-mini"
    return model

# Agent to be used as a tool for web searching
web_searcher_tool = Agent(
        name="Web searcher",
        instructions="You are a helpful agent.",
        model=get_model(model_name="gpt")
        )

# System prompt for agents responsible for web searching and reporting
SYSTEM_PROMPT = """
    You are a helpful agent. You are responsible for searching the web for information about the user's query.
    You have access to the following tools:
    - Web Search
"""

# Details Reporter agent: uses the web_searcher_tool for detailed web searches
# This agent is responsible for providing more detailed information based on a summary
# It uses the web_searcher_tool as a tool

details_reporter = Agent(
        name="Details Reporter",
        instructions=SYSTEM_PROMPT,
        model=get_model(model_name="gpt"),
        tools=[web_searcher_tool.as_tool(
            tool_name="web_search",
            tool_description="""You are responsible for searching the web for detailed information about the user's query.
            you will be given summary of interesting updates, and you will need to search the web for more details.
            """ ,
            )],
    )

# Search Reporter agent: uses the web_searcher_tool for initial web searches
# This agent is responsible for providing a summary of interesting updates
# It uses the web_searcher_tool as a tool

search_reporter = Agent(
        name="Search Reporter",
        instructions=SYSTEM_PROMPT,
        model=get_model(model_name="gpt"),
        tools=[web_searcher_tool.as_tool(
            tool_name="web_search",
            tool_description="You are responsible for searching the web for information about the user's query",
            )],
    )

# Main function to run the multi-agent workflow
async def main(user_input: str):
    
    # Handoffs: first get a summary, then get more details
    with trace("Search Reporter"):
        result_summary = await Runner.run(search_reporter, user_input)
        
    with trace("Details Reporter"):
        result_details = await Runner.run(details_reporter, result_summary.final_output)
    
    print(f"\nResult from search_reporter: \n {result_summary.final_output} \n")
    print(f"\nResult from details_reporter: \n {result_details.final_output} \n {'*' * 100}")

# Entry point for running the script directly
if __name__ == "__main__":
    asyncio.run(
        main(
            user_input="""Search the web for local sports news, and give me 1 interesting update in a sentence.
            I am interested in the Toronto Raptors, Toronto Blue Jays, and Toronto Maple Leafs.
            """
        )
    )
