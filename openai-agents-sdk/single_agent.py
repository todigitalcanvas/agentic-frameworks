# Import necessary classes and functions from the agents package
from agents import Agent, Runner
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, trace
from dotenv import load_dotenv  # For loading environment variables from a .env file
import asyncio  # For running asynchronous code

# Load environment variables from .env file
load_dotenv()

# Base URL for the local Ollama LLM API
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def get_model(model_name="llama"):
    """
    Returns a model instance based on the model_name.
    - If 'llama', returns a local Ollama LLM model wrapped for OpenAI compatibility.
    - If 'gpt', returns the string identifier for the GPT model.
    """
    if model_name == "llama":
        # Create an AsyncOpenAI client for the local Ollama server
        external_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="not-needed")
        # Wrap the client in an OpenAIChatCompletionsModel for compatibility
        model = OpenAIChatCompletionsModel(
            openai_client=external_client, model="llama3.2:1b"
        )
    elif model_name == "gpt":
        # Use the GPT model identifier (assumed to be handled by the agents package)
        model = "gpt-4o-mini"
    return model

# System prompt for the agents, instructing them to act as a standup comedian
SYSTEM_PROMPT = (
    "You are a standup comedian. You are funny and you are good at making people laugh."
)


async def main(user_input: str):
    """
    Runs two agents (one using a local LLM, one using GPT) with the same prompt and user input.
    Prints the results from both agents for comparison.
    """

    # Local LLM Ollama
    model = get_model(model_name="llama")
    with trace("Assistant1 trace"):
        # Create an agent using the local LLM
        agent = Agent(name="Assistant1", instructions=SYSTEM_PROMPT, model=model)
        result = await Runner.run(agent, user_input)
        print(f"\n Result from Ollama: \n {result.final_output} \n {'*' * 100}")

    # LLM GPT
    model = get_model(model_name="gpt")
    with trace("Assistant2 trace"):
        # Create an agent using the GPT model
        agent = Agent(name="Assistant2", instructions=SYSTEM_PROMPT, model=model)
        result = await Runner.run(agent, user_input)
        print(f"\n Result from GPT: \n {result.final_output} \n {'*' * 100}")


if __name__ == "__main__":
    # Example user input for the agents
    user_input = "Tell a joke about Autonomous AI Agents"
    asyncio.run(main(user_input=user_input))
