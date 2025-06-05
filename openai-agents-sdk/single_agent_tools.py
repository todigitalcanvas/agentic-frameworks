# Import necessary classes and functions from the agents package
from agents import Agent, Runner
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, trace, function_tool
import asyncio  # For running asynchronous code
from dotenv import load_dotenv  # For loading environment variables from a .env file
from typing import Any  # For type annotations
import httpx  # For making async HTTP requests

# Load environment variables from .env file
load_dotenv()

# Base URL for the National Weather Service API
NWS_API_BASE = "https://api.weather.gov"
# User-Agent string required by the NWS API
USER_AGENT = "weather-app/1.0"

# Make an async request to the NWS API with error handling
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,  # Required by NWS API
        "Accept": "application/geo+json"  # Request GeoJSON response
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)  # Make async GET request
            response.raise_for_status()  # Raise error for bad status
            return response.json()  # Return parsed JSON
        except Exception:
            return None  # Return None on any error

# Format a single alert feature into a readable string
def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]  # Extract properties from feature
    return f"""
                Event: {props.get('event', 'Unknown')}
                Area: {props.get('areaDesc', 'Unknown')}
                Severity: {props.get('severity', 'Unknown')}
                Description: {props.get('description', 'No description available')}
                Instructions: {props.get('instruction', 'No specific instructions provided')}
            """

# Tool function to get weather alerts for a US state
@function_tool
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"  # Build API URL for state
    data = await make_nws_request(url)  # Fetch data from NWS API
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    if not data["features"]:
        return "No active alerts for this state."
    alerts = [format_alert(feature) for feature in data["features"]]  # Format each alert
    return "\n---\n".join(alerts)  # Join alerts with separator

# Get a model instance for either local LLM (Ollama) or GPT
def get_model(model_name="llama"):
    if model_name == "llama":
        # Create AsyncOpenAI client for local Ollama server
        external_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
        # Wrap client in OpenAIChatCompletionsModel for compatibility
        model = OpenAIChatCompletionsModel(openai_client=external_client, model="llama3.2:1b")
    elif model_name == "gpt":
        # Use GPT model identifier
        model = "gpt-4o-mini"
    return model

# System prompt for the weather reporter agent
SYSTEM_PROMPT = """
You are a weather reporter. You are responsible for reporting the weather alerts for a given state.
You have access to the following tools:
- get_alerts

You will be given a state. You will use the get_alerts tool to get the weather alerts for the state.
"""

# Main function to run the weather reporter agent
async def main(user_input: str):
    model = get_model(model_name="gpt")  # Get the GPT model
    weather_reporter = Agent(
        name="Weather Reporter",  # Name of the agent
        instructions=SYSTEM_PROMPT,  # System prompt for the agent
        model=model,  # Model to use
        tools=[get_alerts]  # Register the get_alerts tool
    )
    
    result = await Runner.run(weather_reporter, user_input)  # Run the agent with user input
    print(f"\nResult from weather_reporter: \n {result.final_output} \n {'*' * 100}")  # Print the result


# Entry point for running the script directly
if __name__ == "__main__":
    asyncio.run(main("What is the weather in Texas?"))  # Example user input






