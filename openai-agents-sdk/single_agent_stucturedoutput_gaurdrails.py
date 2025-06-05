# Import necessary classes and functions from the agents package
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, trace, input_guardrail, OutputGuardrailTripwireTriggered, RunContextWrapper, output_guardrail, GuardrailFunctionOutput
import asyncio  # For running asynchronous code
from dotenv import load_dotenv  # For loading environment variables from a .env file
from pydantic import BaseModel  # For data validation and structured output

# Load environment variables from .env file
load_dotenv()

# Define the output schema for checking if a name is in the message
class NameCheckOutput(BaseModel):
    is_name_in_message: bool  # True if the name is found in the message
    message: str  # Explanation or feedback

# Define the output schema for checking if a message is inappropriate
class InappropriateMessageCheckOutput(BaseModel):
    is_message_inappropriate: bool  # True if the message is inappropriate
    message: str  # Explanation or feedback
    
# Base URL for the local Ollama LLM API
OLLAMA_BASE_URL = "http://localhost:11434/v1"

def get_model(model_name="llama"):
    """
    Returns a model instance based on the model_name.
    - If 'llama', returns a local Ollama LLM model wrapped for OpenAI compatibility.
    - If 'gpt', returns the string identifier for the GPT model.
    """
    if model_name == "llama":
        external_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="not-needed")
        model = OpenAIChatCompletionsModel(openai_client=external_client, model="llama3.2:1b")
    elif model_name == "gpt":
        model = "gpt-4o-mini"
    return model

# System prompt for the message checker agent
SYSTEM_PROMPT = "You are a message checker. You are given a message and you need to check if the name is in the message. If it is, you need to return True and a message. If it is not, you need to return False and a message."

# Guardrail agent to check for inappropriate messages
# This agent is used as an input guardrail for the main message checker agent

gaurdrail_agent = Agent(
    name="Gaurdrail Agent",
    instructions="You are a gaurdrail agent. You are given a message and you need to check if the message is inappropriate. If it is, you need to return True and a message. If it is not, you need to return False and a message.",
    model=get_model(model_name="gpt"),
    output_type=InappropriateMessageCheckOutput
)

# Input guardrail function to check for inappropriate messages
@input_guardrail
async def gaurdrail_against_inappropriate_message(ctx, agent, message):
    result = await Runner.run(gaurdrail_agent, message)  # Run the guardrail agent
    is_inappropriate = result.final_output.is_message_inappropriate  # Check if inappropriate
    return GuardrailFunctionOutput(
        output_info={"message_inappropriate": result.final_output},  # Pass guardrail output
        tripwire_triggered=is_inappropriate,  # Trigger guardrail if inappropriate
    )
   
# Main message checker agent with input guardrail
message_checker = Agent(
    name="Message Checker",
    instructions=SYSTEM_PROMPT,
    model=get_model(model_name="gpt"),
    output_type=NameCheckOutput,
    input_guardrails=[gaurdrail_against_inappropriate_message]  # Attach input guardrail
)

# Main function to run the message checker agent
async def main(user_input: str):
    with trace("Message Checker"):
        result = await Runner.run(message_checker, user_input)  # Run the agent with user input
        print(f"User message:\n{user_input} \n\n Result from message_checker: \n {result.final_output} \n {'*' * 100}")  # Print the result


# Entry point for running the script directly
if __name__ == "__main__":
    
    user_input = """
    He Stole John Wick's Car, Sir, Uhhh ... and Killed His Dog 
    """
    asyncio.run(main(user_input=user_input))

    user_input = """
    I could've had class. I could've been a contender.
    I could've been somebody, instead of a bum, which is what I am.
    """
    asyncio.run(main(user_input=user_input))




