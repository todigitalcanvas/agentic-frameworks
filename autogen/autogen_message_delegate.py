import asyncio
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from dotenv import load_dotenv

# This script demonstrates an AutoGen agent that delegates joke generation to an LLM-powered assistant agent.
# It shows how to set up model clients, delegate message handling, and orchestrate agent communication asynchronously.

load_dotenv()

# Helper function to get a model client for the assistant agent
def get_model_client(model_name: str):
    if model_name == "gpt":
        return OpenAIChatCompletionClient(model="gpt-4o-mini")  # Use OpenAI GPT model
    elif model_name == "llama":
        return OllamaChatCompletionClient(model="llama3.2:1b")  # Use local Ollama Llama model


# Step 1: Define the message class
# This class represents the structure of messages exchanged between agents and users
@dataclass
class Message:
    content: str


# Step 2: Define the agent class
# This agent delegates joke generation to an LLM-powered AssistantAgent
class JokeSterAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("JokeSter")  # Initialize the agent with a name
        model_client = get_model_client("gpt")  # Choose the model client (GPT or Llama)
        self._delegate = AssistantAgent("JokeSter", model_client=model_client)  # Delegate for LLM responses

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        # This handler is called when the agent receives a message
        user_message = message.content
        
        # Wrap the user message as a TextMessage for the assistant
        text_message = TextMessage(content=user_message, source="user")
        # Delegate the message to the AssistantAgent and await the LLM's response
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        response_content = response.chat_message.content
        
        # Compose the reply, echoing the user's message and including the LLM-generated joke
        reply = f"""This is {self.id.type}-{self.id.key}.
        You said: '{user_message}'
        Here's a joke for you: {response_content}"""
        return Message(content=reply)

## send message and run the agent
async def main():
    # Step 3: Create the runtime (single-threaded for simplicity)
    runtime = SingleThreadedAgentRuntime()
    # Register the JokeStarAgent with the runtime
    await JokeSterAgent.register(runtime, "joke_ster_agent", lambda: JokeSterAgent())
    runtime.start()
    # Step 4: Create an AgentId for the registered agent
    agent_id = AgentId("joke_ster_agent", "default")
    
    # Send a message to the agent and await the response
    result = await runtime.send_message(
        Message("Well hi there! Can you tell me a AI joke?"), agent_id
        )
    
    print(f"\n Response: \n {result.content}")  # Print the agent's response



if __name__ == "__main__":
    asyncio.run(main())
