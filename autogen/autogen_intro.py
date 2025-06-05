import asyncio
from dataclasses import dataclass
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, SingleThreadedAgentRuntime
import random

# This script demonstrates a simple AutoGen agent that responds to user messages with a random AI-themed joke.
# It shows how to define message and agent classes, register the agent, and send/receive messages asynchronously.

# Helper function to return a random AI-themed joke
def get_ai_joke():
    jokes = [
        "Why did the AI go to therapy? It had deep learning issues.",
        "I asked my AI assistant to tell me a joke. It replied: 'I'm afraid I can't do that, Dave.'",
        "Why did the neural network break up with the algorithm? It lost its sense of direction.",
        "Why was the robot so bad at soccer? It kept kicking up errors.",
        "Why don't AI systems make good comedians? Their timing is always off… by milliseconds.",
        "How does an AI flirt? It gives you its number… of training epochs.",
        "Why did the AI cross the road? It was part of its training data.",
        "What did the human say to the AI after losing a chess game? 'I need to process my feelings, not just data!'",
        "My AI friend told me it dreams of electric sheep. I told it to lay off the sci-fi books.",
        "Why was the AI so calm during the apocalypse? It had already simulated the outcome 10,000 times."
    ]
    return random.choice(jokes)

# Step 1: Define the message class
# This class represents the structure of messages exchanged between agents and users
@dataclass
class Message:
    content: str


# Step 2: Define the agent class
# This agent responds to incoming messages with a joke and echoes the user's message
class JokeStarAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("JokeStar")  # Initialize the agent with a name

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        # This handler is called when the agent receives a message
        joke = get_ai_joke()  # Get a random joke
        user_message = message.content
        reply = f"""This is {self.id.type}-{self.id.key}.
        You said: '{user_message}'
        Here's a joke for you: {joke}"""
        return Message(content=reply)  # Return the reply as a Message object

## send message and run the agent
async def main():
    # Step 3: Create the runtime (single-threaded for simplicity)
    runtime = SingleThreadedAgentRuntime()
    # Register the JokeStarAgent with the runtime
    await JokeStarAgent.register(runtime, "joke_star_agent", lambda: JokeStarAgent())
    runtime.start()
    # Step 4: Create an AgentId for the registered agent
    agent_id = AgentId("joke_star_agent", "default")
    
    # Send a message to the agent and await the response
    result = await runtime.send_message(
        Message("Well hi there! Can you tell me a joke?"), agent_id
        )
    
    print(f"\n Response: \n {result.content}")  # Print the agent's response



if __name__ == "__main__":
    asyncio.run(main())
