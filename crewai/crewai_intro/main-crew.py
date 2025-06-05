from crewai import Crew, Task, LLM, Process
from agents import ComedyAgents

# This script demonstrates how to orchestrate multiple CrewAI agents for a collaborative comedy workflow.
# It sets up agents, assigns them tasks, and runs them as a Crew.

# Base URL for the local Ollama LLM API (not used directly here, but could be for custom LLM endpoints)
OLLAMA_BASE_URL = "http://localhost:11434"

# Helper function to get an LLM instance by model name
def get_model(model_name="gpt"):
    if model_name == "llama":
        llm = LLM(model="ollama/llama3.2:1b")  # Use local Ollama Llama model
    elif model_name == "gpt":
        llm = LLM(model="openai/gpt-4o-mini")  # Use OpenAI GPT model
    return llm

llm = get_model("llama")  # Choose which LLM to use for the agents

# 1. Create agent instances using the ComedyAgents factory
jokester_agent = ComedyAgents.get_jokester_agent(llm)  # Agent that tells jokes
joke_curator_agent = ComedyAgents.get_joke_curator_agent(llm)  # Agent that collects and organizes jokes

# 2. Define tasks for each agent
# Task for the Jokester: tell a clever and funny joke
# Task for the Joke Curator: collect and organize a list of jokes

tell_joke_task = Task(
    description="Tell a clever and funny joke to entertain the audience.",
    expected_output="A well-crafted joke that is both clever and relatable, guaranteed to make people laugh.",
    agent=jokester_agent
)

collect_jokes_task = Task(
    description="Collect and organize a list of high-quality jokes suitable for various occasions.",
    expected_output="A categorized collection of jokes, organized by theme, style, and audience.",
    agent=joke_curator_agent
)

# 3. Create the Crew to orchestrate the agents and their tasks
crew = Crew(
    agents=[joke_curator_agent, jokester_agent],  # List of agents in the crew
    tasks=[collect_jokes_task, tell_joke_task],   # List of tasks to be performed
    verbose=True,                                # Enable verbose output for debugging
    process=Process.sequential                   # Run tasks sequentially
)

# 4. Kick off the CrewAI workflow and print the results
results = crew.kickoff()
print(results)