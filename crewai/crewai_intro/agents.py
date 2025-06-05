from textwrap import dedent
from crewai import Agent, LLM

# This module defines specialized CrewAI agents for a comedy use case.
# Each agent is designed with a specific role, goal, and backstory to support collaborative agentic workflows.

class ComedyAgents():
    # Factory method to create a Jokester agent
    # This agent's role is to entertain by telling clever and funny jokes.
    def get_jokester_agent(llm):
      return Agent(
        role="JokesterAgent",  # Defines the agent's role in the CrewAI ecosystem
        goal='To entertain and lighten the mood by telling clever and funny jokes.',  # The agent's primary objective
        backstory=dedent("""
            You are a stand-up comedian with a knack for crafting jokes that are both clever and relatable.
            You are known for your quick wit and ability to make people laugh.
            """),  # Provides context and personality for the agent
        verbose=False,  # Controls verbosity of agent's output (CrewAI feature)
        allow_delegation=False,  # Prevents this agent from delegating tasks to others
        llm=llm  # The language model instance powering this agent
      )
      
    # Factory method to create a Joke Curator agent
    # This agent's role is to gather, organize, and maintain a collection of jokes.
    def get_joke_curator_agent(llm):
        return Agent(
            role="JokeCuratorAgent",  # Defines the agent's role in the CrewAI ecosystem
            goal="To gather, organize, and maintain a top-notch collection of jokes for every occasion.",  # The agent's primary objective
            backstory=dedent("""
                As a passionate archivist of humor, you scour the world for the best jokes—old and new. 
                Your mission is to ensure the Jokester always has access to a diverse and hilarious repertoire. 
                You take pride in categorizing jokes by theme, style, and audience, making it easy to find the perfect joke for any moment.
            """),  # Provides context and personality for the agent
            verbose=False,  # Controls verbosity of agent's output (CrewAI feature)
            allow_delegation=False,  # Prevents this agent from delegating tasks to others
            llm=llm  # The language model instance powering this agent
        )