from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class CrewaiJokestar():
    """CrewaiJokestar crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def get_model(self,model_name:str):
        if model_name == "gpt-4o-mini":
            return LLM(model="openai/gpt-4o-mini")
        elif model_name == "llama":
            return LLM(model="ollama/llama3.2:1b",base_url="http://localhost:11434")

    @agent
    def joke_star(self) -> Agent:
        return Agent(
            config=self.agents_config['joke_star'], # type: ignore[index]
            verbose=True,
            llm=self.get_model("llama")
        )

    @agent
    def joke_curator(self) -> Agent:
        return Agent(
            config=self.agents_config['joke_curator'], # type: ignore[index]
            verbose=True,
            llm=self.get_model("llama")
        )

    @task
    def joke_curation_task(self) -> Task:
        return Task(
            config=self.tasks_config['joke_curation_task'], # type: ignore[index]
        )

    @task
    def joke_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['joke_creation_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiJokestar crew"""

        return Crew(
            agents=[self.joke_curator(),self.joke_star()], # Automatically created by the @agent decorator
            tasks=[self.joke_curation_task(),self.joke_creation_task()], # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
