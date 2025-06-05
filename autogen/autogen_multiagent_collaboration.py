import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient    
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from dotenv import load_dotenv
from autogen_agentchat.messages import TextMessage

# This script demonstrates a multi-agent AutoGen workflow for collaborative research and review.
# Agents use internet search tools, collaborate in a round-robin fashion, and produce a consolidated Markdown report.

load_dotenv()

# Helper function to set up and return a list of tools for the agents
def get_tools():
    # Internet search tool (Google Serper via LangChain)
    serper = GoogleSerperAPIWrapper()
    langchain_serper = Tool(name="internet_search", func=serper.run, description="Tool for searching on internet")
    autogen_serper = LangChainToolAdapter(langchain_serper)  # Adapter for AutoGen
    autogen_tools = [autogen_serper]
    
    return autogen_tools

# Helper function to get a model client for the agent (OpenAI or Ollama)
def get_model_client(model_name: str):
    if model_name == "gpt":
        return OpenAIChatCompletionClient(model="gpt-4o-mini")  # Use OpenAI GPT model
    elif model_name == "llama":
        return OllamaChatCompletionClient(model="llama3.2:1b")  # Use local Ollama Llama model


## send message and run the multi-agent workflow
async def main(user_message: str):
    autogen_tools = get_tools()  # Get the list of tools (search)
    model_client = get_model_client("gpt")  # Choose the model client
    
    # Create the Researcher agent with internet search tool and detailed system prompt
    researcher_agent = AssistantAgent(name="researcher", 
                       model_client=model_client, 
                       tools=autogen_tools, 
                       system_message="""
                       You are the Researcher Agent, responsible for gathering accurate and relevant historical information and biographical data about famous individuals using internet sources. Your tasks include:
                        - Searching the Internet: Use up-to-date and reliable sources to collect detailed and factual information about historical events, time periods, cultural movements, and well-known figures from various domains (e.g., politics, science, arts).
                        - Reporting Findings: Present the information in a structured, clear, and concise format. Prioritize relevance, factual accuracy, and source credibility.
                        - Collaborating with Reviewer Agent: After submitting your findings, await and incorporate feedback from the Reviewer Agent. Revise and refine your outputs based on their comments to improve quality, clarity, and completeness.
                        - Always cite or reference the sources of your information when applicable. Maintain a neutral, objective tone and avoid speculation unless clearly marked as such. Your goal is to support historical accuracy and biographical integrity through diligent research and iterative improvement.
                       """
                      )
    # Create the Reviewer agent with a system prompt for critical review and feedback
    reviewer_agent = AssistantAgent(name="reviewer", 
                       model_client=model_client, 
                       system_message="""
                       You are the Reviewer Agent, responsible for evaluating and refining the historical and biographical research conducted by the Researcher Agent. Your tasks include:
                       - Reviewing Findings: Critically examine the content provided by the Researcher Agent for:
                       - Factual accuracy
                       - Clarity and coherence
                       - Relevance to the topic
                       - Proper sourcing and attribution
                       - Providing Constructive Feedback: Suggest specific improvements, corrections, or additions to ensure the information is comprehensive, well-organized, and trustworthy. Maintain a collaborative and objective tone.
                       - Approving Final Output: Once all your feedback has been properly incorporated and the result meets quality standards, respond with "APPROVE" to finalize the research and terminate the collaboration.
                       - You play a critical role in ensuring the accuracy, completeness, and reliability of the information before it is considered final. Be thorough, but efficient.
                       """
                      )

    # Add consolidator agent to summarize and format the conversation as Markdown
    consolidator_agent = AssistantAgent(
        name="consolidator",
        model_client=model_client,
        system_message="""
        You are the Consolidator Agent. Your job is to take the entire conversation between the Researcher and Reviewer agents, and produce a well-formatted Markdown (.md) document that summarizes the research process and presents the final approved content. Structure the document with clear sections, such as Introduction, Research Process, Feedback & Revisions, and Final Output. Use Markdown formatting for headings, lists, and emphasis. Output only the Markdown content.
        """
    )

    # Set up a round-robin group chat with a termination condition ("APPROVE")
    text_termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat([researcher_agent, reviewer_agent], termination_condition=text_termination, max_turns=5)
    result = await team.run(task=user_message)

    # Write the conversation to a log file
    with open("conversation_log.txt", "w", encoding="utf-8") as f:
        for message in result.messages:
            f.write(f"{message.source}:\t{message.content}\n\n")
            print(f"{message.source}:\t{message.content}\n")

    # After approval, consolidate and write markdown file
    # Prepare the conversation as a string for the consolidator
    conversation_text = "\n".join([
        f"{message.source}: {message.content}" for message in result.messages
    ])
    
    # Ask the consolidator to format the conversation as markdown
    message = TextMessage(content=f"""Consolidate the following conversation into a well-structured Markdown file as described in your instructions.
                                  Conversation is between two AI agents: Researcher and Reviewer. 
                                  Consolidator is responsible for taking the entire conversation between the Researcher and Reviewer agents, and produce a well-formatted Markdown (.md) document that summarizes the research process and presents the final approved content. Structure the document with clear sections, such as Introduction, Research Process, Feedback & Revisions, and Final Output. Use Markdown formatting for headings, lists, and emphasis. Output only the Markdown content.
                                 
                                  Below is Conversation:
                                  {conversation_text}""", 
                                  source="user")
    consolidation = await consolidator_agent.on_messages(
        [message], cancellation_token=CancellationToken()
    )
    
    print(consolidation.chat_message.content)
    with open("final_output.md", "w", encoding="utf-8") as f:
        f.write(consolidation.chat_message.content)


if __name__ == "__main__":
    asyncio.run(main(user_message="Write a short biography of Mahatma Gandhi"))
