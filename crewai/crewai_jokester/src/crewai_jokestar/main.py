#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crewai_jokestar.crew import CrewaiJokestar

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your crew locally

def run():
    """
    Run the crew.
    """
    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = input("Enter joke topic: ")
        
    inputs = {
        'topic': topic
    }
    
    try:
        result=CrewaiJokestar().crew().kickoff(inputs=inputs)
        print(f" \n Outcome from agent : \n {result}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
    
