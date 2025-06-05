#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crewai_tools_context.crew import CrewaiToolsContext

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    
    
    if len(sys.argv) > 1:
        company = sys.argv[1]
    else:
        company = input("Enter company name: ")
    
    inputs = {
        "company": company,
        'topic': "Financial and stock market.",
        'current_year': str(datetime.now().year)
    }
    
    try:
        results = CrewaiToolsContext().crew().kickoff(inputs=inputs)
        print(f"\n Outcome from Agent: \n {results}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
