import warnings
from datetime import datetime
from crewai_version.crew import DataScienceCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    inputs = {
        'user_question': input("What your need ? specifiy how the output format is : "),
        'dataset_path' : input("dataset path : "),
        'target_column' : input("target column : ")
    }

    try:

        DataScienceCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
