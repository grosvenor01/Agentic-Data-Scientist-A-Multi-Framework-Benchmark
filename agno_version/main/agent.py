from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from .config import Settings
from .tools import *
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.local_file_system import LocalFileSystemTools
from agno.tools.file_generation import FileGenerationTools
from pathlib import Path


# knt nawi ndirha docstring
""" Name: 
        Role: AI Trainer Agent, this agent recieve trainig set as input, precising the target, and it will run and train the appropriate machine learning algorithms 
        Retrun Type: Machine learning model with trained parameters.
"""

# api_key= settings.gemini_api_key
#settings = Settings()

Data_Agent = Agent(
    # model = Gemini(id = "gemini-2.5-flash-lite", api_key=  ""),
    model = OpenAIChat(id = 'gpt-4.1-nano', api_key= ""),
    name = "Data_agent",
    role = "Agent that splits datasets and prepares features",
    tools = [dataLoader],
    instructions = ["You split the dataset into Train and test sets, recognize and extract target feature based on your understanding of user query"],
    add_history_to_context = True,
    markdown = True
)
Trainer_Agent = Agent(
    # model = Gemini(id = "gemini-2.5-flash-lite",  api_key= ""),
    # model = Ollama(id= "llama3.1"),
    model = OpenAIChat(id = 'gpt-4.1-nano', api_key= ""),
    name= "Trainer Agent",
    role= "Agent that trains ML models accordingly to the need of the user and input/output features specifications",
    tools = [MLTools()],
    instructions = ["You are an AI engineer agent, your will recieve preprocessed datasets, and your mission is to train models on the provided data to perform predictions. you will chose the appropriate machine learning algorithms to select based on the nature of the provided data, and user queries/ specifications."],
    add_history_to_context = True,
    markdown = True
)

Evaluator_Agent = Agent(
    # model= Gemini(id = "gemini-2.5-flash-lite", api_key= ""),
    model = Ollama(id= "llama3.1"),
    name = "Eval Agent",
    role= "Agent that evaluates performances of the model outputed by the Trainer Agent, and returns the appropraiate report with the right metrics. He also Can save models and save reports, by creating files, JSON, CSV, PICKEL, JOBLIB, ....",
    tools = [EvaluationTools(), FileGenerationTools(output_directory="output"), LocalFileSystemTools(target_directory="output"), FileTools(base_dir= Path("output")), save_model_with_joblib, save_model_with_pickle],
    instructions = ["You are the evaluator agent, you will recieve a model and a test set, and your role is to chose the approriate tools to evaluate based on the nature of the model you recieve and the type of task classification/regression or clustering, return a Clean Well structure report including all the necessary metrics and explanations ",
                    "When asked, you must save the report into a file in the appropriate format using the appropriate provided tools for that.",
                    "When asked you will also save the models that you reciev and evaluate, either as pickle or as joblib as specified, or as you see fit if not specified.",
                    "In case of failure in saving a file, Raise an error and inform of the problem you encountered at saving"],
    add_history_to_context = True,
    markdown = True
)


# Just To Test

