from agno.agent import Agent
from agno.models.google import Gemini
from .config import Settings
from .tools import MLTools, EvaluationTools, dataLoader, dataToFeatures

# knt nawi ndirha docstring
""" Name: 
        Role: AI Trainer Agent, this agent recieve trainig set as input, precising the target, and it will run and train the appropriate machine learning algorithms 
        Retrun Type: Machine learning model with trained parameters.
"""
settings = Settings()
Data_Agent = Agent(
    model = Gemini(id = "gemini-2.5-flash-lite", api_key= settings.gemini_api_key),
    tools = [dataLoader, dataToFeatures],
    instructions = "You Load the dataset, split it into Train and test sets, recognize and extract target feature based on your understanding of user query, and transform data to numpy arrays ",
    add_history_to_context = True,
    markdown = True
)
Trainer_Agent = Agent(
    model = Gemini(id = "gemini-2.5-flash-lite", api_key= settings.gemini_api_key),
    tools = [MLTools()],
    instructions = "You are an AI engineer agent, your will recieve preprocessed datasets, and your mission is to train models on the provided data to perform predictions. you will chose the appropriate machine learning algorithms to select based on the nature of the provided data, and user queries/ specifications.",
    add_history_to_context = True,
    markdown = True
)

Evaluator_Agent = Agent(
    model= Gemini(id = "gemini-2.5-flash-lite", api_key= settings.gemini_api_key),
    tools = [EvaluationTools()],
    instructions = "You are the evaluator agent, you will recieve a model and a test set, and your role is to chose the approriate tools to evaluate based on the nature of the model you recieve and the type of task classification/regression or clustering, return a Clean Well structure report including all the necessary metrics and explanations ",
    add_history_to_context = True,
    markdown = True
)
