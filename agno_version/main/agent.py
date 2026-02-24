from agno.agent import Agent
from agno.models.google import Gemini
from .config import Settings
from .tools import MLTools, dataLoader

# knt nawi ndirha docstring
""" Name: 
        Role: AI Trainer Agent, this agent recieve trainig set as input, precising the target, and it will run and train the appropriate machine learning algorithms 
        Retrun Type: Machine learning model with trained parameters.
"""
Trainer_Agent = Agent(
    model = Gemini(id = "gemini-2.5-flash-lite", api_key= Settings.gemini_api_key),
    tools = [MLTools(), dataLoader],
    instructions = "You are an AI engineer agent, your will recieve preprocessed datasets, and your mission is to train models on the provided data to perform predictions. you will chose the appropriate machine learning algorithms to select based on the nature of the provided data, and user queries/ specifications.",
    add_history_to_context = True,
    markdown = True
)


