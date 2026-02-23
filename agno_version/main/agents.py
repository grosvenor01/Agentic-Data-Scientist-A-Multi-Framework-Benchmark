from .tools import *
from .prompts import *
from agno.agent import Agent
from agno.models.google import Gemini
from .config import Settings

settings = Settings()
analysis_agent = Agent(
    name="analysis_agent",
    role="Analyse the datasets and return a report of it",
    model=Gemini(id="gemini-2.5-flash-lite" , api_key=settings.gemini_api_key),
    tools=analysis_tools,
    instructions=analysis_instruction,
    add_history_to_context=True,
    num_history_runs=5
)

preprocessing_agent = Agent(
    name="preprocessing_agent",
    role="Performe cleaning and preprocessing to the given data",
    model=Gemini(id="gemini-2.5-flash-lite" , api_key=settings.gemini_api_key),
    tools=preprocessing_tools,
    instructions=preprocessing_instruction,
    add_history_to_context=True,
    num_history_runs=5
)

training_agent = Agent(
    name="training_agent",
    role="Train different machine learning models ",
    model=Gemini(id="gemini-2.5-flash-lite" , api_key=settings.gemini_api_key),
    tools=training_tools,
    instructions=training_instruction,
    add_history_to_context=True,
    num_history_runs=5
)

evaluation_agent = Agent(
    name="evaluation_agent",
    role="Evaluate machine learning models and test",
    model=Gemini(id="gemini-2.5-flash-lite" , api_key=settings.gemini_api_key),
    tools=evaluation_tools,
    instructions=evaluation_instrcution,
    add_history_to_context=True,
    num_history_runs=5
)


