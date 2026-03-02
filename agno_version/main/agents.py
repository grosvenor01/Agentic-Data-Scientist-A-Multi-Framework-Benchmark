from .tools import *
from .prompts import *
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from .config import Settings

settings = Settings()
analysis_agent = Agent(
    name="analysis_agent",
    description="Analyse the datasets and return a report of it based on dataset path",
    role="Analyse the datasets and return a report of it",
    #model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    #model = Gemini(id= "gemini-2.0-flash" , api_key=settings.gemini_api_key),
    model=Ollama(id="qwen2.5"),
    tools=analysis_tools,
    instructions=analysis_instruction,
    add_history_to_context=True,
)

preprocessing_agent = Agent(
    name="preprocessing_agent",
    description="Performe cleaning and preprocessing to the given data based on dataset path",
    role="Performe cleaning and preprocessing to the given data, based on knowing the classification target column",
    model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    #model = Gemini(id= "gemini-2.0-flash" , api_key=settings.gemini_api_key),
    #model=Ollama(id="qwen2.5"),
    instructions=preprocessing_instruction,
    tools=preprocessing_tools,
    add_history_to_context=True,
)

Trainer_Agent = Agent(
    model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    #model = Gemini(id= "gemini-2.0-flash" , api_key=settings.gemini_api_key),
    #model=Ollama(id="qwen2.5"),
    name= "Trainer_Agent",
    role= "Agent that trains ML models accordingly to the need of the user and input/output features specifications",
    tools = training_tools,
    instructions = training_instruction,
    markdown = True, 
    add_history_to_context=True
)

Evaluator_Agent = Agent(
    model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    #model = Gemini(id= "gemini-2.0-flash" , api_key=settings.gemini_api_key),
    #model=Ollama(id="qwen2.5"),
    name = "Eval_Agent",
    role= "Agent that evaluates performances of the model outputed by the Trainer Agent, and returns the appropraiate report with the right metrics. He also Can save models and save reports, by creating files, JSON, CSV, PICKEL, JOBLIB, ....",
    tools = evaluation_tools,
    instructions = evaluation_instrcution,
    add_history_to_context = True,
    markdown = True
)
