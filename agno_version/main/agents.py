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
    description="Data Analysis Agent: Examines raw datasets to understand structure, data types, missing values, and distribution patterns. Generates comprehensive EDA reports with visualizations.",
    role="ANALYST - Receives dataset path → Performs comprehensive analysis → Returns JSON report with dataset structure, data quality issues, patterns, and preprocessing recommendations",
    model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    tools=analysis_tools,
    instructions=analysis_instruction,
    markdown=True,
)

preprocessing_agent = Agent(
    name="preprocessing_agent",
    description="Data Preprocessing Agent: Cleans, transforms, and normalizes raw data based on analysis recommendations. Handles missing values, outliers, encoding, and feature scaling.",
    role="DATA ENGINEER - Receives raw dataset path + analysis report → Generates preprocessing code → Executes transformation → Returns path to cleaned CSV in datasets/results/",
    model=OpenAIChat(id="gpt-4o-mini" , api_key=settings.openai_api_key),
    instructions=preprocessing_instruction,
    tools=preprocessing_tools,
    markdown=True,
)

Trainer_Agent = Agent(
    model = OpenAIChat(id ='gpt-4.1-nano', api_key=settings.openai_api_key),
    name= "Trainer_Agent",
    description="Model Training Agent: Splits preprocessed data and trains appropriate ML models (regression, classification, clustering). Selects best algorithms based on task and data characteristics.",
    role="ML ENGINEER - Receives preprocessed dataset path + target column + task type → Splits data (train/test) → Trains selected model(s) → Returns model_path, history_path, and train_score",
    tools = training_tools,
    instructions = training_instruction,
    markdown = True
)

Evaluator_Agent = Agent(
    model = OpenAIChat(id ='gpt-4.1-nano', api_key=settings.openai_api_key),
    name = "Evaluator_Agent",
    description="Model Evaluation Agent: Assesses trained model performance using appropriate metrics (accuracy, F1, ROC-AUC for classification; MAE, RMSE, R² for regression; Silhouette score for clustering).",
    role="QUALITY ASSANCER - Receives model_path + test_data paths → Detects task type → Runs appropriate metrics → Returns structured evaluation_report with performance_score and interpretation",
    tools = evaluation_tools,
    instructions = evaluation_instruction,
    markdown = True
)

