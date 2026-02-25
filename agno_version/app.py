
from agno.team import Team
from agno.models.google import Gemini
from main.agent import Trainer_Agent, Evaluator_Agent, Data_Agent
from main.config import Settings

settings = Settings()
team = Team(
    name="Research Team",
    members=[Data_Agent, Trainer_Agent, Evaluator_Agent ],
    model= Gemini(id = "gemini-2.5-flash-lite", api_key= settings.gemini_api_key),
    instructions= "You are the superviser of Three agents conducting ML tasks, your goal is to make them collaborate to result in a good model to perform the desired predictions by the user."
)

team.print_response("the file 'Data/data.csv' contains breast cancer data, with their 'diagnosis' columns value either 'B' or 'M', I need you train a classification model that would help me predict the diagnosis in my future data for next times, use the provided tools to load the data, the target feature is the 'diagnosis' columns as previously mentionned. ", stream=True)
