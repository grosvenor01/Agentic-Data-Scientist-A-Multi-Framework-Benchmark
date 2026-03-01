from agno.team.team import Team
from .agents import *
from agno.models.openai import OpenAIChat
from .prompts import supervisor_prompt
from main.config import Settings
from agno.db.sqlite import SqliteDb

settings = Settings()
db = SqliteDb(db_file="tmp/data.db")
MAS = Team(
    name = "Multi-agent-system",
    model=OpenAIChat(id="gpt-4o-mini" , api_key=settings.openai_api_key),
    members=[analysis_agent , preprocessing_agent , Trainer_Agent , Evaluator_Agent],
    instructions=supervisor_prompt,
    show_members_responses=True,
    add_history_to_context=True,
    num_history_messages=10,
    max_iterations=10,
    db=db,
)