from agno.team.team import Team
from .agents import *
from agno.models.openai import OpenAIChat
from .prompts import supervisor_prompt
from main.config import Settings

settings = Settings()
MAS = Team(
    name = "Multi-agent-system",
    model=OpenAIChat(id="gpt-4.1-nano" , api_key=settings.openai_api_key),
    members=[training_agent, evaluation_agent],
    instructions=supervisor_prompt,
    show_members_responses=True,
    markdown=True,
    store_member_responses=True
)