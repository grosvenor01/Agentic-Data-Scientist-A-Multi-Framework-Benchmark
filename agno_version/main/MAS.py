from agno.team.team import Team
from .agents import *
from agno.models.google import Gemini
from .prompts import supervisor_prompt
MAS = Team(
    name = "Multi-agent-system",
    model=Gemini(id="gemini-2.5-flash-lite"),
    members=[analysis_agent , preprocessing_agent],
    instructions=supervisor_prompt,
    show_members_responses=True,
    markdown=True
)