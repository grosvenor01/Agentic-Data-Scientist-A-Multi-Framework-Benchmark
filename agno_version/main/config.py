from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings): 
    gemini_api_key : str = Field(alias="GEMINI_API_KEY")
    openai_api_key : str = Field(alias="OPENAI_API_KEY")
    model_config = {
        "env_file" : ".env",
        "env_file_encoding" : "utf8" 
    }



    