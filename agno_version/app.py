from main.MAS import MAS
from main.agents import analysis_agent

user_query = input("actions needed : ")
dataset_path = input("Dataset path : ")

analysis_agent.print_response(
    f"actions needed : {user_query} , dataset path : {dataset_path}", 
    stream=True
)
"""MAS.print_response(
    f"actions needed : {user_query} , dataset path : {dataset_path}",
    stream=True
)"""