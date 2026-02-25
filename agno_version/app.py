from main.MAS import MAS
from main.agents import analysis_agent

user_query = input("actions needed : ")
dataset_path = input("Dataset path : ")


response = MAS.run(
    f"actions needed : {user_query} , dataset path : {dataset_path}",
)

print(response.content)
print(response.metrics.to_dict()) # since i used the same model i will keep it like this and not by memeber