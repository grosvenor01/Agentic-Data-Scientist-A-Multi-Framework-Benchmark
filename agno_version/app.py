from main.MAS import MAS
from main.agents import analysis_agent
import joblib , numpy as np
from sklearn.metrics import classification_report
user_query = input("actions needed : ")
dataset_path = input("Dataset path : ")


response = MAS.print_response(
    f"actions needed : {user_query} , dataset path : {dataset_path}"
)
