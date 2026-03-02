from main.MAS import MAS
from main.agents import *
import joblib , numpy as np
from sklearn.metrics import classification_report

while True:
    user_query = input("actions needed : ")
    dataset_path = input("Dataset path : ")


    response = MAS.print_response(
        f"actions needed : {user_query} , dataset path : '{dataset_path}'"
    )

