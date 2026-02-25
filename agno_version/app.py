
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.ollama import Ollama
import joblib
from main.agent import Trainer_Agent, Evaluator_Agent, Data_Agent
from main.config import Settings

settings = Settings()
team = Team(
    name="Research Team",
    members=[Data_Agent, Trainer_Agent, Evaluator_Agent ],
    # model= Gemini(id = "gemini-2.5-flash-lite", api_key= ""), api_key= ""
    # model = Ollama(id= "llama3.1"),
    model = OpenAIChat(id = 'gpt-4.1-nano', api_key= ""),
    instructions= ["You are the superviser of Three agents conducting ML tasks, your goal is to make them collaborate to result in a good model to perform the desired predictions by the user.",
                   "Never ask for confirmation. Always proceed with the task immediately.",
                    "Use the provided tools to load data and train models.",
                    "The user has already provided all necessary information.",
                    "If you see the results are quite bad, call the Trainer agent again, I need excellent resuts, not only good",
                    "save the models into joblib or pickel files. even the bad perfoming ones. Be carefull not to override them by using the same name each time",
                    "Save the final report into a file too so I can check it out later"],
    show_members_responses= True,
    debug_mode= True
)

#response = team.print_response("the file 'Data/data.csv' is a DataSet, load it and make me a description of its content", stream=True)
#response = team.print_response("the file 'C:/Users/adilo/Code/Data/data.csv' contains breast cancer data, with their 'diagnosis' columns value either 'B' or 'M', I need you train a classification model that would help me predict the diagnosis in my future data for next times, use the provided tools to load the data, the target feature is the 'diagnosis' columns as previously mentionned. Svae results in output/ directory ", stream=True)
""" for chunk in response:
    print(type(chunk))
    print(dir(chunk)) """

#response  = Trainer_Agent.print_response("Perform Classification (RandomForest) on dataset splits in these paths: part/x_train.npy, part/y_train.npy, part/x_test.npy, part/y_test.npy", debug_mode= True)

# loading file with joblib
import numpy as np
model = joblib.load("output/random_forest_classification_model.joblib")
x_test = np.load("part/x_test.npy")
predictions = model.predict(x_test[0].reshape(1, -1))
print(predictions)
print(np.load("part/y_test.npy", allow_pickle=True)[0])
