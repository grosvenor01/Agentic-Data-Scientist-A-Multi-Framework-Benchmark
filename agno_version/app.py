
from main.agent import Trainer_Agent

model = Trainer_Agent.print_response("the file 'Data/data.csv' contains breast cancer data, with their 'diagnosis' columns value either 'B' or 'M', I need you train a classification model that would help me predict the diagnosis in my future data for next times, use the provided tools to load the data, the target feature is the 'diagnosis' columns as previously mentionned. ", stream=True)

# all params (works for Pipeline too)
print("~"*80)
print("Returned model Results")
print("~"*80)
params = model.get_params()
print(params)

# nicer: only the top-level step names if it's a Pipeline
print(model)
print("~"*80)