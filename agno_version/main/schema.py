from pydantic import BaseModel , Field
from typing import List

class AnalysisOutputSchema(BaseModel):
    report : str

class AnalysisInputSchema(BaseModel):
    dataset_path : str = Field(description="path to the dataset to return its information")

class PreprocessingOutputSchema(BaseModel):
    report : str 
    preprocessed_dataset_path : str

class PreprocessingInputSchema(BaseModel):
    task : str = Field(description="The tasks that the preprocessing should do step by step and detailled methods and operation to do")
    dataset_description : str = Field(description="The key information about dataset values and columns")
    dataset_path : str = Field(description="Path to dataset that needs preprocessing")

class TrainingOutputSchema(BaseModel):
    report : str
    trained_models_paths : str
    path_splits_folder : str

class TrainingInputSchema(BaseModel):
    preprocessed_dataset_path : str = Field(description="Path to preprocessed dataset or clean dataset")
    tasks : str = Field("Tasks and algorithmes to use for this training")

class EvaluationOutputSchema(BaseModel):
    report : str

class EvaluationInputSchema(BaseModel):
    path_splits_folder : str = Field("Path to folder that contains splits (this should be provided by the trainer agent)")
    trained_model_path : str = Field("Path to the trained model .joblib format")
    task : str = Field(description= "Task to do ")