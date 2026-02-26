import joblib , pickle

def save_model_with_joblib(model_object: object, file_path: str) -> str:
    joblib.dump(model_object, file_path)
    return f"Model saved successfully to {file_path}"


def save_model_with_pickle(model_object: object, file_path: str) -> str:
    with open(file_path, "wb") as f:
        pickle.dump(model_object, f)
    return f"Model saved successfully to {file_path}"