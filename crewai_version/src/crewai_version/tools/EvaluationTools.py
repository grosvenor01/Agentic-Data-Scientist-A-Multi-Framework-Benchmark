import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from crewai.tools import tool

# ------------------ Regression Tools ------------------ #

@tool("MAE Regression")
def perform_mae(x_test_path: str, y_test_path: str, model_path: str) -> str:
    """Computes Mean Absolute Error (MAE) for regression."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return f"MAE: {mae}"

@tool("RMSE Regression")
def perform_rmse(x_test_path: str, y_test_path: str, model_path: str) -> str:
    """Computes Root Mean Squared Error (RMSE) for regression."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return f"RMSE: {rmse}"

@tool("R2 Regression")
def perform_r2(x_test_path: str, y_test_path: str, model_path: str) -> str:
    """Computes R^2 Score for regression."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return f"R2: {r2}"

# ------------------ Classification Tools ------------------ #

@tool("Accuracy Classification")
def perform_accuracy(x_test_path: str, y_test_path: str, model_path: str) -> str:
    """Computes Accuracy for classification."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f"Accuracy: {accuracy}"

@tool("F1 Classification")
def perform_f1(x_test_path: str, y_test_path: str, model_path: str, average: str = "weighted") -> str:
    """Computes F1 score for classification."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average=average)
    return f"F1 Score ({average}): {f1}"

@tool("ROC-AUC Classification")
def perform_roc_auc(x_test_path: str, y_test_path: str, model_path: str, multi_class: str = "ovr") -> str:
    """Computes ROC-AUC score for classification."""
    X_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path , allow_pickle=True)
    model = joblib.load(model_path)
    y_score = model.predict_proba(X_test)
    
    # Binary case
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    # Multi-class
    if y_score.ndim == 2 and y_score.shape[1] > 2:
        roc_auc = roc_auc_score(y_test, y_score, multi_class=multi_class, average="macro")
    else:
        roc_auc = roc_auc_score(y_test, y_score)

    return f"ROC-AUC: {roc_auc}"

# ------------------ Clustering Tools ------------------ #

@tool("Silhouette Score Clustering")
def perform_silhouette(x_test_path: str, model_path: str, metric: str = "euclidean") -> str:
    """Computes Silhouette Score for clustering."""
    X_test = np.load(x_test_path, allow_pickle=True)
    model = joblib.load(model_path)
    labels = model.predict(X_test)
    score = silhouette_score(X_test, labels, metric=metric)
    return f"Silhouette Score: {score}"

# ------------------ PCA Tools ------------------ #

@tool("PCA Explained Variance")
def perform_pca_variance(model_path: str) -> str:
    """Returns explained variance ratio per PCA component."""
    model = joblib.load(model_path)
    
    # Handle pipeline
    if hasattr(model, "steps"):
        pca_est = None
        for name, step in model.steps:
            if hasattr(step, "explained_variance_ratio_"):
                pca_est = step
                break
        if pca_est is None:
            raise ValueError("No PCA step found in pipeline.")
    else:
        pca_est = model

    if not hasattr(pca_est, "explained_variance_ratio_"):
        raise ValueError("Model is not a fitted PCA model.")
    
    variance_percent = np.asarray(pca_est.explained_variance_ratio_) * 100
    variance_str = ", ".join([f"{v:.2f}%" for v in variance_percent])
    return f"Explained Variance per Component: {variance_str}"