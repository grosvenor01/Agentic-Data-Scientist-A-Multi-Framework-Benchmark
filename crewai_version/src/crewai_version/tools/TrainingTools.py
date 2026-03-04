from crewai.tools import tool
import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# ----------------- Data Loader -----------------
@tool("Dataset spliter into train test splits")
def dataLoader(csv_path: str, target_col: str, test_size: float = 0.2, random_state: int = 42, shuffle: bool = True) -> dict:
    """This tool splits dataframe into X_train, y_train, x_test, y_test and saves them into npy files and then returns there paths
    csv_path : the path to the preprocessed CSV file (relative or absolute)
    target_col : the name of the target column
    """
    part_dir = "part"
    os.makedirs(part_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    X = np.asarray(df.drop(columns=[target_col]))
    y = np.asarray(df[target_col]).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    paths = {}
    for name, arr in zip(["x_train", "y_train", "x_test", "y_test"], [X_train, y_train, X_test, y_test]):
        path = os.path.join(part_dir, f"{name}.npy")
        np.save(path, arr)
        paths[name + "_path"] = path
    paths["message"] = "Data splits saved successfully"
    return paths

# ----------------- Linear Regression -----------------
@tool("Linear Regression")
def performLinearRegression(x_train_path: str, y_train_path: str, model_name: str = "linear_regression") -> dict:
    """ This function performs a Linear regression
        Input: paths to training data and target features
        Output: Trained Linear regression model saved to joblib file and training history
        """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "LinearRegression", "score": score, "intercept": float(model.intercept_), "coefficients": model.coef_.tolist()}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)

    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- Polynomial Regression -----------------
@tool("Polynomial Regression")
def performPolynomialRegression(x_train_path: str, y_train_path: str, degree: int = 2, include_bias: bool = True, model_name: str = "polynomial_regression") -> dict:
    """Performs Polynomial Regression (PolynomialFeatures + LinearRegression).
        Input: paths to training data X, target y, polynomial degree
        Output: trained pipeline model saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=include_bias), LinearRegression())
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "PolynomialRegression", "score": score, "degree": degree, "include_bias": include_bias}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- SVR -----------------
@tool("SVR")
def performSVR(x_train_path: str, y_train_path: str, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1, gamma: str = "scale", model_name: str = "svr") -> dict:
    """Performs Support Vector Regression (SVR).
        Input: paths to training data X, target y, SVR hyperparameters
        Output: trained SVR pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "SVR", "score": score, "kernel": kernel, "C": C, "epsilon": epsilon, "gamma": gamma}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- Gradient Boosting Regression -----------------
@tool("Gradient Boosting Regression")
def performGradientBoostingRegression(x_train_path: str, y_train_path: str, n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, subsample: float = 1.0, random_state: int = 42, model_name: str = "gradient_boosting_regression") -> dict:
    """Performs Gradient Boosting Regression (sklearn).
        Input: paths to training data X, target y, GBR hyperparameters
        Output: trained GradientBoostingRegressor model saved to joblib file and training history
        """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=random_state)
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "GradientBoostingRegressor", "score": score, "n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- Gradient Boosting Classification -----------------
@tool("Gradient Boosting Classification")
def performGradientBoostingClassification(x_train_path: str, y_train_path: str, n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, subsample: float = 1.0, random_state: int = 42, model_name: str = "gradient_boosting_classification") -> dict:
    """Performs Random Forest Classification.
        Loads X/y from .npy paths, trains a RandomForestClassifier, saves model with joblib,
        saves training metadata ("history"), and returns a score.

        If x_test_path and y_test_path are provided: score is computed on test data.
        Otherwise: score is computed on training data.
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=random_state)
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "GradientBoostingClassifier", "score": score, "n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- Logistic Regression Classification -----------------
@tool("Logistic Regression Classification")
def performLogisticRegressionClassification(x_train_path: str, y_train_path: str, penalty: str = "l2", C: float = 1.0, solver: str = "lbfgs", max_iter: int = 1000, random_state: int = 42, model_name: str = "logistic_regression_classification") -> dict:
    """Performs Logistic Regression Classification.
        Input: paths to training data X, target y, LogisticRegression hyperparameters
        Output: trained LogisticRegression pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, random_state=random_state))
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "LogisticRegression", "score": score, "penalty": penalty, "C": C, "solver": solver, "max_iter": max_iter}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- SVM Classification -----------------
@tool("SVM Classification")
def performSVMClassification(x_train_path: str, y_train_path: str, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", degree: int = 3, probability: bool = True, random_state: int = 42, model_name: str = "svm_classification") -> dict:
    """Performs SVM Classification (SVC).
        Input: paths to training data X, target y, SVM hyperparameters
        Output: trained SVM pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=probability, random_state=random_state))
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "SVC", "score": score, "kernel": kernel, "C": C, "gamma": gamma, "degree": degree, "probability": probability}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- KNN Classification -----------------
@tool("KNN Classification")
def performKNNClassification(x_train_path: str, y_train_path: str, n_neighbors: int = 5, weights: str = "uniform", algorithm: str = "auto", p: int = 2, model_name: str = "knn_classification") -> dict:
    """Performs K-Nearest Neighbors Classification.
        Input: paths to training data X, target y, KNN hyperparameters
        Output: trained KNN pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p))
    model.fit(X, y)
    score = float(model.score(X, y))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "KNeighborsClassifier", "score": score, "n_neighbors": n_neighbors, "weights": weights, "algorithm": algorithm, "p": p}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- KMeans Clustering -----------------
@tool("KMeans Clustering")
def performKMeansClustering(x_train_path: str, n_clusters: int = 3, init: str = "k-means++", max_iter: int = 300, random_state: int = 42, model_name: str = "kmeans_clustering") -> dict:
    """Performs K-Means Clustering.
        Input: path to data X, Kmeans hyperparameters
        Output: trained KMeans pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state))
    model.fit(X)
    labels = model.predict(X)
    score = float(silhouette_score(X, labels))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    kmeans_est = model.named_steps['kmeans']
    history = {"model_type": "KMeans", "score": score, "n_clusters": n_clusters, "init": init, "max_iter": max_iter, "inertia": float(kmeans_est.inertia_), "n_iter": int(kmeans_est.n_iter_)}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

@tool("Random Forest Classification")
def performRandomForestClassification(
    x_train_path: str,
    y_train_path: str,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int = None,
    random_state: int = 42,
    model_name: str = "random_forest_classification"
) -> dict:
    """
    Train a Random Forest Classifier and save model + training history.
    """
    from sklearn.ensemble import RandomForestClassifier
    X = np.load(x_train_path, allow_pickle=True)
    y = np.load(y_train_path, allow_pickle=True).ravel()
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    ))
    model.fit(X, y)
    score = float(model.score(X, y))

    # Save outputs
    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    rf_est = model.named_steps['randomforestclassifier']
    history = {
        "model_type": "RandomForestClassifier",
        "score": score,
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "feature_importances": rf_est.feature_importances_.tolist()
    }
    with open(history_path, "w") as f: json.dump(history, f, indent=2)

    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}

# ----------------- PCA -----------------
@tool("PCA")
def performPCA(x_train_path: str, n_components: int = 2, random_state: int = 42, model_name: str = "pca") -> dict:
    """Performs Principal Component Analysis (PCA).
        Input: path to data X
        Output: trained PCA pipeline model (with scaling) saved to joblib file and training history
    """
    X = np.load(x_train_path, allow_pickle=True)
    if X.ndim == 1: X = X.reshape(-1, 1)

    model = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=random_state))
    model.fit(X)
    pca_est = model.named_steps['pca']
    score = float(np.sum(pca_est.explained_variance_ratio_))

    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    joblib.dump(model, model_path)

    history = {"model_type": "PCA", "score": score, "n_components": n_components, "explained_variance_ratio": pca_est.explained_variance_ratio_.tolist()}
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return {"model_path": model_path, "history_path": history_path, "score": score, "score_on": "train"}