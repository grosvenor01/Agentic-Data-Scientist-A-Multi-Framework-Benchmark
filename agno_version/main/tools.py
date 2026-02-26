import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys, traceback
import subprocess
import numpy as np 
from sklearn.model_selection import train_test_split
from agno.tools import Toolkit
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# classifcators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from services.model_savers import save_model_with_joblib , save_model_with_pickle
import pickle 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib , json

def EDA(file_path:str , output_dir:str="figs/"):
    """
    A tool to perform automated exploratory data analysis (EDA) on a CSV dataset.
    Generates summary statistics, missing values, unique counts, and visualizations.
    """
    df = pd.read_csv(file_path)

    col_info = df.dtypes.to_dict()
    missing_values = df.isna().sum().to_dict()
    unique_counts = df.nunique().to_dict()
    stats = df.describe(include='all').to_dict()

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # Visualizations
    charts = []

    # Histograms for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        chart_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(chart_path)
        plt.close()
        charts.append(chart_path)

    # Barplots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45)
        chart_path = os.path.join(output_dir, f"{col}_bar.png")
        plt.savefig(chart_path)
        plt.close()
        charts.append(chart_path)

        # Correlation heatmap (if numeric columns exist)
        if len(numeric_cols) > 1:
            plt.figure(figsize=(8,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            charts.append(heatmap_path)

        # Construct report
        report = {
            "columns_info": col_info,
            "missing_values": missing_values,
            "unique_counts": unique_counts,
            "statistics": stats,
            "charts": charts,
            "report": "Dataset structure, missing values, key patterns, correlations, and visualizations."
        }
        return report
analysis_tools = [EDA]

def run_python_script(code: str) -> str:
    filename = "excutables/code_to_run.py"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return stdout or "Execution successful (no output)."

        return f"Execution failed:\n{stderr}"

    except Exception:
        return f"Unexpected error:\n{traceback.format_exc()}"
preprocessing_tools = [run_python_script]

def dataLoader(csv_path: str, target_col: str, test_size=0.2, random_state=42, shuffle=True):
    """
        This tool splits dataframe into X_train, y_train, x_test, y_test and saves them into npy files
    """
    df = pd.read_csv(csv_path)
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert to numpy
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    np.save("part/x_train.npy", X_train)
    np.save("part/y_train.npy", y_train)
    np.save("part/x_test.npy", X_test)
    np.save("part/y_test.npy", y_test)
    return "Splits saved in directory part/x_train.npy, part/y_train.npy, part/x_test.npy, part/y_test.npy"

class MLTools(Toolkit):
    def __init__(self, **kwargs):
        tools = [
            self.performLinearRegression,
            self.performPolynomialRegression,
            self.performSVR,
            self.performGradientBoostingRegression,
            self.performRandomForestClassification,
            self.performGradientBoostingClassification,
            self.performLogisticRegressionClassification,
            self.performSVMClassification,
            self.performKMeansClustering,
            self.performPCA
        ]
        super().__init__(name="Regression_tools", tools=tools, instructions= "Use these tools to perform Machine learning tasks, such as Regression Analaysis, Classification, or clustering, chose the appropriate tool function to your use case based on the type of data you have and the user intents, for each ML model there are default haper parameters set, you can change them based on your specefic use case by passing the as arguments using the same parameters name.", **kwargs)

    def performLinearRegression(self, x_train_path, y_train_path, model_output_path="output/linear_regression_model.joblib", history_output_path="output/linear_regression_history.json"):
        """ This function performs a Linear regression
        Input: paths to training data and target features
        Output: Trained Linear regression model saved to joblib file and training history
        """
        import os
        
        x = np.load(x_train_path)
        y = np.load(y_train_path)
        x = np.asarray(x)
        y = np.asarray(y).ravel()

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        
        # Compute score on training data
        score = float(model.score(x, y))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {"model_type": "LinearRegression", "score": score, "intercept": float(model.intercept_), "coefficients": model.coef_.tolist()}
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performPolynomialRegression(self, x_train_path, y_train_path, degree=2, include_bias=True, model_output_path="output/polynomial_regression_model.joblib", history_output_path="output/polynomial_regression_history.json"):
        """Performs Polynomial Regression (PolynomialFeatures + LinearRegression).
        Input: paths to training data X, target y, polynomial degree
        Output: trained pipeline model saved to joblib file and training history
        """
        
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()     

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=include_bias),
            LinearRegression()
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        history = {"model_type": "PolynomialRegression", "score": score, "degree": degree, "include_bias": include_bias}
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performSVR(self, x_train_path, y_train_path, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale", model_output_path="output/svr_model.joblib", history_output_path="output/svr_history.json"):
        """Performs Support Vector Regression (SVR).
        Input: paths to training data X, target y, SVR hyperparameters
        Output: trained SVR pipeline model (with scaling) saved to joblib file and training history
        """
        import os
        
        # Load data from paths
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()      # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)       # make X 2D if single feature

        # SVR usually works much better with scaled features
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {"model_type": "SVR", "score": score, "kernel": kernel, "C": C, "epsilon": epsilon, "gamma": gamma}
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performGradientBoostingRegression(
        self,
        x_train_path,
        y_train_path,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42,
        model_output_path="output/gradient_boosting_regression_model.joblib",
        history_output_path="output/gradient_boosting_regression_history.json"
    ):
        """Performs Gradient Boosting Regression (sklearn).
        Input: paths to training data X, target y, GBR hyperparameters
        Output: trained GradientBoostingRegressor model saved to joblib file and training history
        """
        import os
        
        # Load data from paths
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # make X 2D if single feature

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {
            "model_type": "GradientBoostingRegressor",
            "score": score,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performRandomForestClassification(
        self,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str = None,
        y_test_path: str = None,
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42
    ):
        """Performs Random Forest Classification.
        Loads X/y from .npy paths, trains a RandomForestClassifier, saves model with joblib,
        saves training metadata ("history"), and returns a score.

        If x_test_path and y_test_path are provided: score is computed on test data.
        Otherwise: score is computed on training data.
        """

        model_output_path="output/random_forest_classification_model.joblib"
        history_output_path="output/random_forest_classification_history.json"
        # ---- Load train data ----
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path, allow_pickle=True)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()  # make y 1D

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)   # make X 2D if single feature

        # ---- Model ----
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # ---- Choose evaluation set ----
        if x_test_path is not None and y_test_path is not None:
            X_eval = np.load(x_test_path)
            y_eval = np.load(y_test_path, allow_pickle=True)

            X_eval = np.asarray(X_eval)
            y_eval = np.asarray(y_eval).ravel()

            if X_eval.ndim == 1:
                X_eval = X_eval.reshape(-1, 1)

            score = float(model.score(X_eval, y_eval))  # accuracy for classification
            score_on = "test"
        else:
            score = float(model.score(X_train, y_train))
            score_on = "train"

        # ---- Save model (joblib) ----
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)

        # ---- Save "history" (metadata) ----
        history = {
            "model_type": "RandomForestClassifier",
            "score": score,
            "score_on": score_on,
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight,
                "random_state": random_state
            }
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": score_on
        }

    def performGradientBoostingClassification(
        self,
        x_train_path,
        y_train_path,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42,
        model_output_path="output/gradient_boosting_classification_model.joblib",
        history_output_path="output/gradient_boosting_classification_history.json"
    ):
        """Performs Gradient Boosting Classification (sklearn).
        Input: paths to training data X, target y, GBC hyperparameters
        Output: trained GradientBoostingClassifier model saved to joblib file and training history
        """
        import os
        
        # Load data from paths
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # make X 2D if single feature

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {
            "model_type": "GradientBoostingClassifier",
            "score": score,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performLogisticRegressionClassification(
        self,
        x_train_path,
        y_train_path,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=42,
        model_output_path="output/logistic_regression_classification_model.joblib",
        history_output_path="output/logistic_regression_classification_history.json"
    ):
        """Performs Logistic Regression Classification.
        Input: paths to training data X, target y, LogisticRegression hyperparameters
        Output: trained LogisticRegression pipeline model (with scaling) saved to joblib file and training history
        """
        import os
        
        # Load data from paths
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # make X 2D if single feature

        # Scaling helps logistic regression converge and perform better
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state
            )
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {
            "model_type": "LogisticRegression",
            "score": score,
            "penalty": penalty,
            "C": C,
            "solver": solver,
            "max_iter": max_iter
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }

    def performSVMClassification(
        self,
        x_train_path,
        y_train_path,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        degree=3,
        probability=True,
        class_weight=None,
        random_state=42,
        model_output_path="output/svm_classification_model.joblib",
        history_output_path="output/svm_classification_history.json"
    ):
        """Performs SVM Classification (SVC).
        Input: paths to training data X, target y, SVM hyperparameters
        Output: trained SVM pipeline model (with scaling) saved to joblib file and training history
        """
        
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()  

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                probability=probability,
                class_weight=class_weight,
                random_state=random_state
            )
        )
        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        history = {
            "model_type": "SVC",
            "score": score,
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "degree": degree,
            "probability": probability
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }
    
    def performKNNClassification(
        self,
        x_train_path,
        y_train_path,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        p=2,
        model_output_path="output/knn_classification_model.joblib",
        history_output_path="output/knn_classification_history.json"
    ):
        """Performs K-Nearest Neighbors Classification.
        Input: paths to training data X, target y, KNN hyperparameters
        Output: trained KNN pipeline model (with scaling) saved to joblib file and training history
        """
        X = np.load(x_train_path)
        y = np.load(y_train_path)
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                p=p
            )
        )

        model.fit(X, y)
        
        # Compute score on training data
        score = float(model.score(X, y))
        
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        history = {
            "model_type": "KNeighborsClassifier",
            "score": score,
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "p": p
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        } 
    
    def performKMeansClustering(
        self,
        x_train_path,
        n_clusters=3,
        init="k-means++",
        max_iter=300,
        random_state=42,
        model_output_path="output/kmeans_clustering_model.joblib",
        history_output_path="output/kmeans_clustering_history.json"
    ):
        """Performs K-Means Clustering.
        Input: path to data X, Kmeans hyperparameters
        Output: trained KMeans pipeline model (with scaling) saved to joblib file and training history
        """
        X = np.load(x_train_path)
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            StandardScaler(),
            KMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                random_state=random_state,
                n_init=10
            )
        )

        model.fit(X)
        
        # Compute silhouette score
        labels = model.predict(X)
        score = float(silhouette_score(X, labels))
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        kmeans_est = model.named_steps['kmeans']
        history = {
            "model_type": "KMeans",
            "score": score,
            "n_clusters": n_clusters,
            "init": init,
            "max_iter": max_iter,
            "inertia": float(kmeans_est.inertia_) if hasattr(kmeans_est, 'inertia_') else None,
            "n_iter": int(kmeans_est.n_iter_) if hasattr(kmeans_est, 'n_iter_') else None
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }
    
    def performPCA(
        self,
        x_train_path,
        n_components=2,
        random_state=42,
        model_output_path="output/pca_model.joblib",
        history_output_path="output/pca_history.json"
    ):
        """Performs Principal Component Analysis (PCA).
        Input: path to data X
        Output: trained PCA pipeline model (with scaling) saved to joblib file and training history
        """
        import os
        
        # Load data from path
        X = np.load(x_train_path)
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = make_pipeline(
            StandardScaler(),
            PCA(
                n_components=n_components,
                random_state=random_state
            )
        )

        model.fit(X)
        
        # Get total variance explained as score
        pca_est = model.named_steps['pca']
        score = float(np.sum(pca_est.explained_variance_ratio_)) if hasattr(pca_est, 'explained_variance_ratio_') else 0.0
        
        # Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        
        # Create and save training history
        history = {
            "model_type": "PCA",
            "score": score,
            "n_components": n_components,
            "explained_variance_ratio": pca_est.explained_variance_ratio_.tolist() if hasattr(pca_est, 'explained_variance_ratio_') else None
        }
        os.makedirs(os.path.dirname(history_output_path), exist_ok=True)
        with open(history_output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        
        return {
            "model_path": model_output_path,
            "history_path": history_output_path,
            "score": score,
            "score_on": "train"
        }
training_tools = [MLTools() , dataLoader]

class EvaluationTools(Toolkit):

    def __init__(self, **kwargs):
        tools = [
            self.performMAERegression,
            self.performRMSERegression,
            self.performR2Regression,
            self.performAccuracyClassification,
            self.performF1Classification,
            self.performROCAUCClassification,
            self.performSilhouetteScoreClustering,
            self.performPCAExplainedVariancePercent
        ]
        super().__init__(name="Regression_tools", tools=tools, instructions= "Use these tools to perform Machine learning tasks, such as Regression Analaysis, Classification, or clustering, chose the appropriate tool function to your use case based on the type of data you have and the user intents, for each ML model there are default haper parameters set, you can change them based on your specefic use case by passing the as arguments using the same parameters name.", **kwargs)

    def performMAERegression(self, x_test_path, y_test_path, model_path):
        """Computes Mean Absolute Error (MAE) for regression.
        Loads model and test data, makes predictions, returns MAE as string.
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        return f"Mean Absolute Error (MAE): {mae}"

    def performRMSERegression(self, x_test_path, y_test_path, model_path):
        """Computes Root Mean Squared Error (RMSE) for regression.
        Loads model and test data, makes predictions, returns RMSE as string.
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        return f"Root Mean Squared Error (RMSE): {rmse}"

    def performR2Regression(self, x_test_path, y_test_path, model_path):
        """Computes R^2 score for regression.
        Loads model and test data, makes predictions, returns R^2 as string.
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return f"R^2 Score: {r2}"

    def performAccuracyClassification(self, x_test_path, y_test_path, model_path):
        """Computes Accuracy for classification.
        Loads model and test data, makes predictions, returns accuracy as string.
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return f"Accuracy: {accuracy}"

    def performF1Classification(self, x_test_path, y_test_path, model_path, average="weighted"):
        """Computes F1 score for classification.
        Loads model and test data, makes predictions, returns F1 as string.
        average: 'binary', 'macro', 'micro', 'weighted' (default)
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average=average)
        
        return f"F1 Score ({average}): {f1}"

    def performROCAUCClassification(self, x_test_path, y_test_path, model_path, multi_class="ovr"):
        """Computes ROC-AUC for classification.
        Loads model and test data, gets probability predictions, returns ROC-AUC as string.
        """
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path , allow_pickle=True)
        model = joblib.load(model_path)
        
        # Get probability predictions
        y_score = model.predict_proba(X_test)
        
        # If probabilities are (n_samples, 2) for binary, use positive class prob
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        
        # If multiclass
        if y_score.ndim == 2 and y_score.shape[1] > 2:
            roc_auc = roc_auc_score(y_test, y_score, multi_class=multi_class, average="macro")
        else:
            roc_auc = roc_auc_score(y_test, y_score)
        
        return f"ROC-AUC Score: {roc_auc}"

    def performSilhouetteScoreClustering(self, x_test_path, model_path, metric="euclidean"):
        """Computes Silhouette Score for clustering.
        Loads model and test data, makes cluster predictions, returns silhouette score as string.
        """
        X_test = np.load(x_test_path)
        model = joblib.load(model_path)
        
        labels = model.predict(X_test)
        silhouette = silhouette_score(X_test, labels, metric=metric)
        
        return f"Silhouette Score: {silhouette}"

    def performPCAExplainedVariancePercent(self, model_path):
        """Returns explained variance ratio (%) for each PCA component as string.
        Loads PCA model and returns variance percentages.
        """
        model = joblib.load(model_path)
        
        # Handle pipeline or standalone PCA
        if hasattr(model, "steps"):
            pca_est = None
            for name, step in model.steps:
                if hasattr(step, "explained_variance_ratio_"):
                    pca_est = step
                    break
            if pca_est is None:
                raise ValueError("No PCA step found in the provided pipeline/model.")
        else:
            pca_est = model
        
        if not hasattr(pca_est, "explained_variance_ratio_"):
            raise ValueError("Provided model does not look like a fitted PCA model.")
        
        variance_percent = (np.asarray(pca_est.explained_variance_ratio_) * 100.0)
        variance_str = ", ".join([f"{v:.2f}%" for v in variance_percent])
        
        return f"Explained Variance Ratio per Component: {variance_str}"
evaluation_tools = [EvaluationTools()]