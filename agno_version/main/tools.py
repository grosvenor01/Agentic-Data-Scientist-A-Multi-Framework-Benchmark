from agno.tools import Toolkit
from typing import List, Any

# Data
from sklearn.model_selection import train_test_split

# Regressors
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


# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def dataLoader(csv_path):
    """Reads a CSV file and returns a pandas DataFrame"""
    df = pd.read_csv(csv_path)
    return df


def dataToFeatures(df, target_col, test_size=0.2, random_state=42, shuffle=True):
    """
    Splits dataframe into train/test sets.
    
    Input:
        df : pandas DataFrame
        target_col : name of target column
        test_size : proportion for test split
        random_state : reproducibility
        shuffle : whether to shuffle before split
    
    Output:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """

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

    return X_train, X_test, y_train, y_test

class MLTools(Toolkit):
    # Regression models
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

    def performLinearRegression(x,y):
        """ This function performs a Linear regression
        Input: training data and target feautures
        Output: Trained Linear regression model
        
        """

        x = np.asarray(x)
        y = np.asarray(y).ravel()      # make y 1D

        if x.ndim == 1:
            x = x.reshape(-1, 1)       # make x 2D if user passed a single feature

        model = LinearRegression()
        model.fit(x,y)
        return model


    def performPolynomialRegression(X, y, degree=2, include_bias=True):
        """Performs Polynomial Regression (PolynomialFeatures + LinearRegression).
        Input: training data X, target y, polynomial degree
        Output: trained pipeline model
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()      # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)       # make X 2D if single feature

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=include_bias),
            LinearRegression()
        )
        model.fit(X, y)
        return model

    def performSVR(X, y, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"):
        """Performs Support Vector Regression (SVR).
        Input: training data X, target y, SVR hyperparameters
        Output: trained SVR pipeline model (with scaling)
        """
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
        return model

    def performGradientBoostingRegression(
        X, y,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    ):
        """Performs Gradient Boosting Regression (sklearn).
        Input: training data X, target y, GBR hyperparameters
        Output: trained GradientBoostingRegressor model
        """
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
        return model



    # Classification models

    def performRandomForestClassification(
        X, y,
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42
    ):
        """Performs Random Forest Classification.
        Input: training data X, target y, RandomForest hyperparameters
        Output: trained RandomForestClassifier model
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # make X 2D if single feature

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
        model.fit(X, y)
        return model

    def performGradientBoostingClassification(
        X, y,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    ):
        """Performs Gradient Boosting Classification (sklearn).
        Input: training data X, target y, GBC hyperparameters
        Output: trained GradientBoostingClassifier model
        """
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
        return model


    def performLogisticRegressionClassification(
        X, y,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=42
    ):
        """Performs Logistic Regression Classification.
        Input: training data X, target y, LogisticRegression hyperparameters
        Output: trained LogisticRegression pipeline model (with scaling)
        """
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
        return model

    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def performSVMClassification(
        X, y,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        degree=3,
        probability=True,
        class_weight=None,
        random_state=42
    ):
        """Performs SVM Classification (SVC).
        Input: training data X, target y, SVM hyperparameters
        Output: trained SVM pipeline model (with scaling)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # make y 1D

        if X.ndim == 1:
            X = X.reshape(-1, 1)   # make X 2D if single feature

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
        return model
    
    def performKNNClassification(
        X, y,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        p=2
    ):
        """Performs K-Nearest Neighbors Classification.
        Input: training data X, target y, KNN hyperparameters
        Output: trained KNN pipeline model (with scaling)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # KNN is distance-based → scaling is important
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                p=p  # p=2 Euclidean, p=1 Manhattan
            )
        )

        model.fit(X, y)
        return model

    # Clustering 
    def performKMeansClustering(
        X,
        n_clusters=3,
        init="k-means++",
        max_iter=300,
        random_state=42
    ):
        """Performs K-Means Clustering.
        Input: data X, Kmeans hyperparameters
        Output: trained KMeans pipeline model (with scaling)
        """
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
        return model
    

    def performPCA(
        X,
        n_components=2,
        random_state=42
    ):
        """Performs Principal Component Analysis (PCA).
        Input: data X
        Output: trained PCA pipeline model (with scaling)
        """
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
        return model



class EvaluationTools(Toolkit):
    # Regression models
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

    def performMAERegression(y_true, y_pred):
        """Computes Mean Absolute Error (MAE) for regression."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return mean_absolute_error(y_true, y_pred)


    # ----------------------------
    # Regression metrics
    # ----------------------------


    def performMAERegression(y_true, y_pred):
        """Computes Mean Absolute Error (MAE) for regression."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return mean_absolute_error(y_true, y_pred)

    def performRMSERegression(y_true, y_pred):
        """Computes Root Mean Squared Error (RMSE) for regression."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def performR2Regression(y_true, y_pred):
        """Computes R^2 score for regression."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return r2_score(y_true, y_pred)


    # ----------------------------
    # Classification metrics
    # ----------------------------
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.metrics import silhouette_score

    def performAccuracyClassification(y_true, y_pred):
        """Computes Accuracy for classification."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return accuracy_score(y_true, y_pred)

    def performF1Classification(y_true, y_pred, average="binary"):
        """Computes F1 score for classification.
        average: 'binary' (default), 'macro', 'micro', 'weighted'
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return f1_score(y_true, y_pred, average=average)

    def performROCAUCClassification(y_true, y_score, multi_class="ovr"):
        """Computes ROC-AUC for classification.
        y_score: probabilities or decision scores.
        - binary: shape (n_samples,) or (n_samples, 2) -> we will use positive class if 2D
        - multiclass: shape (n_samples, n_classes)
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score)

        # If probabilities are passed as (n_samples, 2) for binary, use positive class prob
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

        # If multiclass, sklearn needs multi_class + usually average='macro' or 'weighted'
        if y_score.ndim == 2 and y_score.shape[1] > 2:
            return roc_auc_score(y_true, y_score, multi_class=multi_class, average="macro")

        return roc_auc_score(y_true, y_score)


    # ----------------------------
    # Clustering metric
    # ----------------------------


    def performSilhouetteScoreClustering(X, labels, metric="euclidean"):
        """Computes Silhouette Score for clustering.
        Input: X data, cluster labels
        """
        X = np.asarray(X)
        labels = np.asarray(labels).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return silhouette_score(X, labels, metric=metric)


    # ----------------------------
    # PCA metric
    # ----------------------------
    def performPCAExplainedVariancePercent(pca_model):
        """Returns explained variance ratio (%) for each PCA component.
        Works for PCA in a pipeline too (pass the PCA estimator itself).
        """
        if hasattr(pca_model, "steps"):
            # If a Pipeline was passed, grab the PCA step
            pca_est = None
            for name, step in pca_model.steps:
                if hasattr(step, "explained_variance_ratio_"):
                    pca_est = step
                    break
            if pca_est is None:
                raise ValueError("No PCA step found in the provided pipeline/model.")
            pca_model = pca_est

        if not hasattr(pca_model, "explained_variance_ratio_"):
            raise ValueError("Provided model does not look like a fitted PCA model.")

        return (np.asarray(pca_model.explained_variance_ratio_) * 100.0)