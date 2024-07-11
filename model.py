from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model(model_type, hyperparameters):
    if model_type == 'RandomForest':
        return RandomForestClassifier(**hyperparameters)

    elif model_type == 'LogisticRegression':
        return LogisticRegression(**hyperparameters)

    elif model_type == 'SVC':
        return SVC(**hyperparameters)

    elif model_type == 'rf1':
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier())
        ])

    else:
        raise ValueError(f"Model type {model_type} is not supported.")
