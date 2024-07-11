from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_model(model_type, hyperparameters):
    if model_type == 'RandomForest':
        return RandomForestClassifier(**hyperparameters)

    elif model_type == 'LogisticRegression':
        return LogisticRegression(**hyperparameters)

    elif model_type == 'SVC':
        return SVC(**hyperparameters)

    else:
        raise ValueError(f"Model type {model_type} is not supported.")
