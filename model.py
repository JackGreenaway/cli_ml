from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector


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
    
    elif model_type == "rf2":
        steps = Pipeline([
                ("scaler", StandardScaler()),
            ])

        ct = ColumnTransformer(
            [
                ("numerical", steps, make_column_selector(dtype_include=["int", "float"]))
            ]
        )

        model = Pipeline(
            [
                ("ct", ct),
                ("model", RandomForestClassifier())
            ]
        )
        
        return model

    else:
        raise ValueError(f"Model type {model_type} is not supported.")
