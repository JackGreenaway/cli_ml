import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
import joblib
import os
import random
import string

def load_data(data_path):
    # For simplicity, we'll use the Iris dataset from sklearn
    # In practice, you would load your dataset from the data_path
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
