import argparse
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import build_model
from utils import load_data, save_model, generate_unique_model_name, increment_experiment_number

def main(args):
    # Increment and retrieve experiment number
    # experiment_number = increment_experiment_number('src/experiment_counter.txt')
    experiment_number = 1

    # Start MLflow run
    with mlflow.start_run():
        # Log experiment description
        mlflow.set_tag("description", args.description)
        mlflow.set_tag("dataset", args.dataset)

        # Load data
        X_train, X_test, y_train, y_test = load_data(args.data_path)

        # Log parameters
        mlflow.log_params(args.hyperparameters)

        # Build model
        model = build_model(args.model_type, args.hyperparameters)

        # log model type
        mlflow.log_param("model", type(model).__name__)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "f1": f1_score(y_test, y_pred, average="macro"),
        }
        print(f"Model accuracy: {metrics['accuracy']:.2f}")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Generate unique model name
        model_name = generate_unique_model_name(args.model_type, experiment_number)

        # Save model
        model_path = os.path.join(args.model_dir, model_name)
        save_model(model, model_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--data_path', type=str, default='data/iris.csv', help='Path to the dataset')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save the trained model')
    parser.add_argument('--model_type', type=str, default='RandomForest', help='Type of model to train')
    parser.add_argument('--hyperparameters', type=dict, default={}, help='Hyperparameters for the model')
    parser.add_argument('-des', '--description', type=str, default='', help='Description of the experiment')
    parser.add_argument('-ds', '--dataset', type=str, default='', help='selected ds')

    args = parser.parse_args()
    main(args)
