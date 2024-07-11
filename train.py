import argparse
import joblib
import os, json
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import build_model
from utils import load_data, save_model

def main(args):

    # Start MLflow run
    with mlflow.start_run():
        # Log experiment description
        mlflow.set_tag("description", args.description)
        mlflow.set_tag("dataset", args.dataset)

        # Load data
        X_train, X_test, y_train, y_test = load_data(args.data_path)

        # Log parameters
        # mlflow.log_params(args.hyperparameters)

        # Build model
        model = build_model(args.model_type, args.hyperparameters)
        
        model_name = type(model["model"]).__name__ + "_" + mlflow.active_run().info.run_id[-3:]
        json_filename = f"{model_name}_params.json"
        json_path = 'model_parameters/' + json_filename  # Example path, adjust as needed
        
        with open(json_path, 'w') as f:
            json.dump({k: str(v) for k, v in model.get_params().items()}, f, indent=4)

        mlflow.log_param("pipeline_steps", model.named_steps)
        mlflow.log_artifact(json_path, artifact_path=json_filename)

        # log model type
        model_type = type(model).__name__
        if model_type == "Pipeline":
            mlflow.log_param("model", type(model["model"]).__name__)
        else:
            mlflow.log_param("model", model_type)

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

        # Save model
        model_path = os.path.join(args.model_dir, f"{model_name}.joblib")
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
