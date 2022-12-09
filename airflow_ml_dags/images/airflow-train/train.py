import os
import pickle

import click
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# os.environ["MLFLOW_TRACKING_URI"] = f"http://0.0.0.0:5000/"
@click.command("train")
@click.option("--train-dir")
@click.option("--model-dir")
def train(train_dir: str, model_dir: str):
    mlflow.set_tracking_uri("http://192.168.1.70:5000")
    with mlflow.start_run(run_name='train'):
        X = pd.read_csv(os.path.join(train_dir, 'x_train.csv'))
        y = pd.read_csv(os.path.join(train_dir, 'y_train.csv'))
        model = RandomForestClassifier()
        model.fit(X, y)

        model_params = model.get_params()
        for param in model_params:
            mlflow.log_param(param, model_params[param])

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model",
            registered_model_name='rf_model')

        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    train()
