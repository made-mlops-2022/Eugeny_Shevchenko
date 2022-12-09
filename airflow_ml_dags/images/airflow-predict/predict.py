import os
import pickle

import click
import pandas as pd
import mlflow


@click.command("predict")
@click.option("--input-dir")
# @click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, output_dir: str):
    mlflow.set_tracking_uri("http://192.168.1.70:5000")
    # fetch latest production model
    model = mlflow.pyfunc.load_model(
        model_uri='models:/rf_model/Production'
    )
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))

    X["predict"] = model.predict(X)
    os.makedirs(output_dir, exist_ok=True)
    X["predict"].to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
