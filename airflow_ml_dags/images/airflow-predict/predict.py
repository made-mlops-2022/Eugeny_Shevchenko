import os
import pickle

import click
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))

    with open(model_dir, 'rb') as f:
        model = pickle.load(f)
    X["predict"] = model.predict(X)
    os.makedirs(output_dir, exist_ok=True)
    X["predict"].to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
