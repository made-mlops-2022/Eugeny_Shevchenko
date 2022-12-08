import os
import pandas as pd
import click
from sklearn
@click.command("train")
@click.option("--train-dir")
def train(train_dir: str):
    data = pd.read_csv(os.path.join(train_dir, "data.csv"))
    target = pd.read_csv(os.path.join(train_dir, "target.csv"))
    # do something instead
    data["features"] = 0

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"))


if __name__ == '__main__':
    preprocess()