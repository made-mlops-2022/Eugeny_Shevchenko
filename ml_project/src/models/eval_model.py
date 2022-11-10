import json

import click
import mlflow
import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params
from src.models.train_model import (deserialize_model,
                                    predict_model,
                                    evaluate_model)


@click.command()
@click.argument("config_path")
@click.argument("test_path")
def eval_command(config_path: str, test_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    feature_params = training_pipeline_params.feature_params
    model = deserialize_model(training_pipeline_params.output_model_path)
    test_df = pd.read_csv(test_path)
    test_target = test_df[feature_params.target_col]
    test_df = test_df.drop([feature_params.target_col], axis=1)
    y_pred = predict_model(
        model,
        test_df
    )

    metrics = evaluate_model(
        y_pred,
        test_target
    )
    mlflow.log_metrics(metrics["1"], 0)

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == "__main__":
    eval_command()
