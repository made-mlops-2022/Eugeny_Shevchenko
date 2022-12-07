import json
import logging
import os.path
import sys

import click
import mlflow
import pandas as pd

from features.build_features import (
    build_transformer
)
from src.data.load_raw_data_from_s3 import s3_to_csv
from src.data.split_data import split_train_test_data
from src.enities.train_pipeline_params import read_training_pipeline_params
from src.models.train_model import (train_model, predict_model,
                                    evaluate_model, serialize_model,
                                    create_inference_pipeline)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    if training_pipeline_params.use_mlflow:

        mlflow.set_tracking_uri(training_pipeline_params.mlflow_uri)
        mlflow.set_experiment(training_pipeline_params.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    download_param = training_pipeline_params.downloading_params
    split_param = training_pipeline_params.splitting_params
    feature_param = training_pipeline_params.feature_params
    raw_data_path = download_param.output_path + download_param.raw_filename
    if not os.path.exists(raw_data_path):
        logger.info(f"load dataset from s3 to {download_param.output_path}")
        s3_to_csv(download_param)

    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = pd.read_csv(download_param.output_path + download_param.raw_filename)
    logger.info(f"data.shape is {data.shape}")

    logger.info(f"split dataset to train and test ...")
    train_df, test_df = split_train_test_data(data, split_param)
    test_target = test_df[feature_param.target_col]
    train_target = train_df[feature_param.target_col]
    train_df = train_df.drop([feature_param.target_col], axis=1)
    test_df = test_df.drop([feature_param.target_col], axis=1)
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"test_df.shape is {test_df.shape}")

    if not (training_pipeline_params.train_params.scaler is None):
        transformer = build_transformer(training_pipeline_params.feature_params)
        train_df = transformer.fit_transform(train_df)
    else:
        transformer = None

    model = train_model(
        train_df, train_target, training_pipeline_params.train_params
    )

    inference_pipeline = create_inference_pipeline(model, transformer)
    y_pred = predict_model(
        inference_pipeline,
        test_df
    )

    metrics = evaluate_model(
        y_pred,
        test_target
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        model, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
