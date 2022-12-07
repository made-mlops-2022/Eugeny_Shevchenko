import logging

import click
import pandas as pd

from src.enities.train_pipeline_params import (
    read_training_pipeline_params,
    TrainingPipelineParams)
from src.models.train_model import (
    deserialize_model,
    predict_model,
)

logger = logging.getLogger()


def predict_pipeline(config_path: str, data_path: str,
                     pred_path: str):
    predicting_pipline_params = read_training_pipeline_params(config_path)
    return run_predict_pipeline(predicting_pipline_params, data_path, pred_path)


def run_predict_pipeline(predicting_pipeline_params: TrainingPipelineParams,
                         data_path: str,
                         pred_path: str):
    logger.info(f"__Start predicting :: params = {predicting_pipeline_params}")
    data_frame = pd.read_csv(data_path)
    if predicting_pipeline_params.feature_params.target_col in data_frame.columns:
        data_frame = data_frame.drop([predicting_pipeline_params.feature_params.target_col], axis=1)

    model_pipeline = deserialize_model(
        predicting_pipeline_params.output_model_path
    )
    logger.info(f"__Got Model :: {model_pipeline}")

    pred = predict_model(
        model_pipeline,
        data_frame
    )

    pd.DataFrame(pred).to_csv(pred_path, index=False)


@click.command()
@click.argument("config_path")
@click.argument("data_path")
@click.argument("pred_path")
def main(config_path: str, data_path, pred_path: str):
    predict_pipeline(config_path, data_path, pred_path)


if __name__ == "__main__":
    main()
