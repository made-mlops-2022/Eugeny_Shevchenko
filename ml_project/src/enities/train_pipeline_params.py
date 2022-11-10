from dataclasses import dataclass, field
from typing import Optional

import yaml
from marshmallow_dataclass import class_schema

from src.enities.download_params import DownloadParams
from src.enities.feature_params import FeatureParams
from src.enities.split_params import SplittingParams
from src.enities.train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    output_model_path: str = field(default="models/model.pkl")
    metric_path: str = field(default="models/metric.pkl")
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    mlflow_uri: str = "http://18.156.5.226/"
    mlflow_experiment: str = "inference_demo"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
