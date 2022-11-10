import pickle
from typing import Union

import click
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.enities.train_params import TrainingParams
from src.enities.train_pipeline_params import read_training_pipeline_params

SklearnClassifierModel = Union[
    KNeighborsClassifier,
    RandomForestClassifier
]


def train_model(
        features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> KNeighborsClassifier:
    if train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=train_params.n_neighbors,
            weights=train_params.weights,
            metric=train_params.metric
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    return model.predict(features)


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> dict:
    return classification_report(
        target,
        predicts,
        output_dict=True
    )


def create_inference_pipeline(
        model: SklearnClassifierModel, transformer: ColumnTransformer
) -> Pipeline:
    if transformer is None:
        return Pipeline([("model_handling", model)])

    return Pipeline([("feature_handling", transformer), ("model_handling", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(input: str) -> Pipeline:
    with open(input, "rb") as f:
        model = pickle.load(f)
    return model


# mlflow.sklearn.autolog()


@click.command(name="train_pipeline")
@click.argument("config_path")
@click.argument("train_path")
def train_command(config_path: str, train_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    train_params = training_pipeline_params.train_params
    feature_params = training_pipeline_params.feature_params
    train_df = pd.read_csv(train_path)
    train_target = train_df[feature_params.target_col]
    train_df = train_df.drop([feature_params.target_col], axis=1)
    model = train_model(train_df, train_target, train_params)
    serialize_model(model, training_pipeline_params.output_model_path)


if __name__ == "__main__":
    train_command()
