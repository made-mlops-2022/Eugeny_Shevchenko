import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.enities.feature_params import FeatureParams


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return num_pipeline


def build_categirical_pipeline() -> Pipeline:
    cat_pipeline = Pipeline([
        ("minmax", MinMaxScaler()),
    ])
    return cat_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
            (
                "categorical_pipeline",
                build_categirical_pipeline(),
                params.categorical_features,
            ),
        ]
    )
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)
