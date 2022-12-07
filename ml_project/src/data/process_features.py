import logging
import pickle

import click
import pandas as pd

from src.enities.train_pipeline_params import read_training_pipeline_params
from src.features.build_features import build_transformer

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('process data')


def feature_transform(X: pd.DataFrame, y: pd.Series, feature_params, filename):
    transformer = build_transformer(feature_params)
    transformer.fit(X, y)
    logger.info('Save encoder...')
    with open(feature_params.transformer_path, 'wb') as file:
        pickle.dump(transformer, file)
    x_transform = pd.DataFrame(transformer.transform(X))
    x_to_dump = pd.concat([x_transform, y], axis=1)
    x_to_dump.to_csv(feature_params.processed_path + filename, index=False)
    logger.info('Preprocess finished')
    return x_transform


@click.command(name="process_features")
@click.argument('config_path')
@click.argument('train_data_path')
@click.argument('test_data_path')
def process_features(config_path, train_data_path, test_data_path):
    train_params = read_training_pipeline_params(config_path)
    feature_params = train_params.feature_params
    logger.info('Begin preprocessing...')
    for path in [train_data_path, test_data_path]:
        interim_data_path = path
        filename = path.split('/').pop()

        df = pd.read_csv(interim_data_path)
        x_input = df.drop(columns=[feature_params.target_col])
        y_input = df[feature_params.target_col]
        feature_transform(x_input, y_input, feature_params, filename)


if __name__ == "__main__":
    process_features()
