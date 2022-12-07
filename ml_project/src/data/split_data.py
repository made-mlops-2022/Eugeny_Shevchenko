from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from src.enities.split_params import SplittingParams
from src.enities.train_pipeline_params import read_training_pipeline_params


def split_train_test_data(
        data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, test_data


def get_datapath_with_prefix(data_path: str) -> tuple[str, str]:
    data_path = data_path.split('/')
    filename = data_path.pop()
    _ = data_path.pop()
    test_data_path = '/'.join(data_path + ['interim', 'test_' + filename])
    train_data_path = '/'.join(data_path + ['interim', 'train_' + filename])
    return train_data_path, test_data_path


@click.command()
@click.argument('config_path')
@click.argument('data_path')
def main(config_path: str, data_path: str):
    train_params = read_training_pipeline_params(config_path)
    split_params = train_params.splitting_params
    data = pd.read_csv(data_path)
    train_data, test_data = split_train_test_data(data, split_params)
    train_data_path, test_data_path = get_datapath_with_prefix(data_path)
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)


if __name__ == "__main__":
    main()
