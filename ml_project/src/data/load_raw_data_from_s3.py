import boto3
import click
import pandas as pd

from src.enities.train_pipeline_params import (
    read_training_pipeline_params,
)


@click.command(name="load_dataset")
@click.argument("config_path")
def download_data_from_s3(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    download_params = training_pipeline_params.downloading_params
    df = pd.read_csv(download_params.s3_path)
    df.to_csv(download_params.output_path+download_params.raw_filename,
              index=False)


if __name__ == "__main__":
    download_data_from_s3()
