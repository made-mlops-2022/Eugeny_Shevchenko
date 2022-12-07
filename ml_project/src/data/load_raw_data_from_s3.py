import click
import click
import pandas as pd

from src.enities.download_params import DownloadParams
from src.enities.train_pipeline_params import read_training_pipeline_params


def s3_to_csv(param: DownloadParams):
    df = pd.read_csv(param.s3_path)
    df.to_csv(param.output_path + param.raw_filename,
              index=False)


@click.command(name="load_dataset")
@click.argument("config_path")
def download_data_from_s3(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    download_params = training_pipeline_params.downloading_params
    s3_to_csv(download_params)


if __name__ == "__main__":
    download_data_from_s3()
