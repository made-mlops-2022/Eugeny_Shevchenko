import click
import numpy as np
import pandas as pd
from sklearn import datasets


@click.command('generate')
@click.option('--output-dir', type=click.Path(),
              help='Path to store train data')
def generate_data(output_dir: str):
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    saved_df = df.sample(frac=0.7)
    saved_data = saved_df.drop(["target"], axis=1)
    saved_target = saved_df["target"]
    saved_data.to_csv(output_dir + "/data.csv", index=False)
    saved_target.to_csv(output_dir + "/target.csv", index=False)


if __name__ == '__main__':
    generate_data()
