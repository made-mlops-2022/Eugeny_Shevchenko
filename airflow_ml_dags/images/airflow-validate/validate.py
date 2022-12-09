import json
import os
import pickle

import click
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command('validate')
@click.option('--input-dir', type=click.Path(),
              help='Path to splitted data')
@click.option('--model-dir', type=click.Path(),
              help='Path to model')
@click.option('--output-dir', type=click.Path(),
              help='Path to store metrics')
def validate(input_dir: str, model_dir: str, output_dir: str):
    mlflow.set_tracking_uri("http://192.168.1.70:5000")
    model_name = os.listdir(model_dir)[0]
    run_id = model_name[9:-4]
    with mlflow.start_run(run_id=run_id):

        X = pd.read_csv(os.path.join(input_dir, 'x_val.csv'))
        y = pd.read_csv(os.path.join(input_dir, 'y_val.csv'))

        with open(os.path.join(model_dir, model_name), 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(X)

        metrics = {'accuracy': accuracy_score(y, y_pred),
                   'f1_score': f1_score(y, y_pred, average='macro')}
        for metric in metrics:
            mlflow.log_metric(metric, metrics[metric])

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metric.json'), 'w') as metric_file:
            json.dump(metrics, metric_file)


if __name__ == '__main__':
    validate()
