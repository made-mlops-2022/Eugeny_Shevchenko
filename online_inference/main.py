import pickle
import urllib.request
from typing import Any

import pandas as pd
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi_health import health

from data_model import Data

app = FastAPI()

model: Any = None
transformer: Any = None


@app.on_event('startup')
def load_model():
    global model
    global transformer
    envs = dict(dotenv_values(".env"))
    MODEL_URL_PATH = envs["MODEL_PATH"]
    TRANSFORMER_URL_PATH = envs["TRANSFORMER_PATH"]
    model = pickle.load(urllib.request.urlopen(MODEL_URL_PATH))
    transformer = pickle.load(urllib.request.urlopen(TRANSFORMER_URL_PATH))


@app.get('/')
async def main():
    return {'message': "hello, go to /docs for run swagger api interface"}


@app.post('/predict')
async def predict(data: Data):
    data_df = pd.DataFrame([data.dict()])
    X = transformer.transform(data_df)
    y = model.predict(X)
    condition = 'healthy' if not y[0] else 'sick'
    return {'condition': condition}


def check_ready():
    return model is not None and transformer is not None


async def success_handler(**kwargs):
    return 'Model ready'


async def failure_handler(**kwargs):
    return 'Model not ready'


app.add_api_route('/health', health([check_ready],
                                    success_handler=success_handler,
                                    failure_handler=failure_handler))
