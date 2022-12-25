import logging

import pandas as pd
import requests

logger = logging.getLogger("requester")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

df = pd.read_csv("data_sample.csv")
X_request = df.drop(["condition"], axis=1)
y_request = df["condition"]
for i in range(y_request.shape[0]):
    resp = requests.post("http://0.0.0.0:8888/predict",
                         json=dict(X_request.loc[i, :]))
    logger.info('app response:')
    logger.info(f'status Code: {resp.status_code}')
    logger.info(f'message: {resp.json()}')
    predict_result = 1 if resp.json()["condition"] == "healthy" else 0
    logger.info(f'predict_result: {predict_result == y_request[i]}')
    logger.info("\n")
