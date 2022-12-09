FROM airflow-ml-base:latest

COPY predict.py requirements.txt /

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "predict.py"]