FROM airflow-ml-base:latest
ENV GIT_PYTHON_REFRESH=quiet
COPY train.py requirements.txt  /

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]