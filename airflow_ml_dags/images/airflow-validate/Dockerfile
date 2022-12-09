FROM airflow-ml-base:latest

COPY validate.py requirements.txt  /

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "validate.py"]