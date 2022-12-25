FROM python:3.8-slim-buster

COPY main.py data_model.py requirements.txt .env  /

RUN pip install --no-cache-dir -r /requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
