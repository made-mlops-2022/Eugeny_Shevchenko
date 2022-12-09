FROM airflow-ml-base:latest
RUN python3 -m pip install --upgrade pip

COPY generate_data.py requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "generate_data.py"]