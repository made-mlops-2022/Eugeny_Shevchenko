# Homework №2

------------------

### build image
- add `.env` file with `MODEL_PATH` and `TRANSFORMER_PATH` path
- From source: from `online_inference/` run:
```
docker build -t 3530385/online_inference:v1 .
```


From DockerHub:
```
docker pull 3530385/online_inference:v1
```
--------------------

### run container
```
docker run --name inference -p 8888:8888 3530385/online_inference:v1
```
Model running on http://127.0.0.1:8888/docs

----------------------

### make requests
clone repo and install environment
~~~
python -m venv .venv
source .venv/bin/activate
pip install --upgrate pip
pip install -r requirements.dev.txt
~~~

from `online_inference/` run:
```
python make_requests.py
```
### run tests
```
python -m pytest test_main.py
```

-----------------


### Docker Image Optimizations
1. First build 

    ![image](https://user-images.githubusercontent.com/45338087/204732220-af744b8f-032d-4933-b009-04fa12dedf76.png)
    ![image](https://user-images.githubusercontent.com/45338087/204732289-142b2972-344b-4030-8f4f-3780f826ad99.png)
    
    `requirements.txt` contains pytest and requests. all `online_inference` folder copy in docker image.
    Size 1.35 GB 
2. Don’t install unnecessary packages 

    ![image](https://user-images.githubusercontent.com/45338087/204733048-44a0ffe6-f8d7-4f49-b159-dfacada63976.png)
    ![image](https://user-images.githubusercontent.com/45338087/204733096-5a8858d8-9b26-451c-90dc-86ee867a363b.png)
    
    `requirements.txt` does not contain pytest and requests. only `main.py data_model.py requirements.txt .env` copy in docker image.
    Size 1.28 GB 
3. Choose more lightweight basis  

    ![image](https://user-images.githubusercontent.com/45338087/204733668-7a748dd4-773e-49b2-aa27-d8faed23464d.png)
    ![image](https://user-images.githubusercontent.com/45338087/204733700-fe527263-159e-4e26-9331-b6e7f6e7f391.png)
    
    Used python:3.8.15-slim-buster  
    Size 487.28 MB 
