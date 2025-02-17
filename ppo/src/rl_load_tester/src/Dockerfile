FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime 

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY ppo/src/rl_load_tester/requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ppo/src/rl_load_tester/src/ .  

ENTRYPOINT ["python", "/app/main.py"]

