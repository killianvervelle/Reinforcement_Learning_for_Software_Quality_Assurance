FROM python:3.11-slim

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY src/rl_load_tester/requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/rl_load_tester/src/ . 

CMD ["python", "/app/main.py"]
