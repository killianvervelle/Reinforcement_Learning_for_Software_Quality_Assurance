FROM python:3.11-slim

WORKDIR /app

COPY src/rl_load_tester/src/ . 

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "/app/main.py"]
