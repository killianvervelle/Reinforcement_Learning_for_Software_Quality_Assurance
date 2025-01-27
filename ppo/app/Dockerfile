FROM python:3.9-slim

WORKDIR /app

RUN pip install fastapi uvicorn docker

RUN apt-get update && apt-get install -y docker.io && apt-get clean

COPY ppo/app/src/main.py /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


