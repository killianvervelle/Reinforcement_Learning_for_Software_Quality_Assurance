FROM python:3.9-slim

WORKDIR /app

RUN pip install -r ppo/requirements.txt

COPY ppo/app/src/main.py /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
