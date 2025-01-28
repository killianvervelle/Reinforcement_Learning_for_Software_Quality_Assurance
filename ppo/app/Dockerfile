FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY ppo/app/src/main.py /app/

COPY ppo/app/requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
