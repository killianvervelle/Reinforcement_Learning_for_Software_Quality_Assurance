FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY ppo/src/rl_load_tester /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


