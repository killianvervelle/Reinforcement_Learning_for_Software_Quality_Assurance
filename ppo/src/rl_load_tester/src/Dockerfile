FROM python:3.11-slim AS builder

WORKDIR /app

COPY src/rl_load_tester/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY src/rl_load_tester/src/ . 

CMD ["python", "/app/main.py"]
