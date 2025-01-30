FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY app/src/sut/main.py /app/
COPY app/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
