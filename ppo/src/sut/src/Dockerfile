FROM python:3.9-slim

WORKDIR /app

COPY src/sut/src/ /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN ls -l /app  # List the files in /app

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
