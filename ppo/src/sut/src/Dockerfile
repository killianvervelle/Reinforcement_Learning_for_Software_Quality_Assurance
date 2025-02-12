FROM python:3.9-slim

WORKDIR /app

COPY src/sut/src/ /app/

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --force-reinstall -r /app/requirements.txt

RUN ls -l /app  # List the files in /app

RUN pip list

ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
