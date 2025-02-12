FROM python:3.11-slim

WORKDIR /app

COPY src/sut/src /app/src

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --force-reinstall -r /app/src/requirements.txt

RUN ls -l /app  # List the files in /app

RUN pip list

ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
