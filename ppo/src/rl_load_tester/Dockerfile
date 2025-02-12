FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY src/rl_load_tester /app/

RUN pip install --no-cache-dir --force-reinstall -r /app/requirements.txt

RUN ls -l /app  # List the files in /app

RUN pip list

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


