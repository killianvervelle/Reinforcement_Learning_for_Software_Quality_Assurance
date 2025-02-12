FROM python:3.11-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY src/rl_load_tester /app/rl_load_tester

RUN pip install --no-cache-dir --force-reinstall -r /app/rl_load_tester/requirements.txt

RUN ls -l /app  # List the files in /app

RUN pip list

ENV PYTHONPATH=/app/

EXPOSE 8000

CMD ["uvicorn", "rl_load_tester.main:app", "--host", "0.0.0.0", "--port", "8000"]


