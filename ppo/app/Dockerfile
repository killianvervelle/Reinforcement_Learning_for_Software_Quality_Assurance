FROM python:3.9-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY app/src/sut.py /app/
COPY app/src/main.py /app/
COPY app/requirements.txt /app/

RUN pip install -r requirements.txt

ENV PORT 8000

EXPOSE 8000 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]


