FROM python:3.11-slim

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/rl_load_tester/ /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV JVM_PATH=/usr/lib/jvm/default-java/lib/server/libjvm.so
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


