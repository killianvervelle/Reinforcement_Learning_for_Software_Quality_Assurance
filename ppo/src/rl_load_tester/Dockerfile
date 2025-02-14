FROM python:3.11-slim

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/rl_load_tester/ ./

RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV JVM_PATH=/usr/lib/jvm/default-java/lib/server/libjvm.so
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONPATH=/app
ENV JVM_ARGS="-Xms256m -Xmx2048m"

ARG THREADS
ARG RAMPUP
ARG LOOPS
ARG SUT_API_URL
ARG REPOSITORY
ARG CLUSTER_NAME

ENV THREADS=$THREADS
ENV RAMP_UP=$RAMPUP
ENV LOOPS=$LOOPS
ENV SUT_API_URL=$SUT_API_URL
ENV REPOSITORY=$REPOSITORY
ENV CLUSTER_NAME=$CLUSTER_NAME


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


