FROM python:3.11-slim AS builder

WORKDIR /app

COPY src/sagemaker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11

RUN pip install --upgrade pip

WORKDIR /app

COPY src/rl_load_tester/src/my_gym my_gym
COPY src/rl_load_tester/src/agent.py agent.py
COPY src/rl_load_tester/src/backboneNetwork.py backboneNetwork.py
COPY src/rl_load_tester/src/virtualMachine.py virtualMachine.py
COPY src/rl_load_tester/src/utils/utilities.py utilities.py
COPY src/sagemaker/inference.py inference.py

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

EXPOSE 8080

ENV SAGEMAKER_PROGRAM inference.py

ENTRYPOINT ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080"]