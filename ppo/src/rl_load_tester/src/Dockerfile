FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime 

# Install Python 3.11
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Copy training script
COPY src/rl_load_tester/src/  /opt/ml/code/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set entrypoint for SageMaker
ENTRYPOINT ["python", "/opt/ml/code/main.py"]


