FROM public.ecr.aws/lambda/python:3.11

RUN pip install --upgrade pip

WORKDIR /app

COPY app/src/lambda/main.py /app/
COPY app/src/lambda/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["main.lambda_handler"]


