FROM public.ecr.aws/lambda/python:3.11

RUN pip install --upgrade pip

WORKDIR /app

COPY app/src/lambda/main.py ${LAMBDA_TASK_ROOT}

COPY app/src/lambda/requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt

CMD ["main.lambda_handler"]


