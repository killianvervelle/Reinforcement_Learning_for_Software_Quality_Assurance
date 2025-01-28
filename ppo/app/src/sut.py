import math
import time
import logging
from fastapi import FastAPI


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Health check successfull."}


@app.post("/cpu_task")
def perform_cpu_task() -> float:
    try:
        logger.info("Starting CPU-heavy task...")
        start_time = time.time()
        x = 0
        for i in range(10**7):
            x += math.sqrt(i)
        response_time = time.time() - start_time
        logger.info(
            f"CPU task completed in {response_time:.4f} seconds")
        return response_time

    except Exception as e:
        logger.error(f"Unexpected error in CPU task: {e}")
        return float('inf')
