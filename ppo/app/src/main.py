import time
import math
from fastapi import FastAPI
from datetime import datetime
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_MEMORY_ALLOCATION = 1024 * 1024 * 500
CPU_TASK_ITERATIONS = 10**7


def perform_mem_task() -> float:
    try:
        logger.info("Starting memory-heavy task...")
        start_time = time.time()
        x = bytearray(MAX_MEMORY_ALLOCATION)
        response_time = time.time() - start_time
        logger.info(f"Memory task completed in {response_time:.4f} seconds")
        return response_time
    except MemoryError:
        logger.error("Memory allocation failed due to insufficient memory.")
        return float('inf')
    except Exception as e:
        logger.error(f"Unexpected error in memory task: {e}")
        return float('inf')


def perform_cpu_task() -> float:
    try:
        logger.info("Starting CPU-heavy task...")
        start_time = time.time()
        x = 0
        for i in range(CPU_TASK_ITERATIONS):
            x += math.sqrt(i)
        response_time = time.time() - start_time
        logger.info(f"CPU task completed in {response_time:.4f} seconds")
        return response_time
    except Exception as e:
        logger.error(f"Unexpected error in CPU task: {e}")
        return float('inf')


@app.post("/compute_testing/")
def adjust_resources(cpu_quota: int, memory: str):
    try:
        logger.info(
            f"Received request with CPU Quota: {cpu_quota} and Memory: {memory} MB")

        response_time_cpu = perform_cpu_task()
        response_time_mem = perform_mem_task()

        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "adjusted_cpu_quota": cpu_quota,
            "adjusted_memory": memory,
            "response_time_cpus": response_time_cpu,
            "response_time_mems": response_time_mem
        }
    except Exception as e:
        logger.error(f"Error in adjust_resources endpoint: {e}")
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "adjusted_cpu_quota": None,
            "adjusted_memory": None,
            "response_time_cpus": None,
            "response_time_mems": None
        }
