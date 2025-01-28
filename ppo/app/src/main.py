import math
import time
import logging
import docker
from fastapi import FastAPI, HTTPException


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


@app.post("/adjust_container/")
def adjust_resource_quotas(cpu_quota: int,
                           memory: int,
                           container: str,
                           img_name: str) -> None:
    try:
        client = docker.from_env()
        container = client.containers.get(container)
        logger.info(f"Container {container} found. Updating resources...")

        container.update(
            cpu_period=100000,
            cpu_quota=cpu_quota,
            mem_limit=f"{memory}m",
            memswap_limit=f"{memory * 2}m",
            oom_kill_disable=True,
            detach=True
        )

        container.reload()

        logger.info(
            f"Container resources updated: CPU quota={cpu_quota}, Memory={memory}MB")
    except docker.errors.NotFound:
        logger.error(f"Container {container} not found.")
        raise HTTPException(status_code=404, detail="Container not found")
    except docker.errors.APIError as e:
        logger.error(f"Failed to update container: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update container")

    """try:
        logger.info("Stopping the container...")
        container.stop()
        time.sleep(8)

        container.remove(force=True)
        time.sleep(2)
        logger.info(
            "Container removed. Updating resources and restarting...")

        new_container = client.containers.run(
            img_name,
            ports={'8000/tcp': 8000},
            name=container,
            cpu_period=100000,
            cpu_quota=cpu_quota,
            mem_limit=f"{memory}m",
            memswap_limit=f"{memory * 2}m",
            oom_kill_disable=True,
            detach=True
        )
        time.sleep(2)

        new_container.reload()

        logger.info(f"New container started with ID: {new_container.id}")

    except docker.errors.DockerException as e:
        logger.error(f"Error adjusting container resources: {e}")
        raise HTTPException(
            status_code=500, detail="Error adjusting container resources")"""
