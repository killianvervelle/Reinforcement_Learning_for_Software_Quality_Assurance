from typing import Dict, Any
import docker
from fastapi import HTTPException
import requests
import csv
import time
import logging
import os

LOG_FILE = "/response_times.csv"
API_URL = "http://127.0.0.1:8000/compute_testing/"
LOCAL_LOG_FILE = "response_times.csv"
CONTAINER_NAME = "resource-manager"
IMAGE_NAME = "dynamic-resource-fastapi"
CPU_PERIOD = 100000
CLIENT = docker.from_env()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_api(cpu_quota: int, memory: str) -> Dict[str, Any] | None:
    try:
        response = requests.post(
            API_URL + f"?cpu_quota={cpu_quota}&memory={memory}m"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling API: {e}")
        return None


def log_to_csv(data: Dict[str, Any]) -> None:
    file_exists = os.path.exists(LOCAL_LOG_FILE)

    with open(LOCAL_LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "CPU", "Memory",
                            "ResponseTimeCpus", "ResponseTimeMems"])
        writer.writerow([data["timestamp"], data["adjusted_cpu_quota"],
                         data["adjusted_memory"], data["response_time_cpus"],
                         data["response_time_mems"]])


def adjust_container_resources(client: docker.DockerClient, container: docker.models.containers.Container, cpu_quota: int, memory: int) -> None:
    try:
        logger.info("Stopping the container...")
        container.stop()
        time.sleep(8)
        container.remove(force=True)
        time.sleep(2)
        logger.info("Container removed. Updating resources and restarting...")

        new_container = client.containers.run(
            IMAGE_NAME,
            ports={'8000/tcp': 8000},
            name=CONTAINER_NAME,
            cpu_period=CPU_PERIOD,
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
            status_code=500, detail="Error adjusting container resources")


def main():
    iterations = [
        {"cpu_quota": 13750, "memory": 175},
        {"cpu_quota": 72500, "memory": 1200},
        {"cpu_quota": 83000, "memory": 950},
        {"cpu_quota": 45000, "memory": 600}
    ]

    for iteration in iterations:
        try:
            container = CLIENT.containers.get(CONTAINER_NAME)
            adjust_container_resources(
                CLIENT, container, iteration['cpu_quota'], iteration['memory'])

            container = CLIENT.containers.get(CONTAINER_NAME)
            container_id = container.id
            logger.info(f"Container ID: {container_id}")

            status = container.stats(decode=None, stream=False)
            if not container_id:
                raise HTTPException(
                    status_code=500, detail="Could not determine container ID")

            logger.info(
                f"Calling API with: Cpu_quota={iteration['cpu_quota']}, Memory={iteration['memory']}m")
            result = call_api(iteration["cpu_quota"], iteration["memory"])

            if result:
                logger.info(f"API result: {result}")
                log_to_csv(result)
            time.sleep(2)
        except HTTPException as e:
            logger.error(f"HTTP error: {e.detail}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
