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
            memswap_limit=f"{memory * 2}m"
        )

        time.sleep(5)

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

