import logging
import random
import boto3
import requests
import pandas as pd

from utilities import Utilities


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ECS_CLIENT = boto3.client('ecs', region_name='eu-west-3')
ECR_CLIENT = boto3.client('ecr', region_name='eu-west-3')
API_URL = "http://my-load-balancer-978472547.eu-west-3.elb.amazonaws.com:8000/"
SUT_URL = "http://my-load-balancer-978472547.eu-west-3.elb.amazonaws.com:8001/"
REPOSITORY = "my_ecr_repository"
CLUSTER_NAME = "my_ecs_cluster"


def generate_samples(num_samples: int):
    return [
        {
            "cpu_quota": random.randint(13750, 100000),
            "memory": random.randint(175, 1300)
        }
        for _ in range(num_samples)
    ]


def get_latest_task():
    response = ECS_CLIENT.list_tasks(
        cluster=CLUSTER_NAME,
        desiredStatus='RUNNING'
    )

    task_arns = response.get('taskArns', [])

    if not task_arns:
        print("No running tasks found.")
        return None

    latest_task_arn = task_arns[-1]
    return latest_task_arn


def get_container_id(last_task):
    response = ECS_CLIENT.describe_tasks(
        cluster=CLUSTER_NAME,
        tasks=[last_task]
    )

    containers = response.get('tasks', [])[0].get('containers', [])

    if not containers:
        print(f"No container found in task {last_task}.")
        return None

    for container in containers:
        if container["name"] == "sut-container":
            return container.get('runtimeId')

    print(f"No container found with the name api-container.")
    return None


def adjust_container(cpu_quota: int, memory: int, container: str, image_name: str) -> bool:
    """Adjust container resources using an API."""
    try:
        response = requests.post(
            f"{API_URL}adjust_container/",
            params={
                "cpu_quota": cpu_quota,
                "memory": memory,
                "container": container,
                "img_name": image_name
            }
        )

        response.raise_for_status()
        logger.info("Container adjusted successfully.")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Error adjusting container: {e}")
        return False


def simulate_cpu_task() -> float:
    try:
        response = requests.post(f"{SUT_URL}cpu_task")
        response.raise_for_status()
        return round(response.json(), 3)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error simulating CPU task: {e}")
        return 0.0


def build_dataset() -> pd.DataFrame:
    last_task = get_latest_task()
    container_id = get_container_id(last_task)
    if not container_id:
        logger.error("No active container found. Aborting dataset creation.")
        return pd.DataFrame()

    samples = generate_samples(200)

    df = pd.DataFrame(columns=["cpu", "memory", "responseTime"])

    for sample in samples:
        cpu_quota = sample["cpu_quota"]
        memory = sample["memory"]

        try:
            adjust_container(cpu_quota, memory,
                             container_id, image_name="None")
        except:
            logger.error("Failed to adjust container. Skipping sample.")
            continue

        response_time = simulate_cpu_task()
        print("Response_time", response_time)
        if response_time > 0:
            new_row = pd.DataFrame([{
                "cpu": cpu_quota / 1000,
                "memory": memory / 1000,
                "responseTime": response_time * 1000
            }])

            df = pd.concat([df, new_row], ignore_index=True)

        else:
            logger.error("Failed to get response time. Skipping sample.")
    return df


if __name__ == "__main__":
    utilities = Utilities(logger)

    logger.info("Starting dataset creation...")
    dataset = build_dataset()

    if not dataset.empty:
        logger.info("Saving dataset to storage..")
        utilities.save_data(dataset)
        logger.info("Dataset saved successfully.")
    else:
        logger.warning("Dataset creation failed or returned an empty dataset.")
