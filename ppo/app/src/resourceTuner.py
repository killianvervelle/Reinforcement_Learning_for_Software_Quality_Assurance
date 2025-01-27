import logging
import random
import boto3
import requests
import pandas as pd

from ppo.app.src.utilities import Utilities


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ECS_CLIENT = boto3.client('ecs', region_name='eu-west-3')
ECR_CLIENT = boto3.client('ecr', region_name='eu-west-3')
API_URL = "http://myLoadBalancer-851677411.eu-west-3.elb.amazonaws.com/"
REPOSITORY = "my_ecr_repository"
CONTAINER_NAME = "my-app"
IMG_TAG = "latest"


def generate_samples(num_samples: int):
    return [
        {
            "cpu_quota": random.randint(13750, 100000),
            "memory": random.randint(175, 1300)
        }
        for _ in range(num_samples)
    ]


def get_latest_img_tag():
    try:
        response = ECR_CLIENT.list_images(repositoryName=REPOSITORY)
        image_ids = response.get('imageIds', [])

        if not image_ids:
            logger.error("No images found in the repository.")
            return ""

        describe_response = ECR_CLIENT.describe_images(
            repositoryName=REPOSITORY,
            imageIds=image_ids
        )
        image_details = describe_response.get('imageDetails', [])

        if not image_details:
            logger.error("No image details available.")
            return ""

        sorted_images = sorted(
            image_details,
            key=lambda img: img.get('imagePushedAt', 0),
            reverse=True
        )

        latest_image = sorted_images[0]
        image_tag = latest_image.get('imageTags', [])
        return image_tag[0]

    except Exception as e:
        logger.error(f"Error fetching the latest image tag: {e}")
        return ""


def adjust_container(cpu_quota: int, memory: int) -> bool:
    """Adjust container resources using an API."""
    try:
        response = requests.post(
            f"{API_URL}adjust_container/",
            params={
                "container": CONTAINER_NAME,
                "cpu_quota": cpu_quota,
                "memory": memory,
                "img_tag": IMG_TAG
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
        response = requests.post(f"{API_URL}cpu_task")
        response.raise_for_status()
        response_time = response.json().get("responseTime", 0.0)
        return float(response_time)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error simulating CPU task: {e}")
        return 0.0


def build_dataset() -> pd.DataFrame:
    latest_img_tag = get_latest_img_tag()
    if not latest_img_tag:
        logger.error("No valid image tag found. Aborting dataset creation.")
        return pd.DataFrame()

    samples = generate_samples(100)
    df = pd.DataFrame(columns=["cpu", "memory", "responseTime"])

    for sample in samples:
        cpu_quota = sample["cpu_quota"]
        memory = sample["memory"]

        # Adjust container resources
        if not adjust_container(cpu_quota, memory):
            logger.error("Failed to adjust container. Skipping sample.")
            continue

        # Simulate workload and capture response time
        response_time = simulate_cpu_task()
        if response_time > 0:
            df = df.append(
                {"cpu": cpu_quota / 1000, "memory": memory /
                    1000, "responseTime": response_time * 1000},
                ignore_index=True
            )
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
