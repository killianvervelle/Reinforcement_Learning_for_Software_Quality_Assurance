import asyncio
import os
import time
import docker
import logging

import boto3
from fastapi import FastAPI
from http.client import HTTPException
from contextlib import asynccontextmanager

from unittest import TestCase, main
from pymeter.api.samplers import HttpSampler
from pymeter.api.reporters import HtmlReporter
from pymeter.api.timers import UniformRandomTimer
from pymeter.api.config import TestPlan, ThreadGroupWithRampUpAndHold


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ECS_CLIENT = boto3.client('ecs', region_name='eu-west-3')
CLUSTER_NAME = os.getenv("CLUSTER_NAME", "")
SUT_API_URL = os.getenv("SUT_API_URL", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")

THREADS = os.getenv("THREADS", "")
RAMPUP = os.getenv("RAMPUP", "")
LOOPS = os.getenv("LOOPS", "")

latest_task_arn = ""
container_id = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global latest_task_arn, container_id
    while not latest_task_arn:
        logger.info("Polling ECS for latest task ARN...")
        await asyncio.sleep(20)
        latest_task_arn = get_latest_task()
        logger.info(f"Fetched latest_task_arn={latest_task_arn}")

    if not latest_task_arn:
        logger.error("Failed to fetch latest_task_arn")
        yield

    container_id = get_container_id(latest_task_arn)
    logger.info(
        f"Initialized: latest_task_arn={latest_task_arn}, container_id={container_id}")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health_check():
    return {"message": "Health check successfull."}


@app.post("/stress_test/")
def run_stress_test():
    pass


@app.post("/load_test/")
def run_load_test():
    pass


@app.get("/get_container_id")
def get_container_id():
    return {"Container id": container_id}


@app.post("/adjust_container_resources/")
def adjust_container_resources(cpu: int,
                               memory: int):
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        logger.info(f"Container {container} found. Updating resources...")

        container.update(
            cpu_period=100000,
            cpu_quota=cpu * 1000,
            mem_limit=f"{memory * 1000}m",
            memswap_limit=f"{memory * 1000 * 2}m"
        )

        time.sleep(5)

        container.reload()

        logger.info(
            f"Container resources updated: CPU quota={cpu*1000}, Memory={memory*1000}MB")

    except docker.errors.NotFound:
        logger.error(f"Container {container} not found.")
        raise HTTPException(status_code=404, detail="Container not found")
    except docker.errors.APIError as e:
        logger.error(f"Failed to update container: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update container")


@app.post("/run_jmeter_test_plan/")
def run_jmeter_test_plan(threads: int,
                         rampup: int,
                         loops: int):
    try:
        test_plan = build_test_plan(
            threads=threads,
            rampup=rampup,
            loops=loops
        )

        response_time = execute_test_plan(test_plan)

        return {"response_time": response_time}

    except Exception as e:
        logger.error(f"JMeter test execution failed: {e}")
        raise HTTPException(
            status_code=500, detail="JMeter test execution failed")


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
        if container["name"] == "sut-api-container":
            return container.get('runtimeId')

    print(f"No container found with the name sut-api-container.")
    return None


def build_test_plan(threads, rampup, loops):
    timer = UniformRandomTimer(250, 1000)
    print(threads, rampup, loops, f"{SUT_API_URL}"+"food-supply")
    http_sampler1 = HttpSampler(
        "echo_get_request", f"{SUT_API_URL}"+"food-supply")
    thread_group_main = ThreadGroupWithRampUpAndHold(
        threads, rampup, loops, http_sampler1, timer)

    http_sampler2 = HttpSampler(
        "echo_get_request", f"{SUT_API_URL}"+"undernourishement-data")
    thread_group_main2 = ThreadGroupWithRampUpAndHold(
        threads, rampup, loops, http_sampler2, timer)

    http_sampler3 = HttpSampler(
        "echo_get_request", f"{SUT_API_URL}"+"nutritional-data-country/USA")
    thread_group_main3 = ThreadGroupWithRampUpAndHold(
        threads, rampup, loops, http_sampler3, timer)

    http_sampler4 = HttpSampler(
        "echo_get_request", f"{SUT_API_URL}"+"utilization-data/USA/Production")
    thread_group_main4 = ThreadGroupWithRampUpAndHold(
        threads, rampup, loops, http_sampler4, timer)

    return TestPlan(thread_group_main, thread_group_main2, thread_group_main3, thread_group_main4)


def execute_test_plan(test_plan):
    stats = test_plan.run()

    return stats.sample_time_mean_milliseconds
