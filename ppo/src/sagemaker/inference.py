import logging
import gym
import boto3
import tarfile
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI

from agent import Agent
from backboneNetwork import Network
from backboneNetwork import ActorCriticModel
from virtualMachine import VirtualMachine
from utilities import Utilities


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

model_uri = "s3://mybucketingbucket/sagemaker/  #ppo-training-1739957781/output/model.tar.gz"
bucket_name = "mybucketingbucket"
prefix = "sagemaker/ppo-training-"
s3_model_path = "sagemaker/"
local_model_path = "sagemaker/model.tar.gz"

s3_client = boto3.client('s3')
utils = Utilities(logger=logger)


def initialize_vms(n, requirement_res_times):
    vm_list = []

    for i in range(n):
        vm = VirtualMachine(
            Requirement_ResTime=requirement_res_times[i]
        )
        vm_list.append(vm)

    return vm_list


def initialize_env(model):
    requirement_res_times = [3000, 3000]
    vms = initialize_vms(2, requirement_res_times)
    print(f"Initialized {len(vms)} VMs.")

    if "Env-v1" not in gym.envs.registry:
        gym.register(
            id="Env-v1",
            entry_point="my_gym.myGymStress:ResourceStarving"
        )

    env_train = gym.make("Env-v1", vm=vms[0], model=model)
    env_test = gym.make("Env-v1", vm=vms[0], model=model)

    return env_train, env_test


def get_latest_model_key():
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix)

    sorted_objects = sorted(
        response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)

    return sorted_objects[0].get("Key")


def load_model_from_s3(bucket_name, local_model_dir='/tmp/model'):
    s3 = boto3.client('s3')

    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    latest_model_key = get_latest_model_key()
    local_model_path = os.path.join(local_model_dir, 'model.tar.gz')
    s3.download_file(bucket_name, latest_model_key, local_model_path)

    with tarfile.open(local_model_path, "r:gz") as tar:
        tar.extractall(path=local_model_dir)

    model_checkpoint = torch.load(os.path.join(
        local_model_dir, 'ppo_trained_model.pth'))

    return model_checkpoint


trained_model = utils.load_model()

_, env_test = initialize_env(trained_model)

agent = Agent(
    env_train=None,
    env_test=env_test
)

model_checkpoint = load_model_from_s3(bucket_name=bucket_name)
agent.trained_actor = Network(in_features=3, hidden_dimensions=64,
                              out_features=4, dropout=0.2)
agent.trained_critic = Network(in_features=3, hidden_dimensions=64,
                               out_features=1, dropout=0.2)
agent.trained_actor.load_state_dict(model_checkpoint['actor_state_dict'])
agent.trained_critic.load_state_dict(model_checkpoint['critic_state_dict'])
model = ActorCriticModel(actor=agent.trained_actor,
                         critic=agent.trained_critic)
model.eval()


@app.get("/ping")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    total_reward = 0
    episodes = 5
    episode_results = []

    for episode in range(episodes):
        state = env_test.reset()
        done = False
        episode_reward = 0

        while not done:
            state = state[0] if isinstance(state, tuple) else state
            state = torch.FloatTensor(state).unsqueeze(0).view(1, -1)
            action_pred, _ = model(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            state, reward, done, _, _ = env_test.step(action.item())
            episode_reward += reward
        total_reward += episode_reward

        episode_results.append({
            "episode": episode,
            "final_state": state.tolist(),  # Convert tensor to list
            "episode_reward": episode_reward
        })

    return {
        "episodes": episode_results
    }
