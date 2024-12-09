from gym.envs import register

register(
    id="StressingEnv-v0",  
    entry_point="my_gym.envs.StressEnv:StressingEnvironment",
)