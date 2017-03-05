from gym.envs.registration import register
from car_env import CarEnv

register(
    id='car-v0',
    entry_point='car_env:CarEnv',
)
