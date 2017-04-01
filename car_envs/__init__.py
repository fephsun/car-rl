import logging
from gym.envs.registration import register
from car_envs.car_env import CarEnv

logger = logging.getLogger(__name__)

register(
    id='car-v0',
    entry_point='car_envs:CarEnv',
)
