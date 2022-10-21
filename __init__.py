from gym.envs.registration import register
from .envs.HunterEnv import HunterEnv, WALL_ACTION, pointsBetween
from .envs.State import State, HorizontalWall, VerticalWall

register(
    id="hps/HunterEnv-v0",
    entry_point="evasion_gym/envs:HunterEnv",
)
