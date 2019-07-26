import argparse
import os
from copy import deepcopy

import gitpath
import ray
from scratch.downsampled_malmo_env import DownsampledMalmoEnv
from ray.tune import register_env, tune
from pathlib import Path


repo = gitpath.root()
MALMO_MISSION_DIR = os.path.join(str(repo), "worlds")


MALMO_DEFAULTS = {
    'mission': 'maze_multiagent.xml',
    'port': 50000,
    'server': '127.0.0.1',
    'port2': None,
    'server2': None,
    'episodes': 10,
    'episode': 0,
    'role': 0,
    'episodemaxsteps': 0,
    'saveimagesteps': 0,
    'resync': 0,
    'experimentUniqueId': 'test1'
}


def create_malmo(env_config: dict):
    config = deepcopy(MALMO_DEFAULTS)
    config.update(env_config)

    if config['server2'] is None:
        config['server2'] = config['server']

    xml = Path(os.path.join(MALMO_MISSION_DIR, config["mission"])).read_text()
    env = DownsampledMalmoEnv(84, 84, True)
    env.init(xml, config["port"],
             server=config["server"],
             server2=config["server2"], port2=config["port2"],
             role=config["role"],
             exp_uid=config["experimentUniqueId"],
             episode=config["episode"], resync=config["resync"])
    env.observation_space = env._space
    return env


def parse_args():
    parser = argparse.ArgumentParser(description='malmo multi-agents training')
    parser.add_argument('--mission', type=str, default='maze_multiagent.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=50000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--port2', type=int, default=50000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--episodes', type=int, default=10, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync on every N - default 0 meaning never')
    parser.add_argument('--role', type=int, default=0, help='the role to play the game.')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='the episode max steps.')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='the save image steps.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    return parser.parse_args()


def main():
    arg_params = parse_args()
    register_env("malmo", create_malmo)
    config = {
        'mission': arg_params.mission,
        'port': arg_params.port,
        'server': arg_params.server,
        'port2': arg_params.port2,
        'server2': arg_params.server2,
        'episodes': arg_params.episodes,
        'episode': arg_params.episode,
        'role': arg_params.role,
        'episodemaxsteps': arg_params.episodemaxsteps,
        'saveimagesteps': arg_params.saveimagesteps,
        'resync': arg_params.resync,
        'experimentUniqueId': arg_params.experimentUniqueId
    }
    env = create_malmo(config)

    ray.init(num_cpus=20)

    tune.run("IMPALA",
             stop={
                "timesteps_total": 10000,
             },
             config={"env_config": config,
                     "env": "malmo",
                     "num_workers": 1,
                     "num_gpus": 0
                     }
             )

    ray.shutdown()


main()
