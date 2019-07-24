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
    'mission': 'maze_find_chicken.xml',
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



register_env("malmo", create_malmo)
env = create_malmo(env_config={"env": "malmo",
                 "num_workers": 1,
                 "num_gpus": 0}
         )


ray.init(num_cpus=20)

tune.run("IMPALA",
         stop={
             "timesteps_total": 10000,
         },
         config={"env": "malmo",
                 "num_workers": 1,
                 "num_gpus": 0}
         )

ray.shutdown()
