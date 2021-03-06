import os
from copy import deepcopy

import ray
import malmoenv
import marlo
from ray.tune import register_env, tune
from pathlib import Path

# MALMO_XSD_PATH = os.environ['MALMO_XSD_PATH']
# MALMO_ROOT_PATH = os.environ['MINECRAFT_MALMO_ROOT']
# MALMO_MISSION_PATH = MALMO_ROOT_PATH+os.sep+"MalmoEnv"+os.sep
MALMO_MISSION_PATH = os.path.join("/home/alizaidi/oneweek/minetribe/malmo/MalmoEnv/")


MALMO_DEFAULTS = {
    'mission': 'missions/mobchase_single_agent.xml',
    'port': 10000,
    'server': '127.0.0.1',
    'port2': None,
    'server2': None,
    'episodes': 1,
    'episode': 0,
    'role': 0,
    'episodemaxsteps': 0,
    'saveimagesteps': 0,
    'resync': 0,
    'experimentUniqueId': 'test1'
}

def create_marlo(env_config: dict):

    config = deepcopy(MALMO_DEFAULTS)
    config.update(env_config)

    client_pool = [('127.0.0.1', 10000)]
    join_tokens = marlo.make('MarLo-DefaultFlatWorld-v0', 
                params=dict(
                    allowContinuousMovement=["move", "turn"],
					client_pool = client_pool,
                    videoResolution=[84, 84]
                ))
    
    env = marlo.init(join_tokens[0])

    return env

def create_malmo(env_config: dict):
    config = deepcopy(MALMO_DEFAULTS)
    config.update(env_config)

    if config['server2'] is None:
        config['server2'] = config['server']

    xml = Path(MALMO_MISSION_PATH+config["mission"]).read_text()
    env = malmoenv.make()
    env.init(xml, config["port"],
             server=config["server"],
             server2=config["server2"], port2=config["port2"],
             role=config["role"],
             exp_uid=config["experimentUniqueId"],
             episode=config["episode"], resync=config["resync"])

    return env


register_env("malmo", create_malmo)
register_env("marlo", create_marlo)

ray.init(num_cpus=4)

tune.run("IMPALA",
         stop={
             "timesteps_total": 10000,
         },
         config={"env": "marlo",
                #  "model": {
                #     "conv_filters": [224,224,224],
                #     "fcnet_hiddens": [1024,1024],
                #     "framestack": False,
                #     "grayscale": False,
                #     "fcnet_activation":"relu"
                #  },
                 "num_workers": 1,
                 "num_gpus": 0}
         )

ray.shutdown()