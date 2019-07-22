Minecraft Tribes
================

## Setup and Installation

There are two environments, one for [`marlo`](https://github.com/crowdAI/marLo) and one for [`minerl`](https://github.com/minerllabs/minerl/). Both environments are built as conda virtual environments, so please install either miniconda or the full Anaconda distribution before proceeding.

### Marlo

```bash
cd envs
conda env update -f marlo.yml
```

To start using the environment you'll need to start a client:

```bash
$MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
```

### Docker Version

```bash
docker run -it --rm -p 10000:10000 syuntoku/marlo_client
```

Now you can run 

### MineRL

```bash
cd envs
conda env update -f minerl.yml
```