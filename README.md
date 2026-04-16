# Isaac Lab Locomotion Learning

This repository provides a simple framework for training robots using CPGRBF networks with PIBB.


## Installation
Tested: Ubuntu 22.04, IsaacSim 5.1

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/worasuch/IsaacLab-LocoNets.git
```


- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd IsaacLab-LocoNets
python -m pip install -e source/IsaacLabLocoNets
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/ES/train.py --task Slalom --headless --num_envs 2
```


## Training
Train with default configuration (as defined in the config file)
```bash
python scripts/ES/train.py --task <TASK_NAME>
```
Customize training parameters via CLI (without modifying the config file)
```bash
python scripts/ES/train.py --task Slalom --num_envs 1024 --cpg_rbf --headless --wandb
```



## Playing 
Path to collect the model is following 
- logs/es/`TASK_NAME`/`MODEL`/`EXPERIMENT`/ model /`CHECKPOINT`
- models path are in
    - logs/es/`Slalom`/`cpg_rbf`/`2026-04-16_21-42-33`/model/`model_499.pickle`
```bash
python scripts/ES/play.py --cpg_rbf --task Slalom --num_envs 1 --experiment 2026-04-16_21-42-33 --checkpoint model_499.pickle
```

## Logging
- To log training progress, we use wandb for real-time visualization. Before running any experiments, make sure you’re logged in by executing:
```bash
wandb login
```