# mujocoProject

Installation Instructions

Linux

```
pip3 install gym
pip3 install mujoco-py
```

Download https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
and install into ~/.mujoco/mujoco210.

To add to PATH:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

Install necessary libraries to run mujoco
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt install patchelf
sudo apt install parseelf
```

Install Python dependencies
```
pip3 install -r requirements.txt
```

If any dependencies are missing, use pip3 install to install them.

To run genetic algorithm and save results:
```
cd ga
python3 nesterov_og_genetics.py
```

To run reinforcement learning tuning:
```
cd rl
python3 traintune.py
```

To run training for reinforcement learning for Half Cheetah and Humanoid RL and save results:
```
cd rl
python3 halfcheetah.py
python3 humanoid_RL.py
```

To view results of genetic algorithm or reinforcement learning, change the bestAction.py to read the generated results file and then run:
```
cd results
python3 bestAction.py
```
