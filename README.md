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

Install JupyterLab (or Jupyter Notebook)
```
pip3 install jupyterlab
```
