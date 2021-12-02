import gym, ray
from ray.rllib.agents import ppo, sac, ddpg, dqn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glfw

class walker(gym.Env):
    def __init__(self, env_config):
        self.totalReward = 0
        self.totalSteps = 0
        self.allSteps = 0
        self.sess = tf.compat.v1.Session()
        self.env = gym.make('Humanoid-v2')
        self.obs = (self.env.reset()).astype('float32')
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.observation_space = gym.spaces.Box(low=-1000., high=1000., shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-5., high=5., shape=(self.act_dim,), dtype=np.float32)
        
    def step(self, action):
        # print("Stepping")
        self.env.render()
        self.totalSteps += 1
        action = action.reshape((1,-1)).astype(np.float32)
        self.obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
        self.totalReward += reward
        self.obs = self.obs.astype('float32')
        # print(self.observation_space.contains(self.obs))
        # print("Done Stepping")
        return self.obs, reward, done, _
        
        
    def reset(self):
        #glfw.terminate()
        # print("Resetting")
        self.obs = (self.env.reset()).astype('float32')
        self.allSteps += self.totalSteps
        print("TotalReward:", self.totalReward)
        print("TotalSteps:", self.totalSteps)
        f = open("results.txt", "a")
        strout = str(self.allSteps) + ":" + str(self.totalReward) + "\n"
        f.write(strout)
        f.close
        self.totalReward = 0
        self.totalSteps = 0
        # print(self.observation_space.contains(self.obs))
        # print(self.obs.shape)
        # print("Done Resetting")
        return self.obs

ray.shutdown()
ray.init()
trainer = ppo.PPOTrainer(env=walker, config={
     'env_config': {},
     'framework': 'torch',
     'num_gpus': 0,
     'num_workers' : 0
    })
while True:
    print(trainer.train())