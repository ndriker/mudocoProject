import pickle
import gym
import numpy as np
import glfw
import time

with open('bestAction.pkl', 'rb') as bestActionPkl:
    bestAction = pickle.load(bestActionPkl)

print(bestAction)
env = gym.make('Humanoid-v2')



moves_taken = 0
cand_done = False
total_reward = 0
env.render()
time.sleep(5)
env.seed(1)
env.reset()

while (not cand_done):
# while ((not cand_done) and (moves_taken < candidate.action_len)):
    env.render()
    move = bestAction[moves_taken].reshape((1,-1)).astype(np.float32)
    obs, reward, done, _ = env.step(np.squeeze(move, axis=0))
    cand_done = done
    total_reward += reward
    moves_taken += 1
    #print(moves_taken, candidate.action_len, len(candidate.moves))
print("TOTAL REWARD", total_reward)
env.close()
glfw.terminate()