# IMPORT LIBRARIES

import gym
import numpy as np
import glfw
import random

# CONFIGURE HYPERPARAMETERS

ACTION_LEN = 1000
FIRST_GENERATION_SIZE = 100
NEXT_GENERATION_SIZE = 100
NUM_GENERATIONS = 20
DECIMAL_PERISH = 0.6

# ENVIRONMENT SETUP

env = gym.make('HalfCheetah-v2')
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(obs_dim, act_dim)
#env.render()

class Candidate:
    num_instances = 1

    def __init__(self, action_len, gen):
        self.action_len = action_len
        self.gen = gen
        self.cand_num = Candidate.num_instances
        self.reward = 0
        self.moves = []
        Candidate.num_instances += 1
    
    def set_reward(self, reward):
        self.reward = reward
        
    def add_move(self, move):
        self.moves.append(move)
    
    def __str__(self):
        cand_str = "Candidate {cand_num}, Generation {gen}, Reward {reward}\n\n"
        return cand_str.format(cand_num=self.cand_num, gen=self.gen, reward=self.reward)
    
    def __add__(self, other):
        # mates self with other, produces offspring
        offspring = Candidate(self.action_len, self.gen + 1)
        
        split_point = (int) (self.action_len/2)
        
        # coin flip
        if (np.random.choice([True, False])):
            for i in range(self.action_len):
                if (i > split_point):
                      offspring.add_move(self.moves[i])
                else:
                      offspring.add_move(other.moves[i])
        else:
            for i in range(self.action_len):
                if (i > split_point):
                    offspring.add_move(other.moves[i])
                else:
                    offspring.add_move(self.moves[i])            
        return offspring
                
        


def generate_candidate():

    initialCandidate = Candidate(ACTION_LEN, 0)
    for i in range(initialCandidate.action_len):
        move = np.random.uniform(low=-1, high=1, size=act_dim)
        initialCandidate.add_move(move)
    return initialCandidate

def generate_first_generation():
    population = []
    for i in range(FIRST_GENERATION_SIZE):
        population.append(generate_candidate())
    return population

# def generate_first_generation():
#     population = []
#     cand = generate_candidate()
#     for i in range(FIRST_GENERATION_SIZE):
#         population.append(cand)
#     return population

def perform_natural_selection(current_population):
    parents = []
    NUM_PERISH = (int) (NEXT_GENERATION_SIZE * DECIMAL_PERISH)
    
    sorted_by_reward = sorted(current_population, key=lambda cand: cand.reward) 
    #print("SORTED LENGTH", len(sorted_by_reward))
    
    #for i in range(len(sorted_by_reward)):
    #    print(sorted_by_reward[i])
    selected = sorted_by_reward[NUM_PERISH:]
    #print("SELECTED LENGTH", len(selected))
    #for i in range(len(selected)):
    #    print(selected[i])    
    best = selected[-1]
    #print("BEST", best)
    num_to_add = 1
    for candidate in selected:
        for i in range(num_to_add):
            parents.append(candidate)
        num_to_add += 1
    #print("PARENTS LENGTH", len(parents))
    random.shuffle(parents)
    return parents, best

def create_offspring(parents, best):
    offspring = []
    offspring.append(best)
    
    for i in range(NEXT_GENERATION_SIZE):
        f_parent_ind = random.randint(0, len(parents) - 1)
        s_parent_ind = random.randint(0, len(parents) - 1)
        
        f_parent = parents[f_parent_ind]
        s_parent = parents[s_parent_ind]
        
        child = f_parent + s_parent
        offspring.append(child)

    return offspring

def evolve():
    print('START EVOLUTION')
    population = generate_first_generation()
    
    for generation in range(NUM_GENERATIONS):
        for candidate in population:
            env.reset()
            total_reward = 0
            cand_done = False
            moves_taken = 0
            while ((not cand_done) and (moves_taken < candidate.action_len)):
                env.render()
                move = candidate.moves[moves_taken].reshape((1,-1)).astype(np.float32)
                obs, reward, done, _ = env.step(np.squeeze(move, axis=0))
                cand_done = done
                total_reward += reward
                moves_taken += 1
                #print(moves_taken, candidate.action_len, len(candidate.moves))
            candidate.set_reward(total_reward)
            env.reset()

        parents, best = perform_natural_selection(population)
        population = create_offspring(parents, best)
        #sorted_by_reward = sorted(population, key=lambda cand: cand.reward)
        #print("best of offspring", sorted_by_reward[-1])

        print('GENERATION : ', generation, ', BEST : ', best)
    print('EVOLUTION HAS FINISHED')
  
evolve()
env.close()
glfw.terminate()