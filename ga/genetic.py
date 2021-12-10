# IMPORT LIBRARIES

import gym
import numpy as np
import glfw
import random
import pickle

# CONFIGURE HYPERPARAMETERS

# ACTION_LEN = 1000
# FIRST_GENERATION_SIZE = 500
# NEXT_GENERATION_SIZE = 200
# NUM_GENERATIONS = 30
# DECIMAL_PERISH = 0.6
# CHANCE_OF_MUTATION = 0.05

ACTION_LEN = 1000
FIRST_GENERATION_SIZE = 1000

DEFAULT_NEXT_GENERATION_SIZE = 500
NEXT_GENERATION_SIZE = DEFAULT_NEXT_GENERATION_SIZE

NUM_GENERATIONS = 100000
DECIMAL_PERISH = 0.6

DEFAULT_CHANCE_OF_MUTATION = 0.25
CHANCE_OF_MUTATION = DEFAULT_CHANCE_OF_MUTATION

PREVIOUS_MAX_REWARD = 0
NUM_EQUAL_MAX_REWARDS = 0

# ENVIRONMENT SETUP
np.random.seed(2)
random.seed(2)
# env = gym.make('Ant-v2')
env = gym.make('Humanoid-v2')
# env = gym.make('HalfCheetah-v2')
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(obs_dim, act_dim)
#env.render()

f = open("results.txt", "w")
f.close()



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
                mutate = True if (random.random() <= CHANCE_OF_MUTATION) else False

                if (mutate):
                    #print("MUTATING")
                    offspring.add_move(generate_move())
                else:
                    if (i > split_point):
                          offspring.add_move(self.moves[i])
                    else:
                          offspring.add_move(other.moves[i])
        else:
            for i in range(self.action_len):
                mutate = True if (random.random() <= CHANCE_OF_MUTATION) else False

                if (mutate):
                    #print("MUTATING")                    
                    offspring.add_move(generate_move())
                else:                
                    if (i > split_point):
                        offspring.add_move(other.moves[i])
                    else:
                        offspring.add_move(self.moves[i])            
        return offspring
        
def generate_move():
    return np.random.uniform(low=-1, high=1, size=act_dim)


def generate_candidate():

    initialCandidate = Candidate(ACTION_LEN, 0)
    for i in range(initialCandidate.action_len):
        move = generate_move()
        initialCandidate.add_move(move)
    return initialCandidate

def generate_first_generation():
    population = []
    for i in range(FIRST_GENERATION_SIZE):
        population.append(generate_candidate())
    return population

def perform_natural_selection(current_population):
    parents = []
    NUM_PERISH = (int) (NEXT_GENERATION_SIZE * DECIMAL_PERISH)

    print("LEN CURRENT POPULATION", len(current_population))
    sorted_by_reward = sorted(current_population, key=lambda cand: cand.reward) 
    
    print("LEN SORTED BY REWARD", len(sorted_by_reward))
    selected = sorted_by_reward[NUM_PERISH:]
    if (len(selected) == 0):
        print("\n\n\n SOMETHING WENT WRONG \n\n\n")
    best = selected[-1]
    num_to_add = 1
    for candidate in selected:
        for i in range(num_to_add):
            parents.append(candidate)
        num_to_add += 1
    random.shuffle(parents)
    return parents, best

def create_offspring(parents, best):
    offspring = []
    offspring.append(best)
    
    for i in range(NEXT_GENERATION_SIZE-1):
        conditions_met = False
        while (not conditions_met):
            f_parent_ind = random.randint(0, len(parents) - 1)
            s_parent_ind = random.randint(0, len(parents) - 1)
        

            f_parent = parents[f_parent_ind]
            s_parent = parents[s_parent_ind]
            if (f_parent.cand_num != s_parent.cand_num):        
                child = f_parent + s_parent
                offspring.append(child)
                conditions_met = True

    return offspring

def update_params_after_gen(generation):
    global NEXT_GENERATION_SIZE
    global CHANCE_OF_MUTATION
    global DEFAULT_CHANCE_OF_MUTATION
    
    if (generation % 75 == 0):
        DEFAULT_CHANCE_OF_MUTATION = DEFAULT_CHANCE_OF_MUTATION * 0.8
    NEXT_GENERATION_SIZE = (int) (NEXT_GENERATION_SIZE * 0.8)
    if (NEXT_GENERATION_SIZE < 200):
        NEXT_GENERATION_SIZE = 250
    CHANCE_OF_MUTATION = CHANCE_OF_MUTATION * 0.99

def update_after_stagnation():
    # update mutation rate after stagnation
    global CHANCE_OF_MUTATION
    global DEFAULT_CHANCE_OF_MUTATION
    global NEXT_GENERATION_SIZE
    global DEFAULT_NEXT_GENERATION_SIZE    

    CHANCE_OF_MUTATION = CHANCE_OF_MUTATION * 2
    NEXT_GENERATION_SIZE = NEXT_GENERATION_SIZE * 2
    
    if (CHANCE_OF_MUTATION > DEFAULT_CHANCE_OF_MUTATION):
        CHANCE_OF_MUTATION = DEFAULT_CHANCE_OF_MUTATION * 0.8
    
    if (NEXT_GENERATION_SIZE > DEFAULT_NEXT_GENERATION_SIZE):
        NEXT_GENERATION_SIZE = DEFAULT_NEXT_GENERATION_SIZE



def evolve():
    global PREVIOUS_MAX_REWARD
    global NUM_EQUAL_MAX_REWARD
    global CHANCE_OF_MUTATION
    global NEXT_GENERATION_SIZE
    print('START EVOLUTION')
    population = generate_first_generation()
    
    for generation in range(NUM_GENERATIONS):
        for candidate in population:
            env.reset()
            env.seed(1)


            total_reward = 0
            cand_done = False
            moves_taken = 0
            while (not cand_done):
            # while ((not cand_done) and (moves_taken < candidate.action_len)):
                #env.render()
                move = candidate.moves[moves_taken].reshape((1,-1)).astype(np.float32)
                obs, reward, done, _ = env.step(np.squeeze(move, axis=0))
                cand_done = done
                total_reward += reward
                moves_taken += 1
            candidate.set_reward(total_reward)

        parents, best = perform_natural_selection(population)
        population = create_offspring(parents, best)

        if (PREVIOUS_MAX_REWARD == best.reward):
            NUM_EQUAL_MAX_REWARD += 1
            update_params_after_gen(generation)

        else:
            PREVIOUS_MAX_REWARD = best.reward
            NUM_EQUAL_MAX_REWARD = 0
            update_params_after_gen(generation)
        
        if (NUM_EQUAL_MAX_REWARD >= 3):
            update_after_stagnation()
                        

       
        print('GENERATION : ', generation, ', BEST : ', best, 'CHANCE OF MUTATION : ', CHANCE_OF_MUTATION, 'NEXT GEN SIZE : ', NEXT_GENERATION_SIZE)
        f = open("results.txt", "a")
        strout = str(best.cand_num) + ":" + str(best.reward) + "\n"
        f.write(strout)
        f.close()
        
        with open('bestAction.pkl', 'wb') as bestAction:
            pickle.dump(best.moves, bestAction)

                
                


    print('EVOLUTION HAS FINISHED')
  
evolve()
env.close()
glfw.terminate()
