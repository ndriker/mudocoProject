from population import Population
# from selection import Selection
# from crossover import Crossover
# from mutation import Mutation
# import tensorflow as tf
# import time

import gym
import numpy as np

# environment setup
np.random.seed(2)

# models
# environment = gym.make('Ant-v2')
environment = gym.make('Humanoid-v2')
# environment = gym.make('HalfCheetah-v2')
abs = environment.reset()

ACTION_SPACE = dict()
ACTION_SPACE['high'] = environment.action_space.high
ACTION_SPACE['low'] = environment.action_space.low
ACTION_SPACE['size'] = environment.action_space.shape[0]

NUM_WALKS = 10

NUM_ACTIONS = 1000
NUM_GENERATIONS = 100000
NUM_AGENTS = 3


def main():
    best_agent_reward = 0.0
    total_agent_actions_count = 0

    agents = dict()
    for _ in range(NUM_AGENTS):
        population = Population(NUM_ACTIONS, ACTION_SPACE)
        population.generate_random_actions()
        actions = population.get_actions()
        agent_reward, total_agent_actions_count = set_actions_reward(actions)
        agents[agent_reward] = actions
    sorted_agents = sorted(agents.items(), key=lambda item: item[0])
    sorted_agents_list = list(sorted_agents)
    # num_top_agents_to_keep = (int)(NUM_AGENTS * 0.4)
    num_top_agents_to_keep = 0
    agents_to_keep = sorted_agents_list[num_top_agents_to_keep:]

    for _ in range(NUM_GENERATIONS):
        print("BEST_REWARD: ", best_agent_reward)
        print("total_actions_count: ", total_agent_actions_count)

        agents = dict()
        agent_actions_count = dict()
        cross_indx = np.random.randint(low=0, high=NUM_ACTIONS)
        for i in range(NUM_AGENTS):
            length_agents_to_keep = len(agents_to_keep)
            two_rand_parent_pairs = np.random.choice(length_agents_to_keep, 2)
            agent_parent_pair = [agents_to_keep[i][1] for i in two_rand_parent_pairs]

            parent_1 = agent_parent_pair[0]
            parent_2 = agent_parent_pair[1]

            parent_1 = parent_1[cross_indx:]
            parent_2 = parent_2[:cross_indx]
            child = np.concatenate([parent_1, parent_2])

            noise = np.random.standard_normal() * 0.1
            mutation_position = np.random.randint(NUM_ACTIONS)
            mutate_child = child[mutation_position].get_action_one_step() + noise
            child[mutation_position].set_action_one_step(mutate_child)

            actions = child
            agent_reward, total_agent_actions_count = set_actions_reward(actions)
            agents[agent_reward] = actions
            if best_agent_reward < agent_reward:
                best_agent_reward = agent_reward

        sorted_agents = sorted(agents.items(), key=lambda item: item[0])
        sorted_agents_list = list(sorted_agents)
        num_top_agents_to_keep = (int)(len(sorted_agents_list) * 0.4)
        agents_to_keep = sorted_agents_list[num_top_agents_to_keep:]


def set_actions_reward(actions):
    # each walk is sequence of actions/steps
    environment.reset()
    environment.seed(1)

    # ============ begin walking ============

    total_reward = 0.0
    total_actions_count = 0

    for action_number, action in enumerate(actions):

        one_step = action.get_action_one_step()
        # print("one_step: ",one_step ,"\n")
        observation, action_reward, done, _ = environment.step(one_step)
        # time.sleep(0.002)
        # print("action_number: ", action_number,
        #       "action_reward: ", action_reward, "done ", done)
        total_actions_count += 1
        if done:
            # print("Agent lost. reward: ", action_reward)

            break

        total_reward += action_reward
        action.set_action_reward(total_reward)
    # print("total: ", total_reward, "\n")
    return total_reward, total_actions_count


if __name__ == "__main__":
    main()
