from walk import Walk
from action import Action


class Population():

    def __init__(self, num_actions, action_space):
        '''
            :param num_walks:
                total number of design points,
                from which each one is sequence
                of actions and their respective
                reward value(objective function values).

            :param num_actions:
                total number of actions which leads
                the agent to walk, while maintaining
                the center of the gravity.

             :param action_dimension:
        '''

        self.__actions = []
        self.__num_actions = num_actions
        self.__action_space = action_space

    def generate_random_actions(self):
        for i in range(self.__num_actions):
            action = Action(self.__action_space)
            random_action = action.generate_one_random_action()
            action.set_action_one_step(random_action)
            self.__actions.append(action)

    def get_actions(self):
        return self.__actions




