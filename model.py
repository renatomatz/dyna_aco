"""
Copyright (C)
2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
2016 Kenta Shimada(hyperkentakun@gmail.com)
Permission given to modify the code as long as you keep this
declaration at the top

Further modifications made by Renato Zimmermann(renatomatz@gmail.com)
"""


import numpy as np

from environment import Environment
from utils import assert_type


class Model:
    # @env: the environment instance.
    def __init__(self, env):
        # track total time
        self.time = 0

        assert_type(env, Environment, var_name="env")
        self._env = env
        self.reset_model()

    @property
    def env(self):
        return self._env

    def reset_model(self):
        self.model = dict()

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        raise NotImplementedError()

    def sample_state_action(self):
        raise NotImplementedError()

    # randomly sample from previous experience
    def sample(self):
        raise NotImplementedError()


class VanillaDyna(Model):

    def feed(self, state, action, next_state, reward):
        if state not in self.model.keys():
            self.model[state] = dict()
        self.model[state][action] = [next_state, reward]

    def sample_state_action(self):
        state_index = np.random.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]

        action_index = np.random.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]

        return state, action

    def sample(self):
        state, action = self.sample_state_action()
        next_state, reward = self.model[state][action]

        return state, action, next_state, reward


class TimeDyna(VanillaDyna):
    # @timeWeight: also called kappa, the weight for elapsed time in sampling
    #              reward, it need to be small
    def __init__(self, env, time_weight=1e-4):
        super().__init__(env)
        self.time_weight = time_weight

    def feed(self, state, action, next_state, reward):
        self.time += 1
        if state not in self.model.keys():
            self.model[state] = dict()

            # Actions that had never been tried before from a state were
            #   allowed to be considered in the planning step
            for action_ in self.env.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a
                    #   reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[state][action_] = [state, 0, 1]

        self.model[state][action] = [next_state, reward, self.time]

    def sample(self):
        state, action = self.sample_state_action()
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        return state, action, next_state, reward
