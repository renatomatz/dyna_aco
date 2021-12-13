"""
Copyright (C)
2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
2016 Kenta Shimada(hyperkentakun@gmail.com)
Permission given to modify the code as long as you keep this
declaration at the top

Updated by Renato Zimmermann
* States are now a single number instead of coordinates
"""

import numpy as np


# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, env, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(env.actions)
    else:
        values = q_value[state, :]
        return np.random.choice([action for action, value in enumerate(values)
                                 if value == np.max(values)])


# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        if state not in self.model.keys():
            self.model[state] = dict()
        self.model[state][action] = [next_state, reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        return state, action, next_state, reward


# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @env: the environment instance.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling
    #              reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, env, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.env = env

    # feed the model with previous experience
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

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        return state, action, next_state, reward


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(ant, q_value, model, env, dyna_params):
    state, reward, done = env.reset()
    steps = 0
    while not done:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, env, dyna_params)

        # take action
        next_state, reward = env.step(ant, state, action)

        # Q-Learning update
        q_value[state, action] += (
            dyna_params.alpha
            * (reward + ant.gamma * np.max(q_value[next_state, :])
               - q_value[state, action])
        )

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_, action_] += (
                dyna_params.alpha
                * (reward_ + dyna_params.gamma
                   * np.max(q_value[next_state_, :])
                   - q_value[state_, action_])
            )

        state = next_state

        # check whether it has exceeded the step limit
        if steps > env.max_steps:
            break

    return steps
