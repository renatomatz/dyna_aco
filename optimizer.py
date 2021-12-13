import numpy as np

from model import Model
from environment import Environment
from agent import Agent
from utils import assert_type


class Optimizer:

    def __init__(self, env, agent):
        assert_type(env, Environment, var_name="env")
        assert_type(agent, Agent, var_name="env")
        self._env = env
        self._agent = agent

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    def fit(self, iters):

        rewards_history = list()

        state, _, done = self.env.reset(self.agent)
        for _ in range(iters):
            if done:
                state, _, done = self.env.reset(self.agent)
                rewards_history.append(self.agent.discounted_rewards())
                self.agent.clear_history()

            _, state, _, done = \
                self.episode(state)

        rewards_history.append(self.agent.discounted_rewards())
        return rewards_history

    def episode(self, state):
        raise NotImplementedError()


class QLearning(Optimizer):

    def __init__(self, env, agent, alpha=0.9, eps=0.1):
        super().__init__(env, agent)
        self.q_value = np.zeros(env.shape)
        self.alpha = alpha
        self.eps = eps

    def episode(self, state):
        self.agent.age += 1

        action = self.choose_action(state)

        next_state, reward, done = self.env.step(self.agent, state, action)

        self.q_value[state, action] += (
            self.alpha
            * (reward + self.agent.gamma * np.max(self.q_value[next_state, :])
               - self.q_value[state, action])
        )

        self.agent.history.append((state, action, reward))

        return action, next_state, reward, done

    def choose_action(self, state):
        """Make decision based on the merits of the expected and instantaneous
        utilities from a decision.

        Stuck agents will simply pick the state on which they were stuck on.
        Unstuck agents will evaluate their best options and pick, possibly
        decising to get stuck. Options are selected in a greedy way by default
        but stochastically if the self.eps option is specified and
        randomly picked.
        """
        actions = self.env.possible_actions(self.agent)
        if np.random.random() < self.eps:
            # should this be a softmax instead?
            action = np.random.choice(actions)
        else:
            # This randomly selects option if values are the same
            action = np.random.choice(
                    np.flatnonzero(self.q_value[state, actions]
                                   == np.max(self.q_value[state, actions]))
            )

        return action


class DynaQ(QLearning):

    def __init__(self, env, agent, model, alpha=0.9, eps=0.1, planning_steps=50):
        super().__init__(env, agent, alpha=alpha, eps=eps)

        # make sure that the model's environment is the same as this
        assert_type(model, Model, var_name="model")
        if not isinstance(model.env, type(env)):
            raise TypeError("Model.env must be the same as env")

        self._model = model
        self.planning_steps = planning_steps

    @property
    def model(self):
        return self._model

    def episode(self, state):
        action, next_state, reward, done = \
            super().episode(state)

        self.model.time += 1

        # feed the model with experience
        self.model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, self.planning_steps):
            state_, action_, next_state_, reward_ = self.model.sample()
            self.q_value[state_, action_] += (
                self.alpha
                * (reward_ + self.agent.gamma
                   * np.max(self.q_value[next_state_, :])
                   - self.q_value[state_, action_])
            )

        return action, next_state, reward, done