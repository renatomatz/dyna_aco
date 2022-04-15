"""Define Optimizer classes.

Optimizer instances are the base of any simulation routine as they orchestrate
all pieces of a reinforcement learning task.
"""


import pickle

import numpy as np

from time import time

from .model import Model
from .environment import Environment
from .agent import Agent
from .utils import assert_type


class Optimizer:
    """Base Optimizer class."""

    def __init__(self, env, agent):
        """Initialize Optimizer instance."""
        assert_type(env, Environment, var_name="env")
        assert_type(agent, Agent, var_name="agent", or_none=True)
        self._env = env
        self._agent = agent
        self.clear_fit_log()

    @property
    def env(self):
        """Environment instance attribute getter."""
        return self._env

    @property
    def agent(self):
        """Agent instance/generator attribute getter."""
        return self._agent

    def clear_fit_log(self):
        """Clear fitting log."""
        self.fit_log = list()
        self.fit_time = 0

    def fit(self, iters, run_len=np.inf, verbose=False):
        """Run simulations and fit functions with necessary settings."""

        # History is erased for fitting
        self.clear_fit_log()
        self.agent.clear_history()

        start_time = time()

        cur_run = 0
        state, _, done = self.env.reset(self.agent)
        for i in range(1, iters+1):

            if verbose and (i % 100) == 0:
                print(f"{self} Episode: {i}", end='\r')
            # use cur_run to account for early stops
            if done or cur_run == run_len:
                cur_run = 0
                state, _, done = self.env.reset(self.agent)
                self.fit_log.append(self.agent.discounted_rewards())
                self.agent.clear_history()

            cur_run += 1
            _, state, _, done = self.episode(state)

        self.fit_time = time() - start_time

    def test(self, iters, n_eps, verbose=False):
        """Run tests using test settings."""
        raise NotImplementedError()

    def episode(self, state):
        """Process and episode based on a specific environment state."""
        raise NotImplementedError()

    def __str__(self):
        """String representation of Optimizer."""
        return (f"{type(self).__name__} "
                f"[ E: {type(self.env).__name__} "
                f"| A: {type(self.agent).__name__} ]")


class QLearning(Optimizer):

    def __init__(self, env, agent, alpha=0.9, eps=0.1):
        super().__init__(env, agent)
        self.q_value = np.zeros(env.shape)
        self.alpha = alpha
        self.eps = eps

    def episode(self, state):
        self.agent.start_episode()

        action = self.choose_action(state)

        next_state, reward, done = self.env.step(self.agent, state, action)

        self.q_value[state, action] += (
            self.alpha
            * (reward + self.agent.gamma * np.max(self.q_value[next_state, :])
               - self.q_value[state, action])
        )

        self.agent.history.append((state, action, reward))

        return action, next_state, reward, done

    def greedy_action(self, state, actions=None):
        if actions is None:
            actions = self.env.possible_actions(self.agent)
        return np.random.choice(
            np.flatnonzero(self.q_value[state, actions]
                           == np.max(self.q_value[state, actions]))
        )

    def choose_action(self, state):
        """epsilon-greedy policy over possible actions.

        Make decision based on the merits of the expected and instantaneous
        utilities from a decision.
        """
        actions = self.env.possible_actions(self.agent)
        if np.random.random() < self.eps:
            # should this be a softmax instead?
            action = np.random.choice(actions)
        else:
            # This randomly selects option if values are the same
            action = self.greedy_action(state, actions=actions)

        return action

    def test(self, iters, n_eps, verbose=False):

        results = np.zeros(iters)

        for i in range(iters):
            if verbose:
                print(f"{self} Iter: {i}", end='\r')

            self.agent.clear_history()
            state, _, done = self.env.reset(self.agent)
            for _ in range(n_eps):

                if done:
                    break

                action = self.greedy_action(state)
                state, reward, done = self.env.step(self.agent, state, action)
                self.agent.history.append((state, action, reward))

            results[i] = self.agent.discounted_rewards()

        return results


class DynaQ(QLearning):

    def __init__(self, env, agent, model, alpha=0.9, eps=0.1,
                 planning_steps=50):
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
        action, next_state, reward, done = super().episode(state)

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

    def __str__(self):
        return (super().__str__()[:-1] +
                f"| M: {type(self.model).__name__} ]")


class DynaQACO(DynaQ):

    def __init__(self, env, agent, model, n_agents,
                 alpha=0.9, eps=0.1, planning_steps=50):

        super().__init__(env, agent, model,
                         alpha=alpha, eps=eps,
                         planning_steps=planning_steps)
        self.n_agents = n_agents

    def fit(self, iters, run_len=np.inf, verbose=False):

        # History is erased for fitting
        self.clear_fit_log()
        self.agent.clear_history()

        start_time = time()

        cur_run = 0
        state, _, done = self.env.reset(self.agent)
        for i in range(1, iters+1):

            if verbose and (i % 100) == 0:
                print(f"{self} Episode: {i}", end='\r')

            for a in range(self.n_agents):
                # use cur_run to account for early stops
                if done or cur_run == run_len:
                    cur_run = 0
                    state, _, done = self.env.reset(self.agent)
                    self.fit_log.append(self.agent.discounted_rewards())
                    self.agent.clear_history()

                cur_run += 1
                _, state, _, done = self.episode(state)

            self.model_learn()

        self.fit_time = time() - start_time

    def episode(self, state):
        action, next_state, reward, done = super(DynaQ, self).episode(state)

        self.model.time += 1

        # feed the model with experience
        self.model.feed(state, action, next_state, reward)

        return action, next_state, reward, done

    def model_learn(self):

        self.model.end_episodes()

        # sample experience from the model
        for t in range(0, self.planning_steps):
            state_, action_, next_state_, reward_ = self.model.sample()
            self.q_value[state_, action_] += (
                self.alpha
                * (reward_ + self.agent.gamma
                   * np.max(self.q_value[next_state_, :])
                   - self.q_value[state_, action_])
            )

        self.model.evaporate()


def dump_opt(opt, file_name):
    opt.env.dump_dist()
    with open(file_name, "wb") as f:
        pickle.dump(opt, f)
    # load distribution in case another optimizer shares the
    # environment object
    opt.env.load_dist()


def load_opt(file_name):
    with open(file_name, "rb") as f:
        opt = pickle.load(f)
    opt.env.load_dist()
    return opt
