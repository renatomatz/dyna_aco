"""Define Agent classes, which generally model agent behaviours.

Agents are responsible for making decisions (traversing the state space).
The way on which these decisions are made are up to the programmer, and
can represent any kind of intuition or mechanic necessary. Agents should
not change the world in any way, only interact with it. World states are
handled by the `Environment` class, which can be shared between agents.

Agent attributes expose a mechanism for introducing agent heterogeneity into
simulations. These can be set as necessary and be either static, deterministic
or stochastic in nature.

Other classes are defined here to allow parallel executions of the algorithm to
have a similar interface as sequential executions.
"""

import numpy as np

from .utils import assert_type


class Agent:
    """Base Agent class.

    This class contains the most basic parameters for agents in any
    of our tests. This class is mostly a structure for information
    used in the tested environments.
    """

    def __init__(self, gamma=0.99):
        """Initialize agent.
        """
        self.gamma = gamma

        self.clear_history()

    def clear_history(self):
        """Clear agent history"""
        # (state, action, reward) list
        self.history = list()

    def discounted_rewards(self):
        """Return total lifetime value of this agent."""
        if self.history == []:
            return 0

        rewards = np.array(self.history, dtype=np.float64)[:, 2]
        return np.sum(rewards*(self.gamma**np.arange(len(rewards))))

    def start_episode(self):
        """Perform any pre-episode representational updates"""
        pass


class McCallAgent(Agent):

    def __init__(self, death_age, gamma=0.99):
        """Initialize agent.
        """
        super().__init__(gamma=gamma)

        self.death_age = death_age

        self.age = 0

    @property
    def is_dead(self):
        """Return whether the agent is dead based on its death age"""
        return self.age >= self.death_age

    def start_episode(self):
        self.age += 1


class HuggettAgent(Agent):
    """Hugget Agent class.

    This class contains state and preference parameters used during decision
    making in the Huggett model.
    """

    def __init__(self, gamma=0.99, crra_gamma=1.5):
        super().__init__(gamma=gamma)
        self.crra_gamma = crra_gamma

        self.a = 0

    def utility(self, c):
        """Return the CRRA Utility at a given instantaneous consumption level.

        Consumption of zero causes an error, so instead, just return a large
        negative number. We can interpret this as bankruptcy.
        """
        c = max(c, 0.000001)
        return (c**(1-self.crra_gamma))/(1-self.crra_gamma)


class ACOAgent(Agent):
    """Distributed Agent class.

    This class contains the basic parameters from the original ACO
    specification as well as some adaptations for this application.
    """

    def __init__(self, death_age, gamma=0.99,
                 alpha=1.0, beta=0.0, rho=0.2,
                 decision=None, update=None):
        """Initialize agent.
        """
        super().__init__(gamma=gamma)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.decide = decision
        self.update = update

    def adjust_beliefs(self, beliefs, pheromones):
        """Adjust an array of beliefs to be influenced by an array of
        pheromones according to this agent's parameters.
        """
        weighted = (pheromones ** self.alpha) * (beliefs ** self.beta)
        return weighted / np.sum(weighted)

    @classmethod
    def static_generator(cls, *args, **kwargs):
        """Generate new agents with the same set of initializing arguments."""

        while True:
            yield cls(*args, **kwargs)

    @classmethod
    def arg_evaporation_generator(cls, arg_evap,
                                  *args, **kwargs):
        """Generate new agents with one or more arguments evaporating at a
        constagent rate.
        """

        assert_type(arg_evap, (tuple, list), var_name="arg_evap")

        if isinstance(arg_evap, tuple):
            arg_evap = [arg_evap]

        for name, rate in arg_evap:
            assert_type(name, str, var_name="name")
            assert_type(rate, (float, int), var_name="rate")
            if name not in kwargs:
                raise KeyError("all arguments to be evaporated must be "
                               "specified as keywork arguments")

        while True:
            yield cls(*args, **kwargs)
            for name, rate in arg_evap:
                kwargs[name] *= rate
