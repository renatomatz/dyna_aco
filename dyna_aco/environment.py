"""Define Environment classes, which generally models value assignment.

Environments determine the set of actions avaialable to a specific agent and
control how agents affect a shared environment. Random states and distributions
relating to state outcomes should be defined here and draws should also be
uniformly performed within the class. Special attention must be taken when
parallelizing agent decisions in shared environments to ensure random draws are
independent and state updates are shared by all nodes.

This is where most of the economic models will be implemented.
"""

import numpy as np

from scipy.stats import betabinom

from .utils import assert_type
from .agent import McCallAgent, HuggettAgent


class Environment:
    """Abstract Environment class."""

    def __init__(self):
        """Initialize Environment"""
        self.actions = [0]

        self._dist_config = None
        self._draw = np.random.random

    @property
    def shape(self):
        """Shape of the complete action space in this environment. This is
        mostly used for grid environments.
        """
        raise NotImplementedError()

    def draw(self):
        """Draw from distribution."""
        return self._draw()

    def set_dist(self, *args, **kwargs):
        """Initialize distribution. Necessary to enable picking of this class.

        This enables for effective model checkpointing and initializing
        independent distributions in distributed nodes.
        """
        self._dist_config = (args, kwargs)

    def dump_dist(self):
        """Dump distribution for future re-use."""
        self._draw = None

    def load_dist(self):
        """Load dumped distribution."""
        args, kwargs = self._dist_config
        self.set_dist(*args, **kwargs)

    def reset(self, agent):
        """Reset model given an instanciated agent."""
        raise NotImplementedError()

    def step(self, agent, s, a):
        """Take a step in the model given an instanciated agent."""
        raise NotImplementedError()

    def possible_actions(self, agent):
        """Possible actions available to an instanciated agent."""
        return self.actions

    @staticmethod
    def _check_agent(agent):
        """Check for compatibility of an instanciated agent to this
        Environment.

        This is important as agents are implemented to conform to specific
        environment mechanics.
        """
        pass


class BaseMcCall(Environment):
    """McCall model of information and job search.

    Based on the 1970 paper by J. J. McCall.
    https://doi.org/10.2307/1879403

    Agents make decisions based on government benefits, salary prospects and a
    time discount rate.
    """

    def __init__(self, max_wage, c=25):
        super().__init__()

        self.max_wage = max_wage
        self.c = c

        self.actions = [0, 1]

        self.set_dist(200, 100)

    @property
    def shape(self):
        # must account for wage proposals and earning wage
        return (self.max_wage*2, 2)

    def set_dist(self, *args, **kwargs):
        super().set_dist(*args, **kwargs)
        self._draw = lambda: betabinom(self.max_wage,
                                       *args, **kwargs).rvs(1)[0]

    def reset(self, agent):
        return self.draw(), 0, False

    @staticmethod
    def _check_agent(agent):
        assert_type(agent, McCallAgent, var_name="agent")


class McCallModel(BaseMcCall):

    def step(self, agent, s, a):

        self._check_agent(agent)

        if a == 0:  # accept
            if np.isfinite(agent.death_age):
                lv = np.sum((agent.gamma**np.arange(agent.death_age
                                                    - agent.age + 1))*s)
            else:
                lv = s / (1 - agent.gamma)
            return s+self.max_wage, lv, True
        elif agent.age < agent.death_age:  # reject and live on
            return self.draw(), self.c, False
        else:  # reject and die
            return -1, self.c, True


class FullMcCallModel(BaseMcCall):

    @property
    def shape(self):
        # must account for wage proposals and earning wage
        return (self.max_wage*2, 2)

    def step(self, agent, s, a):

        self._check_agent(agent)

        if a == 0:  # accept
            # accepting states are after rejecting wages
            if s > self.max_wage:  # keep job
                return s, s-self.max_wage, False
            else:  # take job
                return self.max_wage+s, s, False
        else:  # reject
            return self.draw(), self.c, False


class HuggettModel(Environment):
    """Huget model of consumption and savings.

    Based on the 1993 paper by Mark Hugget
    https://doi.org/10.1016/0165-1889(93)90024-M

    Agents gain some stochastic income at every turn and chose some level of
    consumption. Savings (or debt) is accrued with consumption, earning (or
    costing) some interest rate.
    """

    def __init__(self, max_assets, max_debt, r=0.1):
        super().__init__()

        self.max_assets = max_assets
        self.max_debt = max_debt
        self.R = 1 + r

        self.actions = np.arange(0, max_assets+max_debt+1)

        self.set_dist(200, 100)

    @property
    def shape(self):
        # You can consume as much as you want.
        # First (max_debt) indices are for negative income
        # Add an extra index to account for zero.
        return (self.max_assets+self.max_debt+1,)*2

    def set_dist(self, *args, **kwargs):
        super().set_dist(*args, **kwargs)
        self._draw = lambda: betabinom(self.max_assets // 10,
                                       *args, **kwargs).rvs(1)[0]

    def reset(self, agent):
        agent.a = self.draw()
        # we add max_debt to account for negative income
        return self.max_debt + agent.a, 0, False

    def step(self, agent, state, action):

        self._check_agent(agent)

        # Keep in mind that agent.a are the agent's assets
        # Can't consume more than your assets and the maximum debt
        action = min(action, agent.a + self.max_debt)

        # Accrue/pay interest
        agent.a += self.R*(agent.a - action)
        # Get a salary draw
        agent.a += self.draw()
        # Assets must be an integer
        agent.a = int(agent.a)

        # Assets are capped
        agent.a = min(agent.a, self.max_assets)
        agent.a = max(agent.a, 0)

        return self.max_debt + agent.a, agent.utility(action), False

    def possible_actions(self, agent):
        return self.actions[self.actions <= (agent.a + self.max_debt)]

    @staticmethod
    def _check_agent(agent):
        assert_type(agent, HuggettAgent, var_name="agent")
