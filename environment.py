"""Define Environment classes, which generally models value assignment.

Models are what agents use to assign values to states based on their
expectations and instagentaneous rewards. This are where most of the
economic models will be implemented, as they define how agents make choices.

Models will additionally define the lifetime value of a certain history of
model choices. This is necessary for pheromone updates as well as being useful
statistics.
"""

import numpy as np

from scipy.stats import betabinom

from utils import assert_type
from agent import McCallAgent, HuggettAgent


class Environment:
    """Abstract Environment class."""

    def __init__(self):
        self.actions = [0]
        self._draw = np.random.random

    @property
    def shape(self):
        raise NotImplementedError()

    def draw(self):
        return self._draw()

    def set_dist(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self, agent):
        raise NotImplementedError()

    def step(self, agent, s, a):
        raise NotImplementedError()

    def possible_actions(self, agent):
        return self.actions

    @staticmethod
    def _check_agent(agent):
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
    """Too close to a bandit scenario
    """

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


class HuggetModel(Environment):
    """Huget model of consumption and savings.

    Based on the 1993 paper by Mark Hugget
    https://doi.org/10.1016/0165-1889(93)90024-M

    Agents gain some stochastic income at every turn and chose some level of
    consumption. Savings (or debt) is accrued with consumption, earning (or
    costing) some interest rate.
    """

    def __init__(self, max_assets, max_debt, r=0.1):

        self.max_assets = max_assets
        self.max_debt = max_debt
        self.R = 1 + r

        self.actions = np.arange(0, max_assets+max_debt+1)

        self.set_dist(200, 100)

    @property
    def shape(self):
        # You can consume as much as you want.
        return (self.max_assets+self.max_debt)*2

    def set_dist(self, *args, **kwargs):
        self._draw = lambda: betabinom(self.max_assets // 10,
                                       *args, **kwargs).rvs(1)[0]

    def reset(self, agent):
        return self.draw(), 0, False

    def step(self, agent, s, a):

        self._check_agent(agent)

        # Keep in mind that agent.a are the agent's assets
        # Can't consume more than your assets and the maximum debt
        a = min(a, agent.a + self.max_debt)

        # Accrue/pay interest
        agent.a += self.R*(self.a - a)
        # Get a salary draw
        agent.a += self.draw()
        agent.a = min(agent.a, self.max_assets)

        return agent.a, agent.utility(a), False

    def possible_actions(self, agent):
        return self.actions[self.actions <= (agent.a + self.max_debt)]

    @staticmethod
    def _check_agent(agent):
        assert_type(agent, HuggettAgent, var_name="agent")
