"""Define Model classes, which generally models value assignment.

Models are what ants use to assign values to states based on their expectations
and instantaneous rewards. This are where most of the economic models will be
implemented, as they define how agents make choices.

Models will additionally define the lifetime value of a certain history of
model choices. This is necessary for pheromone updates as well as being useful
statistics.
"""

import numpy as np


class Model:
    """Abstract Model class."""

    def reset(self, ant):
        raise NotImplementedError()

    def step(self, ant, s, a):
        raise NotImplementedError()


class BaseMcCall(Model):
    """McCall model of information and job search.

    Based on the 1970 paper by J. J. McCall.
    https://doi.org/10.2307/1879403

    Agents make decisions based on government benefits, salary prospects and a
    time discount rate.
    """

    def __init__(self, draw_salary_func, c=25):
        self.draw_salary = draw_salary_func
        self.c = c

        self.actions = [0, 1]

    def reset(self, ant):
        return self.draw_salary(), 0, False


class McCallModel(BaseMcCall):

    def step(self, ant, s, a):
        if a == 0:  # accept
            if np.isfinite(ant.death_age):
                lv = np.sum((ant.gamma**np.arange(ant.death_age
                                                  - ant.age + 1))*s)
            else:
                lv = s / (1 - ant.gamma)
            return -1, lv, True
        elif ant.age < ant.death_age:  # reject and live on
            return self.draw_salary(), self.c, False
        else:  # reject and die
            return -1, self.c, True


class FullMcCallModel(BaseMcCall):
    """Too close to a bandit scenario
    """

    def step(self, ant, s, a):
        if a == 0:  # accept
            return s, s, False
        else:
            return self.draw_salary(), self.c, False


class HuggetModel(Model):
    """Huget model of consumption and savings.

    Based on the 1993 paper by Mark Hugget
    https://doi.org/10.1016/0165-1889(93)90024-M

    Agents gain some stochastic income at every turn and chose some level of
    consumption. Savings (or debt) is accrued with consumption, earning (or
    costing) some interest rate.
    """

    def __init__(self, draw_income_func, max_assets, max_debt, r=0.1):
        self.draw_income = draw_income_func
        self.max_debt = max_debt
        self.R = 1 + r

        self.actions = np.arange(0, max_assets+max_debt+1)

    def reset(self, ant):
        return self.draw_income(), 0, False

    def step(self, ant, s, a):

        # Keep in mind that ant.a are the agent's assets
        # Can't consume more than your assets and the maximum debt
        a = min(a, ant.a + self.max_debt)

        # Accrue/pay interest
        ant.a += self.R*(self.a - a)
        # Get a salary draw
        ant.a += self.draw_income()
        ant.a = min(ant.a, self.max_assets)

        return ant.a, ant.utility(a), False

    def action_mask(self, ant):
        return self.actions <= (ant.a + self.max_debt)
