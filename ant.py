"""Define Ant classes, which generally model agent behaviours.

Ants are respondible for making decisions (traversing the state space) and
updating world pheromones. The way on which these decisions and updates are
made are up to the programmer, and can represent any kind of intuition or
mechanic necessary. Ants should not change the world in any way other than
updating its pheromones.

Other classes are defined here to allow parallel executions of the algorithm to
have a similar interface as sequential executions.
"""

import numpy as np

from collections.abc import Iterable
from collections import defaultdict
from functools import partial

from utils import _jitted_discounted_lifetime, _jitted_reward_accumulate, \
                  assert_type


class Ant:
    """Base Ant class.

    This class contains the basic parameters from the original ACO
    specification as well as some adaptations for this application.
    """

    def __init__(self, death_age, gamma=0.99,
                 alpha=1.0, beta=0.0, rho=0.2,
                 decision=None, update=None):
        """Initialize ant.
        """
        self.death_age = death_age
        self.gamma = gamma

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.decide = decision
        self.update = update

        self.age = 0
        self.a = 0
        self.history = list()

    @property
    def is_dead(self):
        """Return whether the ant is dead"""
        return self.age >= self.death_age

    def adjust_beliefs(self, beliefs, pheromones):
        """Adjust an array of beliefs to be influenced by an array of
        pheromones according to this ant's parameters.
        """
        weighted = (pheromones ** self.alpha) * (beliefs ** self.beta)
        return weighted / np.sum(weighted)

    def to_date_utility(self):
        """Return total lifetime value of this ant."""
        return _jitted_discounted_lifetime(np.array(self.history,
                                                    dtype=np.float64)[:, 2],
                                           len(self.history),
                                           self.gamma)

    @property
    def decide(self):
        """Make a decision based on data available in a World object."""
        if self._decision is None:
            raise AttributeError("You have not assigned an Decide object "
                                 "to this attribute")
        return partial(self._decision, self)

    @decide.setter
    def decide(self, obj):
        assert_type(obj, Decision, "decide", or_none=True)
        self._decision = obj

    @property
    def update(self):
        """Update the pheromone matrix of a World based on this ant's
        experience.
        """
        if self._update is None:
            raise AttributeError("You have not assigned an Update object "
                                 "to this attribute")
        return partial(self._update, self)

    @update.setter
    def update(self, obj):
        assert_type(obj, Update, "update", or_none=True)
        self._update = obj

    def play(self, data, model):
        s, r, done = model.reset(self)
        while not done:
            self.age += 1
            a = self.decide(s, data, model)
            self.history.append((s, a, r))
            s, r, done = model.step(self, s, a)

    @classmethod
    def static_generator(cls, *args, **kwargs):
        """Generate new ants with the same set of initializing arguments."""

        while True:
            yield cls(*args, **kwargs)

    @classmethod
    def arg_evaporation_generator(cls, arg_evap,
                                  *args, **kwargs):
        """Generate new ants with one or more arguments evaporating at a
        constant rate.
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


class Decision:
    def __call__(self, ant, s, data, model):
        raise NotImplementedError()


class EGreedyDeterministicDecision(Decision):
    """Decision based on information about discrete states.

    This ant makes their choices based on the best alternative amongst various
    states. It has extra parameters for extra exploration of the state space
    (to avoid getting stuck on local minima) and the possibility of getting
    stuck on certain states for the rest of its life.
    """

    def __init__(self, eps=1.0):
        self.eps = eps

    def __call__(self, ant, s, data, model):
        """Make decision based on the merits of the expected and instantaneous
        utilities from a decision.

        Stuck ants will simply pick the state on which they were stuck on.
        Unstuck ants will evaluate their best options and pick, possibly
        decising to get stuck. Options are selected in a greedy way by default
        but stochastically if the self.eps option is specified and
        randomly picked.
        """
        q = ant.adjust_beliefs(data.q_beliefs, data.q_pheromones)
        v = q[s]

        # model exploration: pick an option stochastically weighted by
        # expected values.
        if np.random.random() > self.eps:
            # should this be a softmax instead?
            a = np.random.choice(np.arange(v.shape[0]))
        else:
            # This randomly selects option if values are the same
            a = np.random.choice(np.flatnonzero(v == v.max()))

        return a


class Update:

    def __call__(self, ant, old_pher, new_pher, rho_tot):
        raise NotImplementedError()


class MonteCarloUpdate(Update):

    def __call__(self, ant, old_pher, new_pher, rho_tot):
        """Update lifetime value expectations and state probabilities based on
        all states visited by the ant (until stuck) and total value from first
        state.

        Updates are made based on deviations from the world's preconceptions.
        Lifetime values with significant deviations from general beliefs will
        have a larger impact on the pheromone values.
        """

        # TODO: Make all this jitted
        hist_mat = np.array(ant.history)
        flipped_rewards = np.flip(hist_mat[:, 2]).astype(np.float64)
        hist_mat[:, 2] = np.flip(_jitted_reward_accumulate(flipped_rewards,
                                                           ant.gamma))

        visits = defaultdict(lambda: 0)
        for s, a, g in hist_mat:
            visits[s] += 1

            if visits[s] == 1:
                new_pher[s, a] += (ant.rho * (g - old_pher[s, a]))
                rho_tot[s, a] += ant.rho


class SamplingQueue:

    def __init__(self, shape):
        self.shape = shape
        self.maxsize = shape[0]
        self.data = np.empty(shape)
        self.len = 0
        self.i = 0

    @property
    def next_i(self):
        i, self.i = self.i, (self.i + 1) % self.maxsize
        return i

    def put_iter(self, item):
        assert_type(item, Iterable, "item")
        for it in item:
            self.put(it)

    def put(self, item):
        self.data[self.next_i] = item
        self.len = min(self.len + 1, self.maxsize)

    def sample(self, size=1):
        return self.data[np.random.randint(self.len, size=size)]


class QLearningWithReplayBuffer(Update):

    def __init__(self, batch_size, buf_size=1000):
        self.batch_size = batch_size
        self.buf_size = buf_size
        self.replay_buffer = SamplingQueue([self.buf_size, 4])

    def __call__(self, ant, old_pher, new_pher, rho_tot):

        # last decision has sp = None
        self.replay_buffer.put_iter(ant.history)

        batch = self.replay_buffer.sample(size=self.batch_size)
        for s, a, sp, r in batch:
            s, a = int(s), int(a),

            if not np.isnan(sp):
                update = r + ant.gamma*np.max(old_pher[int(sp), :])
            else:
                update = r

            new_pher[s, a] += ant.rho*(update - old_pher[s, a])
