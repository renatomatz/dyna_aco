from collections.abc import Iterable

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from numba import njit, guvectorize, float64


class _NamespaceHandler:

    def __init__(self):
        self._data = None
        self.req_vars = list()
        self.present_vars = list()

    def set_data_attr(self, name, obj):
        self._data.__setattr__(name, obj)
        self.present_vars.append(name)

    def del_data_attr(self, name):
        self.set_data_attr(name, None)
        self.present_vars.remove(name)

    def check_req_vars(self):
        missing = set(self.req_vars).difference(set(self.present_vars))
        if len(missing) > 0:
            raise ValueError("required variable check failed, missing: "
                             ", ".join(missing))

    def clear_req_vars(self):
        for name in self.req_vars:
            self.del_data_attr(name)


class _AntHandler(mp.Process, _NamespaceHandler):
    """Process that handles ants in a parallel environment.

    This class serves the purpose of handling ants when executing the
    algorithm in parallel. It contains necessary structures for parallel
    coordination between processes and data transfers to the master
    process. It also ensures that random draws are different between each
    process and that the sequential interface to the algorithm is
    preserved.
    """

    handlers = 0

    def __init__(self, data, model, comm, lock, barrier):
        """Initialize an AntHandler Process object.
        """
        super(mp.Process, self).__init__()
        super(_NamespaceHandler, self).__init__()

        self._data = data
        self.model = model

        self.comm = comm
        self.lock = lock
        self.barrier = barrier

        type(self).handlers += 1
        self.seed = type(self).handlers


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


def cycle_gen(n):
    i = 0
    while True:
        yield i
        i = (i + 1) % n


def assert_type(res, tgt, var_name=None, or_none=False):
    if not (isinstance(res, tgt) or (or_none and res is None)):
        if not isinstance(tgt, tuple):
            tgt = (tgt,)
        msg = f"Variable {var_name} must be of type "
        msg += ", ".join([t.__name__ for t in tgt])
        if or_none:
            msg += " or None"
        msg += f" not {type(res).__name__}"
        raise TypeError(msg)


def iter_to_str(it, end='\n'):
    return ", ".join([str(item) for item in it]) + end


def plot_log_history(history, stat, start_at=0, ax=None):
    """
    Plot a sequence of value functions.
        * self is an instance of McCallModel
        * ax is an axes object that implements a plot method.
    """

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.plot(np.arange(len(history) - start_at),
            history[start_at:],
            '-', alpha=0.4, label=stat)
    ax.legend(loc='lower right')

    return fig, ax


def plot_arr(arr, ax=None, **plot_opts):
    """Plot the current probability pheromones"""

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.plot(np.arange(arr.shape[0]), arr, **plot_opts)
    ax.legend(loc='upper right')

    return fig, ax


# Jitted functions
@njit
def _jitted_discounted_lifetime(arr, n, gamma):
    return np.sum(arr*(gamma**np.gamma(n)))


@njit
def _jitted_update_prob_pher(rho, state_history, prob_pheromones):

    for i in state_history:
        prob_pheromones[i] += rho

    prob_pheromones /= np.sum(prob_pheromones)
    return prob_pheromones


@guvectorize([(float64[:], float64, float64[:])], '(n),()->(n)')
def _jitted_reward_accumulate(x, y, res):
    res[0] = x[0]
    for i in range(1, x.shape[0]):
        res[i] = x[i] + y*res[i-1]
