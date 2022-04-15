import os
import sys
import csv

import argparse as ap
import subprocess as sp

from collections import defaultdict
from functools import partial

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from .utils import assert_type


class LoggerType:

    def __init__(self):
        self.reset_buf()

    def reset_buf(self):
        self.buf = list()

    def add_item(self, item):
        self.buf.append(item)

    def log(self):
        self.reset_buf()

    def add_and_log(self, item):
        self.add_item(item)
        self.log()


class LogFile(LoggerType):
    def __init__(self, file_name, mode="w"):

        super().__init__()

        self.new_file = False
        if isinstance(file_name, str):
            self.f = open(file_name, mode)
            self.new_file = True
        elif hasattr(file_name, "write"):
            self.f = file_name
        else:
            raise TypeError("file_name must be either a file name "
                            "or have a 'write' attribute")

    def log(self):
        for item in self.buf:
            self.f.write(item)
        super().log()

    def __del__(self):
        if self.new_file:
            self.f.close()


class LogPrint(LogFile):
    def __init__(self):
        super().__init__(sys.stdout)


class LogSave(LoggerType):
    def __init__(self):
        super().__init__()
        self.log_list = list()

    def log(self):
        for item in self.buf:
            self.log_list.append(item)
        super().log()


class AggLog(LoggerType):

    agg_dict = {
        "sum": partial(np.sum),
        "mean": partial(np.mean),
        "std": partial(np.std),
        "col_sum": partial(np.sum, axis=0),
        "col_mean": partial(np.mean, axis=0),
        "col_std": partial(np.std, axis=0),
    }

    def __init__(self, logger, agg_func):
        super().__init__()
        assert_type(logger, LoggerType, "logger")
        self.logger = logger

        if isinstance(agg_func, str):
            if agg_func not in AggLog.agg_dict:
                raise ValueError("agg_func preset not available, please "
                                 "choose one of ",
                                 ", ".join(AggLog.agg_dict.keys()))
            agg_func = AggLog.agg_dict[agg_func]

        if not callable(agg_func):
            raise TypeError("agg_func must be callable")
        self.agg_func = agg_func

    def add_item(self, item):
        assert_type(item, (float, int, np.ndarray), "item")
        super().add_item(item)

    def log(self):
        agg = self.agg_func(np.array(self.buf))
        self.logger.add_and_log(agg)
        super().log()


class LoggingVisitor:

    def __init__(self):
        self.type_logs = defaultdict(lambda: list())
        self.all_loggers = list()

    def is_empty(self):
        return self.all_loggers == []

    def add_log(self, obj_type, logger, func, tag=None):
        assert_type(obj_type, type, "obj_type")
        assert_type(logger, LoggerType, "logger")
        if not callable(func):
            raise TypeError("func must be callable")

        self.type_logs[obj_type].append((logger, func, tag))
        self.all_loggers.append(logger)

    def get_data(self, obj, tag=None):
        for t, ops in self.type_logs.items():
            if isinstance(obj, t):
                for logger, func, op_tag in ops:
                    if op_tag == tag:
                        logger.add_item(func(obj))

    def __call__(self):
        for logger in self.all_loggers:
            logger.log()


class LivePlotter:

    def __init__(self, config, fig=None):

        assert_type(config, (tuple, list), "config")

        self.plotting_opts = list()

        axs = None
        new_fig = fig is None
        if new_fig:
            self.fig, axs = plt.subplots(len(config))
        else:
            self.fig = fig

        for i, plt_conf in enumerate(config):
            path = plt_conf.get("path", None)
            if path is None:
                raise ValueError("All config entries must have a "
                                 "path entry")
            else:
                assert_type(path, str, "path")
                del plt_conf["path"]

            ax = plt_conf.get("ax", None)
            if ax is None:
                if not new_fig:
                    raise ValueError("You must either specify all axes "
                                     "from an existing figure or no axes")
                ax = axs[i]
            else:
                if new_fig:
                    raise ValueError("You must either specify all axes "
                                     "from an existing figure or no axes")
                assert_type(ax, Axes, "ax")
                del plt_conf["ax"]

            ax_fn = plt_conf.get("ax_fn", None)
            if ax_fn is None:
                def ax_fn(x):
                    pass
            else:
                if not callable(ax_fn):
                    ValueError("ax_fn must be callable")
                del plt_conf["ax_fn"]

            plt_opts = plt_conf.get("plt_opts", None)
            if plt_opts is None:
                plt_opts = plt_conf
            else:
                assert_type(plt_opts, dict, "plt_opts")

            self.plotting_opts.append((path, ax, ax_fn, plt_opts))

        self.live_data = None

    def _existent_paths(self):
        for path, _, _, _ in self.plotting_opts:
            yield os.path.exists(path)

    def _create_parser(self):
        self.parser = ap.ArgumentParser(description="Execute Python programs "
                                                    "and an animation")
        self.parser.add_argument("programs", metavar='P', type=str, nargs='+',
                                 help="programs to execute with animation")
        self.parser.add_argument("--py_exec", type=str, default="python",
                                 help="Python execution program")

    def update_graph(self, _):

        used_axs = set()
        for name, ax, ax_fn, plot_opts in self.plotting_opts:
            last_row = np.array([])
            with open(name, 'r') as f:
                reader = csv.reader(f)
                for last_row in reader:
                    pass

            last_row = np.array(last_row, dtype=np.float64)
            rng = np.arange(len(last_row))

            # enable axes to get plotted over
            if ax not in used_axs:
                ax.clear()
                used_axs.add(ax)

            ax.plot(rng, last_row, **plot_opts)

            if "label" in plot_opts:
                ax.legend()

            ax_fn(ax)

    def run(self):

        # generators are faster than list comprehensions
        if not all(self._existent_paths()):
            raise ValueError("all files must exist by the time a LivePlotter "
                             "is running")

        # Create the actual animation object that updates the graph
        self.live_data = FuncAnimation(self.fig,
                                       self.update_graph,
                                       interval=200)

        # Show the graph.
        plt.show()

    def exec(self):

        print("Executing Animation")

        self._create_parser()
        args = self.parser.parse_args()

        po_dict = dict()
        for prog in args.programs:
            if prog in po_dict:
                ValueError("You must specify each program only once")
            po_dict[prog] = sp.Popen([args.py_exec, prog])

        self.run()
        print("Animation Terminated")

        # TODO: Use a thread pool to manage this.
        for prog, po in po_dict.items():
            po.wait()
            print(f"{prog} Terminated")
