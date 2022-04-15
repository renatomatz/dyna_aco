import sys
sys.path.insert(0, "..")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .log_helpers import LivePlotter


fig = plt.figure()
gs = GridSpec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])


plotter = LivePlotter([
    {"path": "media/accept_q_pher.csv",
     "ax": ax1,
     "ax_fn": lambda ax: ax.set_title("Accept Values")},
    {"path": "media/accept_q_belf.csv",
     "ax": ax1},
    {"path": "media/reject_q_pher.csv",
     "ax": ax2,
     "ax_fn": lambda ax: ax.set_title("Reject Values")},
    {"path": "media/reject_q_belf.csv",
     "ax": ax2},
    {"path": "media/policy.csv",
     "ax": ax3,
     "ax_fn": lambda ax: ax.set_title("Policy From Pheromones")},
], fig=fig)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plotter.exec()
    plotter.run()
