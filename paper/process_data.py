import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt

from dyna_aco.optimizer import load_opt


mcm_q_learning = load_opt("mcm_q_learning")
mcm_dyna_q = load_opt("mcm_dyna_q")
mcm_time_dyna_q = load_opt("mcm_time_dyna_q")
hug_q_learning = load_opt("hug_q_learning")
hug_dyna_q = load_opt("hug_dyna_q")
hug_time_dyna_q = load_opt("hug_time_dyna_q")

print("DONE")
