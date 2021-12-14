import numpy as np
import matplotlib.pyplot as plt

from environment import HuggettModel
from agent import HuggettAgent
from optimizer import QLearning, DynaQ, dump_opt
from model import NonDetermVanillaDyna, NonDetermTimeDyna


hug = HuggettModel(20, 5, r=0.1)
agent = HuggettAgent(gamma=0.99, crra_gamma=1.5)

q_learning = QLearning(hug, agent, alpha=0.9, eps=0.1)

dyna_params = {
    "alpha": 0.9,
    "eps": 0.1,
    "planning_steps": 50
}

vanilla_dyna = NonDetermVanillaDyna(hug)
dyna_q = DynaQ(hug, agent, vanilla_dyna, **dyna_params)

time_dyna = NonDetermTimeDyna(hug, time_weight=1e-4)
time_dyna_q = DynaQ(hug, agent, time_dyna, **dyna_params)

fit_params = {
    "iters": int(1e4),
    "run_len": 20,
    "verbose": True
}

q_learning.fit(**fit_params)

dyna_q.fit(**fit_params)

time_dyna_q.fit(**fit_params)

print("DONE")