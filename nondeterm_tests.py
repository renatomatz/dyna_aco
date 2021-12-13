import numpy as np


from environment import HuggettModel
from agent import HuggettAgent
from optimizer import QLearning, DynaQ
from model import NonDetermVanillaDyna, NonDetermTimeDyna


hug = HuggettModel(60, 20, r=0.1)
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

ITERS = 50
q_learning.fit(ITERS)
dyna_q.fit(ITERS)
time_dyna_q.fit(ITERS)

print("DONE")