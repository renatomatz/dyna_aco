import numpy as np


from environment import McCallModel
from agent import McCallAgent
from optimizer import QLearning, DynaQ
from model import VanillaDyna, TimeDyna


mcm = McCallModel(60, c=25)
agent = McCallAgent(np.inf, gamma=0.99)

q_learning = QLearning(mcm, agent, alpha=0.9, eps=0.1)

dyna_params = {
    "alpha": 0.9, 
    "eps": 0.1,
    "planning_steps": 50
}

vanilla_dyna = VanillaDyna(mcm)
dyna_q = DynaQ(mcm, agent, vanilla_dyna, **dyna_params)

time_dyna = TimeDyna(mcm, time_weight=1e-4)
time_dyna_q = DynaQ(mcm, agent, time_dyna, **dyna_params)

ITERS = 50
q_learning.fit(ITERS)
dyna_q.fit(ITERS)
time_dyna_q.fit(ITERS)

print("DONE")
