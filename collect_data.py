import numpy as np
import matplotlib.pyplot as plt

from environment import McCallModel, HuggettModel
from agent import McCallAgent, HuggettAgent
from optimizer import QLearning, DynaQ, dump_opt
from model import (VanillaDyna, TimeDyna,
                   NonDetermVanillaDyna, NonDetermTimeDyna)


mcm = McCallModel(60, c=25)
mcm_agent = McCallAgent(np.inf, gamma=0.99)

hug = HuggettModel(20, 5, r=0.1)
hug_agent = HuggettAgent(gamma=0.99, crra_gamma=1.5)

mcm_q_learning = QLearning(mcm, mcm_agent, alpha=0.9, eps=0.1)
hug_q_learning = QLearning(hug, hug_agent, alpha=0.9, eps=0.1)

dyna_params = {
    "alpha": 0.9, 
    "eps": 0.1,
    "planning_steps": 50
}

mcm_vanilla_dyna = VanillaDyna(mcm)
mcm_dyna_q = DynaQ(mcm, mcm_agent, mcm_vanilla_dyna, **dyna_params)

hug_vanilla_dyna = NonDetermVanillaDyna(hug)
hug_dyna_q = DynaQ(hug, hug_agent, hug_vanilla_dyna, **dyna_params)

mcm_time_dyna = TimeDyna(mcm, time_weight=1e-4)
mcm_time_dyna_q = DynaQ(mcm, mcm_agent, mcm_time_dyna, **dyna_params)

hug_time_dyna = NonDetermTimeDyna(hug, time_weight=1e-4)
hug_time_dyna_q = DynaQ(hug, hug_agent, hug_time_dyna, **dyna_params)

fit_params = {
    "iters": 10, #int(1e4),
    "run_len": 20,
    "verbose": True
}

mcm_q_learning.fit(**fit_params)
res = mcm_q_learning.test(10, 10)
dump_opt(mcm_q_learning, "mcm_q_learning")

mcm_dyna_q.fit(**fit_params)
dump_opt(mcm_dyna_q, "mcm_dyna_q")

mcm_time_dyna_q.fit(**fit_params)
dump_opt(mcm_time_dyna_q, "mcm_time_dyna_q")

hug_q_learning.fit(**fit_params)
dump_opt(hug_q_learning, "hug_q_learning")

hug_dyna_q.fit(**fit_params)
dump_opt(hug_dyna_q, "hug_dyna_q")

hug_time_dyna_q.fit(**fit_params)
dump_opt(hug_time_dyna_q, "hug_time_dyna_q")

print("DONE")