import sys
sys.path.insert(0, "..")

import numpy as np

from dyna_aco.environment import McCallModel, HuggettModel
from dyna_aco.agent import McCallAgent, HuggettAgent
from dyna_aco.optimizer import DynaQACO
from dyna_aco.model import DynaACO

dyna_params = {
    "alpha": 0.9,
    "eps": 0.1,
    "planning_steps": 50
}

mcm = McCallModel(60, c=25)
mcm_agent = McCallAgent(np.inf, gamma=0.9)

mcm_aco_dyna = DynaACO(mcm, time_weight=1e-4)
mcm_aco_dyna_q = DynaQACO(mcm, mcm_agent, mcm_aco_dyna, 10, **dyna_params)

fit_params = {
    "iters": int(1e2),
    "run_len": 50,
    "verbose": True
}

mcm_aco_dyna_q.fit(**fit_params)

hug = HuggettModel(20, 5, r=0.1)
agent = HuggettAgent(gamma=0.99, crra_gamma=1.5)

aco_dyna = DynaACO(hug, time_weight=1e-4)
aco_dyna_q = DynaQACO(hug, agent, aco_dyna, 10, **dyna_params)

aco_dyna_q.fit(**fit_params)

print("DONE")
