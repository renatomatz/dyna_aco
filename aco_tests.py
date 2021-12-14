from environment import HuggettModel
from agent import HuggettAgent
from optimizer import DynaQACO
from model import DynaACO

dyna_params = {
    "alpha": 0.9,
    "eps": 0.1,
    "planning_steps": 50
}

hug = HuggettModel(20, 5, r=0.1)
agent = HuggettAgent(gamma=0.99, crra_gamma=1.5)

aco_dyna = DynaACO(hug, time_weight=1e-4)
aco_dyna_q = DynaQACO(hug, agent, aco_dyna, 10, **dyna_params)

fit_params = {
    "iters": 1000,
    "run_len": 20,
    "verbose": True
}

aco_dyna_q.fit(**fit_params)

print("DONE")
