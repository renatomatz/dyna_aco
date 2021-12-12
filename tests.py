import numpy as np

from scipy.stats import betabinom

from model import McCallModel
from ant import Ant, EGreedyDeterministicDecision, MonteCarloUpdate, \
                QLearningWithReplayBuffer
from optimizer import CUDATabularQOptimizer, \
                      MPITabularQOptimizer, \
                      TabularQOptimizer
import kernels

from log_helpers import LoggingVisitor, AggLog, LogPrint, LogFile
from utils import iter_to_str


T = kernels.DEATH_AGE
n_births = 100
mc_beta = 0.99
max_wage = kernels.STATES

logger = LoggingVisitor()
logger.add_log(TabularQOptimizer,
               LogPrint(),
               lambda it_opt: f"Iteration: {it_opt.iters}")
logger.add_log(Ant,
               AggLog(LogPrint(),
                      lambda utilities: " | Average Utility: "
                                        f" {str(np.mean(utilities))}\n"),
               lambda it_ant: it_ant.to_date_utility())
logger.add_log(TabularQOptimizer,
               LogFile("media/accept_q_pher.csv"),
               lambda it_opt: iter_to_str(it_opt.q_pheromones[:, 0]))
logger.add_log(TabularQOptimizer,
               LogFile("media/reject_q_pher.csv"),
               lambda it_opt: iter_to_str(it_opt.q_pheromones[:, 1]))
logger.add_log(TabularQOptimizer,
               LogFile("media/accept_q_belf.csv"),
               lambda it_opt: iter_to_str(it_opt.q_beliefs[:, 0]))
logger.add_log(TabularQOptimizer,
               LogFile("media/reject_q_belf.csv"),
               lambda it_opt: iter_to_str(it_opt.q_beliefs[:, 1]))
logger.add_log(TabularQOptimizer,
               LogFile("media/policy.csv"),
               lambda it_opt: iter_to_str(np.argmax(it_opt.q_beliefs,
                                                    axis=1)))

dist = betabinom(max_wage, 200, 100)
mcm = McCallModel(lambda: dist.rvs(1)[0], c=kernels.BENEFITS)

opt_kwargs = {
    "model": mcm,
    "n_births": n_births,
    "evaporation_rate": 0.01,
    "logger": logger
}


opt_t = "seq"

if opt_t == "mpi":
    opt = MPITabularQOptimizer(**opt_kwargs)
elif opt_t == "cuda":
    opt_kwargs["n_births"] = 60*2*1000

    logger_cuda = LoggingVisitor()
    logger_cuda.add_log(TabularQOptimizer,
                        LogPrint(),
                        lambda it_opt: f"Iteration: {it_opt.iters}")
    logger_cuda.add_log(float,
                        LogPrint(),
                        lambda utilities: f" | Average Utility: "
                                          f" {str(utilities)}\n",
                        tag="tot_util")
    logger_cuda.add_log(TabularQOptimizer,
                        LogFile("media/accept_q_pher.csv"),
                        lambda it_opt: iter_to_str(it_opt.q_pheromones[:, 0]))
    logger_cuda.add_log(TabularQOptimizer,
                        LogFile("media/reject_q_pher.csv"),
                        lambda it_opt: iter_to_str(it_opt.q_pheromones[:, 1]))
    logger_cuda.add_log(TabularQOptimizer,
                        LogFile("media/accept_q_belf.csv"),
                        lambda it_opt: iter_to_str(it_opt.q_beliefs[:, 0]))
    logger_cuda.add_log(TabularQOptimizer,
                        LogFile("media/reject_q_belf.csv"),
                        lambda it_opt: iter_to_str(it_opt.q_beliefs[:, 1]))
    logger_cuda.add_log(TabularQOptimizer,
                        LogFile("media/policy.csv"),
                        lambda it_opt: iter_to_str(np.argmax(it_opt.q_beliefs,
                                                             axis=1)))

    opt_kwargs["logger"] = logger_cuda

    opt = CUDATabularQOptimizer(**opt_kwargs)
    opt.load_module("ant", kernels)
else:
    opt = TabularQOptimizer(**opt_kwargs)

decision = EGreedyDeterministicDecision(eps=0.8)

monte_carlo = True
if monte_carlo:
    update = MonteCarloUpdate()
else:
    update = QLearningWithReplayBuffer(10, buf_size=T*n_births)

opt.ant_gen = Ant.static_generator(
    T, alpha=1.0, beta=0.5, rho=0.1,
    decision=decision,
    update=update
)

constant_beliefs = True
if constant_beliefs:
    opt.q_beliefs = np.full(
        [max_wage, 2],
        np.sum(dist.mean()*(mc_beta**np.arange(T)))
    )
else:
    opt.q_beliefs = np.zeros([max_wage, 2])
    opt.q_beliefs[:, 0] = (
        np.broadcast_to(np.arange(max_wage), [T, max_wage]).T
        @ (mc_beta**np.arange(T))
    )
    opt.q_beliefs[:, 1] = np.full(
        max_wage,
        np.sum(dist.mean()*(mc_beta**np.arange(T)))
    )

opt.reset_pheromones()

opt.fit(iters=100, n_procs=4)
