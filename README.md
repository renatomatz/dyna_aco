# Dyna-Q Ant Colony Optimization

This library contains the code used in the paper _Modeling Social Learning Using Dyna-Q and Ant Colony Optimization_.
The following is its introduction:

The way humans perceive their environment dictates how they make decisions;
this perception is ever-changing, based on previous conceptions and influenced
by personal experience.
Transitioning from one belief to another is often noisy and subject to the
whims of those who influence public information the most.
The rational expectations (RE) framework dominated macroeconomic
models during late twentieth century, and to a large extent remain the
predominant technique used in modern models.
However, the the theory treats agents as perfectly informed of their
environment's structure, distributions and dynamics, which allows for a more
direct analysis of stable points in the model.
Bounded rationality models in macroeconomics have tried to supplement RE models
by introducing limits to agents' forecasting abilities.
Adaptive learning (AL) has become the most prominent
amongst bounded rationality models, although several other approaches to
modeling learning have been developed.

This paper aims to extend the branch of AL known as _social learning_,
first developed by Arifovic.
Social learning aims to model how one agent learns in the context of other
agents' decisions as well as their own.
We contribute to the literature by using the Dyna-Q model-based
architecture to represent societal beliefs with a
distributed fitting approach inspired by Ant Colony Optimization.
The techniques proposed here also enable models that connect both social and
individual learning, unlike previous techniques which compromised on one
approach.

We also explore how individual learning can take place without full knowledge
of the environment's dynamics and rewards.
Although dynamic programming methods yield quick and unbiased solutions, the
procedure's transition from one state of belief to another is not informative
in and of itself.
The methods outlined in this paper maintain key properties of commonly used
dynamic programming techniques while introducing decentralized information
dynamics based on socially-learned models and possible channels for
misinformation.
We believe that our work will enable a more realistic study of how autonomous
agents influence the dissemination of information based on experience in an
environment with unknown dynamics.
