# fortbo roadmap

`fortbo` is the Bayesian-optimization layer for the lazy-fortran stack. It
owns acquisition functions, sequential design, constrained candidate search,
batch/multi-objective policies, and regret benchmarks. FortML owns surrogate
models and posterior contracts. FortOpt owns local constrained optimizers.
FortNum owns arrays, RNG, and linear algebra. FortAD supplies derivatives and
FortSym supplies proven generated kernels. FortMC is an optional dependency
for fully Bayesian surrogate and Thompson-sampling policies.

All code is MIT licensed. The pre-1.0 API may change without compatibility
shims. A roadmap checkbox is complete only when the API, implementation,
independent oracle, documentation, device refusal behavior, and benchmark
record exist.

## Architecture contract

| Concern | Owner | Required interface |
| --- | --- | --- |
| Surrogate model | FortML | Posterior mean/variance, joint samples, predictive covariance, fit/update, parameter registry |
| Probability and derivatives | FortML + FortAD/FortSym | Log density, reparameterized posterior samples, input/parameter JVP/VJP/HVP products |
| Candidate optimization | FortOpt | Bounded L-BFGS-B, multistart, line search, constraints, stopping diagnostics |
| Numerics and randomness | FortNum | Stable reductions, factorizations, seeded/splittable streams, device-safe arrays |
| Sequential design | FortBO | Acquisition value/derivatives, batch policy, constraints, objective/cost bookkeeping |
| Fully Bayesian sampling | FortMC | Posterior chains and diagnostics for Thompson, integrated acquisition, and model averaging |

Acquisition functions must expose value, input JVP/VJP/HVP where mathematically
defined, parameter products when the surrogate is trainable, and an explicit
refusal at discrete or nonsmooth boundaries. A GPU claim requires the model,
posterior samples, acquisition state, optimizer state, and candidate batches to
remain resident for the complete operation graph. OpenACC is preferred when it
preserves semantics; fixed no-autodiff hot loops may use native CUDA generated
or checked by FortSym. Hidden host fallback is prohibited.

## Work packages

### BO0: package and posterior foundations

- [x] Create the MIT-licensed package boundary and model-agnostic posterior
  protocol.
- [ ] Define a versioned `posterior_t` contract for analytic moments, joint
  samples, reparameterized samples, covariance, and predictive log density.
- [ ] Add observation history with input/output/cost/constraint metadata,
  missing-observation policy, duplicate handling, and checkpoint/resume.
- [ ] Add normalized search-space objects for continuous, integer,
  categorical, mixed, constrained, and conditional variables.

### BO1: acquisition catalog

- [ ] Implement expected improvement, probability of improvement, upper
  confidence bound, knowledge gradient, entropy search, predictive entropy,
  and noisy/ log expected improvement.
- [ ] Implement Monte Carlo acquisition evaluation with common random numbers,
  antithetic draws, reparameterized posterior samples, and exact gradients.
- [ ] Implement batch qEI/qNEI, qUCB, qKG, Thompson sampling, and fantasy
  observations with deterministic seeded fixtures.
- [ ] Implement constrained, cost-aware, multi-fidelity, multi-objective,
  preference, and risk-sensitive acquisitions.

### BO2: surrogate integration

- [ ] Adapt FortML exact, derivative-observation, sparse, variational,
  multi-output, multi-task, and deep-kernel GPs to the posterior protocol.
- [ ] Add heteroskedastic, Student-t, classification, count, and robust
  surrogate likelihood adapters.
- [ ] Add fully Bayesian surrogate hyperparameter integration through FortMC
  and compare integrated versus plug-in acquisition policies.
- [ ] Support user-defined FortML posterior providers without requiring a GP.

### BO3: candidate optimization

- [ ] Use FortOpt L-BFGS-B and multistart as the default local acquisition
  optimizer, with explicit bounds, fixed categorical choices, and constraint
  penalties or feasible-region parameterizations.
- [ ] Add Sobol/random/quasi-random initialization, trust-region BO, TuRBO-like
  radius adaptation, cyclic restarts, and deterministic tie handling.
- [ ] Add mixed-integer and categorical optimizers with typed derivative
  refusals for discrete coordinates and continuous products for the rest.
- [ ] Add parallel/asynchronous workers, pending-point fantasizing, retries,
  timeouts, and failure-aware objective/cost handling.

### BO4: experiment and decision policies

- [ ] Implement multi-objective Pareto archives, hypervolume improvement,
  scalarization policies, preference learning, and noisy dominance.
- [ ] Implement active learning, level-set estimation, contour finding,
  feasibility search, and Bayesian calibration/design of experiments.
- [ ] Add stopping rules based on regret, hypervolume, posterior uncertainty,
  budget, and wall time with machine-readable diagnostics.

### BO5: GPU and performance

- [ ] Add resident posterior sampling, acquisition evaluation, reduction, and
  candidate-batch kernels for CPU/OpenACC/CUDA.
- [ ] Keep FortAD-bearing acquisition graphs on FortAD/FortSym until complete
  device JVP/VJP/HVP products exist. Use CUDA for fixed sampling/reduction
  kernels where OpenACC cannot preserve residency or determinism.
- [ ] Benchmark against BoTorch/GPyTorch, JAX, and deterministic NumPy on
  matched functions, models, precision, seeds, restart counts, and stopping
  criteria. Report regret/sample efficiency separately from wall time.

### BO6: evidence and release

- [ ] Add analytic one-dimensional functions, Branin, Hartmann, Ackley,
  constrained synthetic functions, noisy objectives, and multi-objective
  fixtures with known optima or dense reference grids.
- [ ] Record simple regret, cumulative regret, best feasible value, constraint
  violations, acquisition evaluations, gradient evaluations, ESS for sampled
  policies, memory, transfers, and wall time.
- [ ] Keep CPU, transfer-inclusive GPU, resident GPU, and typed refusal rows
  separate, with source/toolchain/device provenance.

## Definition of done

FortBO is production-ready only when every released policy has an independent
optimization/statistical oracle, deterministic replay or an explicit seeded
stochastic contract, documented derivative boundaries, robust failure handling,
and complete CPU/GPU evidence or a typed refusal. A single successful EI run is
not sufficient evidence for Bayesian-optimization parity.
