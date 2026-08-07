# fortbo roadmap

`fortbo` is the Bayesian-optimization layer for the lazy-fortran stack. It
owns acquisition functions, sequential design, constrained candidate search,
trust-region policies (TuRBO and DTuRBO), batch/multi-objective policies, and
regret benchmarks. FortML owns surrogate
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

## First-principles generation mandate

Every closed-form quantity in FortBO is *derived*, not transcribed. Acquisition
values, their input and parameter derivatives, posterior-derivative kernels,
trust-region geometry rules, hypervolume-improvement expressions, and quadrature
weights are stated once as a symbolic definition and lowered to Fortran (and to
device leaves) by FortSym. The generator stays in the repository, the exact
FortSym revision and regeneration command are recorded next to the emitted
source, and the emitted source is checked in the same test binary against an
oracle that does not read the generated code.

Binding rules:

- No hand-written analytic kernel is accepted where FortSym can derive it.
  Copying a formula out of BoTorch, GPyTorch, a paper, or a previous
  lazy-fortran repository is transcription, not derivation, and does not pass
  review. The symbolic definition is the source of truth; drift is detected by
  regeneration in CI, not by reading.
- Derivative products come from a FortSym-generated kernel, a FortAD graph, or
  an explicitly typed refusal. There is no fourth path, and no silent
  finite-difference substitute.
- **When FortSym cannot yet do what FortBO needs, FortSym is fixed.** Working
  around a FortSym gap inside FortBO is prohibited; the gap is filed against
  the owning FortSym milestone, fixed there, and consumed here. Known gaps that
  FortBO will hit, in the order it will hit them:
  - **M9 (matrices)** — trust-region lengthscale rescaling, posterior
    covariance and its Cholesky-based derivatives, GP-with-derivatives block
    kernels, and the Hessian blocks DTuRBO needs are all matrix-symbolic work
    that FortSym does not yet own.
  - **M13 (codegen completion)** — binding opaque applied functions and their
    `Derivative` nodes to consumer-supplied procedures, which is what lets an
    acquisition be derived against an abstract posterior instead of one
    hard-coded kernel.
  - **M6 (special functions)** — the Gaussian CDF/PDF pair, `erf`/`erfc`, and
    `log`-space forms that EI, logEI, PI, and entropy-search terms need to be
    both derived and numerically safe.
  - **M5 (complex domain)** and **M3 (series)** — asymptotic expansions for the
    numerically hostile branches (logEI tails, small-variance limits) where the
    naive derived form cancels catastrophically.
- A FortSym `UNKNOWN` or a refusal to decide an identity is a legitimate,
  recordable outcome. It becomes a typed refusal in FortBO with the failing
  expression attached, never a guess.
- Numerical evaluation, quadrature, and root finding belong to FortNum, not to
  FortSym. FortSym contributes the expression and the emitted callable.

## Work packages

### BO0: package and posterior foundations

- [x] Create the MIT-licensed package boundary and model-agnostic posterior
  protocol.
- [x] Define a versioned `posterior_t` contract for analytic moments, joint
  samples, reparameterized samples, covariance, and predictive log density.
  `src/fortbo_posterior.f90` carries `FORTBO_POSTERIOR_CONTRACT_VERSION`, a
  capability bitmask, and a typed refusal per undeclared operation that names
  the missing operation. `test_posterior_contract` checks the sampler against a
  Monte Carlo moment oracle and the log density against the closed-form scalar
  normal and the explicit two-by-two inverse, neither of which touches the
  factorization path; that oracle caught a `log(4*pi)` normalization bug.
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

### BO3: candidate optimization and trust regions

- [ ] Use FortOpt L-BFGS-B and multistart as the default local acquisition
  optimizer, with explicit bounds, fixed categorical choices, and constraint
  penalties or feasible-region parameterizations.
- [ ] Add Sobol/random/quasi-random initialization, cyclic restarts, and
  deterministic tie handling.
- [ ] Implement **TuRBO** to the specification below: lengthscale-rescaled
  hyperrectangle trust regions, success/failure counters, halve/double radius
  adaptation, restart on collapse, Thompson-sampling candidate selection, and
  the implicit multi-armed bandit across `m` simultaneous regions (TuRBO-1 and
  TuRBO-`m`).
- [ ] Implement **DTuRBO**, the derivative-enabled trust-region policy, in its
  three composable modes: derivative observations in the surrogate, posterior
  gradient/Hessian local quadratic models solved as bound-constrained
  subproblems, and gradient-based acquisition optimization inside the region.
- [ ] Add mixed-integer and categorical optimizers with typed derivative
  refusals for discrete coordinates and continuous products for the rest.
- [ ] Add parallel/asynchronous workers, pending-point fantasizing, retries,
  timeouts, and failure-aware objective/cost handling.

### BO3T: TuRBO and DTuRBO specification

This is the load-bearing part of BO3 and is specified here so that the
implementation is a derivation, not a port.

#### TuRBO (Eriksson et al., NeurIPS 2019)

Domain is normalized to `[0,1]^d` and observations are standardized. A trust
region is a hyperrectangle centered at the incumbent `x*` — the best observation
in the noise-free case, the observation with the smallest posterior mean under
the local surrogate in the noisy case — with base side length `L` and per-
dimension side lengths

```
L_i = lambda_i * L / (prod_j lambda_j)^(1/d)
```

so the ARD lengthscales `lambda_i` shape the region while the total volume stays
`L^d`. The rescaling identity, its derivative with respect to the lengthscales,
and the volume invariant are FortSym obligations, not literals in a loop.

Adaptation: `tau_succ` consecutive successes give `L <- min(L_max, 2L)`;
`tau_fail` consecutive failures give `L <- L/2`; both counters reset on any
resize. A region with `L < L_min` is discarded and a fresh one is initialized at
`L_init`. Reference defaults from the authors' implementation, to be recorded as
named constants with provenance rather than magic numbers: `L_init = 0.8`,
`L_min = 2^-7`, `L_max = 1.6`, `tau_succ = 3`,
`tau_fail = ceil(max(4, d) / q)`.

Candidates: `n_cand = min(100d, 5000)` Sobol points perturbing the center, each
coordinate perturbed with probability `p = min(1, 20/d)` and at least one
coordinate always perturbed. Selection is Thompson sampling — draw one posterior
realization per region, concatenate across the `m` regions, and take the `q`
minimizers. That single rule is both the within-region acquisition and the
across-region bandit; it must not be split into two heuristics.

- [ ] Derive and generate the lengthscale-rescaling and volume-invariant
  kernels through FortSym, with an independent oracle for the invariant.
- [ ] Implement the trust-region state machine with deterministic replay:
  identical seed, identical batch, identical resize history.
- [ ] Implement Thompson selection over `m` regions on seeded, splittable
  FortNum streams with common random numbers across regions.
- [ ] Record a typed refusal for derivative products of the discrete Thompson
  argmin — it is not differentiable and must not pretend to be.
- [ ] Reproduce the paper's qualitative ordering on Ackley-200, the 60D rover
  trajectory problem, and the 14D robot pushing problem against a pinned
  `uber-research/TuRBO` and the BoTorch `turbo_1` tutorial.

#### DTuRBO (derivative-enabled TuRBO)

DTuRBO is the variant that exploits derivative information the lazy-fortran
stack already has — FortAD adjoints of the objective, and FortML posteriors that
are differentiable in the query point. It is specified as three modes that share
one trust-region state machine, so they can be enabled independently and
benchmarked against each other and against plain TuRBO:

1. **Derivative observations.** The surrogate is a GP with derivative
   observations: the joint kernel block matrix over `[k, d_x k, d_x' k,
   d_x d_x' k]` from the FortML derivative-observation GP. Gradients enter the
   history as first-class observations with their own noise model, and the
   trust-region success test may use predicted decrease rather than observed
   decrease alone. Cost accounting must charge the true adjoint cost, not one
   evaluation.
2. **Posterior-derivative local models.** Inside each region, build the local
   quadratic model from the posterior gradient and Hessian of the *global*
   surrogate, following the Newton-BO construction (Enhancing Trust-Region
   Bayesian Optimization via Newton Methods, arXiv:2508.18423): with `lambda`
   drawn from a truncated normal, use `grad mu + lambda grad sigma` and
   `hess mu + lambda hess sigma`, and select the next point by solving the
   bound-constrained quadratic program over the region intersected with
   `[0,1]^d`. Radius adaptation switches to the ratio test
   `rho = actual decrease / predicted decrease` with expand on `rho >= eta_1`
   and shrink on `rho < eta_0`, which is the classical trust-region rule and
   subsumes the counter rule. Restart when the radius collapses or when
   `||grad mu||` falls below tolerance.
3. **Gradient-based in-region acquisition.** Optimize the acquisition inside
   the region with FortOpt L-BFGS-B on exact FortAD/FortSym products instead of
   sampling `n_cand` candidates, keeping Thompson sampling available as the
   batch and bandit rule.

Mode 2 depends on a lengthscale prior that keeps posterior gradients from
vanishing in high dimension; the dimension-scaled log-normal prior
`log lambda_i ~ Normal(-4 + log(d)/2, 1)` (Hvarfner et al., 2024) is the default
and belongs to FortML's parameter registry, not to FortBO.

- [ ] Define the derivative-observation posterior contract against FortML and
  fail loudly when a surrogate cannot supply `grad mu`, `grad sigma`,
  `hess mu`, `hess sigma`, or their device residency.
- [ ] Derive the posterior gradient and Hessian expressions for each supported
  kernel through FortSym and emit them; block-matrix work blocks on FortSym M9
  and must be fixed there.
- [ ] Verify every generated derivative kernel against a complex-step or
  Richardson-extrapolated finite-difference oracle and against FortAD, with
  the symmetry of the Hessian checked exactly.
- [ ] Solve the bound-constrained quadratic subproblem through FortOpt with a
  documented fallback when the Hessian is indefinite; do not silently project
  to a positive-definite surrogate.
- [ ] Benchmark all three modes plus their combinations against TuRBO on
  matched budgets, counting adjoint cost honestly, and report where derivative
  information does *not* pay for itself.

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
- [ ] Keep the TuRBO/DTuRBO inner loop resident: Sobol candidate generation,
  perturbation masking, per-region Thompson draws, the cross-region argmin, and
  the posterior gradient/Hessian evaluation. A host round trip per region per
  iteration is a failed GPU claim, not a partial one.
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
- [ ] Add the high-dimensional trust-region fixtures with exact gradients:
  Ackley-200 on `[-5,10]^200`, Levy and Rosenbrock at `d = 100..500`, the 60D
  rover trajectory problem, and the 14D robot pushing problem. Gradients come
  from FortAD or a FortSym-generated kernel so DTuRBO modes are exercised on a
  true adjoint, not on finite differences.
- [ ] Record simple regret, cumulative regret, best feasible value, constraint
  violations, acquisition evaluations, gradient evaluations, ESS for sampled
  policies, memory, transfers, and wall time.
- [ ] Record the trust-region trace for every TuRBO/DTuRBO run: per-region
  radius history, success/failure counters, ratio-test values, restart events,
  and which region supplied each accepted batch point. A regret curve without
  the radius history is not evidence that the trust-region logic is correct.
- [ ] Keep CPU, transfer-inclusive GPU, resident GPU, and typed refusal rows
  separate, with source/toolchain/device provenance.

## Definition of done

FortBO is production-ready only when every released policy has an independent
optimization/statistical oracle, deterministic replay or an explicit seeded
stochastic contract, documented derivative boundaries, robust failure handling,
and complete CPU/GPU evidence or a typed refusal. A single successful EI run is
not sufficient evidence for Bayesian-optimization parity. TuRBO and DTuRBO are
release-blocking: FortBO does not reach 1.0 with acquisition functions alone,
because the high-dimensional problems this stack exists to solve are exactly the
ones plain global BO fails on.

## References

- D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek. *Scalable global
  optimization via local Bayesian optimization.* NeurIPS 2019
  ([arXiv:1910.01739](https://arxiv.org/abs/1910.01739)). The TuRBO
  specification above follows this paper; the reference constants follow the
  authors' implementation at `uber-research/TuRBO` and the BoTorch `turbo_1`
  tutorial.
- *Enhancing trust-region Bayesian optimization via Newton methods.*
  [arXiv:2508.18423](https://arxiv.org/abs/2508.18423). Source of the DTuRBO
  mode-2 construction: posterior gradient/Hessian local quadratic models,
  bound-constrained subproblems, and the ratio-test radius rule.
- J. Wu, M. Poloczek, A. G. Wilson, P. Frazier. *Bayesian optimization with
  gradients.* NeurIPS 2017. Source of the derivative-observation acquisition
  (d-KG) used by DTuRBO mode 1.
- C. Hvarfner, E. Hellsten, L. Nardi. *Vanilla Bayesian optimization performs
  great in high dimensions.*
  [arXiv:2402.02229](https://arxiv.org/abs/2402.02229). Source of the
  dimension-scaled lengthscale prior.
- C. E. Rasmussen, C. K. I. Williams. *Gaussian processes for machine
  learning.* MIT Press 2006. Posterior and derivative-observation identities
  that FortSym must derive rather than FortBO transcribe.
