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

## Derivative observations are universal

Derivative information is not a feature of one policy. If a run can measure
gradients — an adjoint from the simulation, a FortAD-differentiated objective,
a FortSym-generated kernel — then **every** method in FortBO must be able to
use them. DTuRBO is where derivative information is exploited most
aggressively, but it is not where it is *allowed*.

The invariant that makes this true is that derivative observations enter at the
history and leave at the posterior, and nothing in between is policy-specific:

```
objective + gradient  ->  history  ->  surrogate  ->  posterior  ->  any policy
```

Concretely, and binding on every work package below:

- The observation history stores gradients as first-class per-row data with
  per-coordinate presence flags, so a row may carry a full gradient, a partial
  gradient, or none. A partially measured gradient never masquerades as a
  complete one.
- Any surrogate that declares the derivative-observation capability consumes
  those rows. The choice of surrogate is independent of the choice of policy.
- Acquisitions and policies see only `posterior_t`. They cannot tell whether
  the posterior was conditioned on values alone or on values and gradients, and
  they must not try to: EI, qNEI, UCB, KG, entropy search, hypervolume
  improvement, active learning, and plain global BO all become
  derivative-informed for free the moment the surrogate is.
- Separately from *observing* derivatives, a policy may *use* posterior
  derivatives (`moment_gradient`, `moment_hessian`) to optimize its
  acquisition. These are two independent axes, and every combination of the
  four is a supported configuration:

  | | value-only surrogate | derivative-observation surrogate |
  | --- | --- | --- |
  | **sampling candidate search** | plain TuRBO, random/Sobol BO | derivative-informed model, sampled search |
  | **gradient candidate search** | gradient-based acquisition optimization | DTuRBO, full derivative use |

- A method that genuinely cannot accept derivative observations issues a typed
  refusal naming the reason. Silently discarding measured gradients is a defect,
  not a design choice: the run paid for that information.
- Benchmarks report the value-only and derivative-informed variants of the same
  method as separate rows, and charge the true adjoint cost rather than
  counting a gradient as one free evaluation.

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
- [x] Add observation history with input/output/cost/constraint metadata,
  gradient observations, missing-observation policy, duplicate handling, and
  checkpoint/resume. `src/fortbo_history.f90` keeps insertion order as the
  replay order, refuses to invent values for failed evaluations, treats an
  unmeasured constraint as unknown rather than satisfied, and stores gradients
  with per-coordinate presence flags so a partial gradient cannot pass as a
  complete one. `test_history` checks the incumbent against a brute-force
  feasible scan and re-derives incumbent, feasibility, cost, and gradient set
  from the restored checkpoint.
- [x] Add normalized search-space objects for continuous, integer,
  categorical, mixed, constrained, and conditional variables.
  `src/fortbo_space.f90` owns the only translation between user variables and
  the unit hypercube: log-scaled continuous variables are uniform in the
  exponent, integers use an affine map with deterministic rounding,
  categoricals occupy a one-hot block decoded by arg-max with the lowest index
  winning ties, and `differentiable_mask` marks exactly the continuous
  coordinates so a relaxed integer never yields a derivative that means
  nothing. Conditional variables pin to declared defaults when inactive, which
  `test_space` checks by the observable rule that an inactive coordinate cannot
  change the decoded point.

### BO1: acquisition catalog

- [x] Implement the analytic family: expected improvement, log expected
  improvement, probability of improvement, and the confidence bound, each with
  its moment derivatives and chain-rule input gradients.
  `app/gen_acquisition_leaf.f90` in FortSym states Phi and phi once and derives
  EI, PI, and all four first-order products; `src/generated` carries the
  emitted leaf with its FortSym revision. `src/fortbo_acquisition.f90` owns
  only what the symbolic form cannot express: the deterministic limit at zero
  variance, and the Mills-ratio asymptotic branch that keeps log EI finite and
  accurate where EI itself underflows. `test_acquisition` checks the values
  against Simpson quadrature of their defining integrals and every gradient
  against central differences. Emitting a `pure` leaf required an additive
  `pure_procedure` option in the FortSym kernel emitter, fixed upstream rather
  than worked around here.
- [ ] Implement knowledge gradient, entropy search, predictive entropy search,
  and noisy expected improvement. Knowledge gradient and the entropy family
  need the one-dimensional inner optimization and the expectation over
  fantasized observations that BO1's Monte Carlo item provides, so they land
  after it rather than before.
- [x] Implement Monte Carlo acquisition evaluation with common random numbers,
  antithetic draws, reparameterized posterior samples, and exact gradients.
  `src/fortbo_monte_carlo.f90` freezes the standard normal base draws once,
  which is what simultaneously gives common random numbers, an exact pathwise
  derivative, and bitwise replay. Antithetic pairing refuses an odd sample
  count rather than silently leaving a draw unpaired. `test_monte_carlo`
  tolerances come from the estimator's own standard error, the antithetic claim
  is measured as a variance across sixty seeds rather than asserted from one
  run, and the pathwise gradient is differenced against the *same* base so the
  check measures the derivative instead of sampling noise. These are the
  marginal estimators; the joint reparameterization through `reparam_sample`
  belongs to the batch item below, where the utility couples the rows.
- [ ] Implement batch qEI/qNEI, qUCB, qKG, Thompson sampling, and fantasy
  observations with deterministic seeded fixtures.
- [ ] Implement constrained, cost-aware, multi-fidelity, multi-objective,
  preference, and risk-sensitive acquisitions.

### BO2: surrogate integration

- [x] Adapt the FortML exact and derivative-observation GPs to the posterior
  protocol. `src/fortbo_fortml.f90` presents both behind the *same* contract
  with the same capability bits, so nothing above the boundary can tell which
  was fitted. `fortbo_fit_from_history` chooses from the data rather than a
  flag: a history carrying complete gradients yields the derivative-observation
  GP, expanding each row into one value observation plus one per coordinate.
  `test_fortml_adapter` checks the invariant by its consequences — the same
  unchanged EI and confidence-bound objects run against both adapters, and on a
  held-out grid the derivative-informed model has strictly lower squared error
  and lower posterior uncertainty than the same model fitted to the same values
  without their gradients.
- [ ] Adapt the FortML sparse, variational, multi-output, multi-task, and
  deep-kernel GPs to the posterior protocol.
- [ ] Add heteroskedastic, Student-t, classification, count, and robust
  surrogate likelihood adapters.
- [ ] Add fully Bayesian surrogate hyperparameter integration through FortMC
  and compare integrated versus plug-in acquisition policies.
- [ ] Support user-defined FortML posterior providers without requiring a GP.

### BO3: candidate optimization and trust regions

- [x] Use FortOpt L-BFGS-B and multistart as the default local acquisition
  optimizer, with explicit bounds. `src/fortbo_optimize.f90` negates once at
  the boundary — the acquisition is maximized, L-BFGS-B minimizes — so every
  acquisition can state its own sign convention and forget the optimizer. A
  start that stalls is tolerated because acquisition surfaces really are flat
  far from the data; a run where *no* start converges is an error naming that
  cause. `test_optimize` checks the result against a dense grid search over the
  same box, verifies the first-order conditions from the acquisition's own
  gradient, and confirms that adding starts never worsens the answer and that
  reordering them does not change it. A surrogate without moment gradients is
  refused with a pointer to the sampling search rather than differenced.
- [ ] Add fixed categorical choices and constraint penalties or feasible-region
  parameterizations to the candidate optimizer.
- [x] Add Sobol/random/quasi-random initialization, cyclic restarts, and
  deterministic tie handling. Sobol did not exist in FortNum and was added
  there rather than reimplemented here: `fortnum_sobol` builds Joe-Kuo
  direction numbers and generates points by the Antonov-Saleev Gray-code
  recurrence with caller-owned state. Its tests check the defining properties
  rather than a stored table — exact one-dimensional equidistribution across
  all 21 tabulated dimensions, the (0,2)-net property on the leading pair over
  every elementary split, and a measured star-discrepancy advantage over
  pseudorandom points. A wrong direction-number table passes a range check and
  fails those.
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
  **Blocked on a FortSym gap, not deferred by choice**: the rescaling is a
  reduction over the dimension, and the kernel emitter currently lowers scalar
  expression graphs only — it has no way to express a loop or a reduction. The
  invariant itself is already checked independently in `test_trust_region`.
  Fix belongs in FortSym alongside M9/M13, and the interim Fortran computes the
  geometric mean in log space so it survives the few hundred dimensions this
  algorithm exists for.
- [x] Implement the trust-region state machine with deterministic replay:
  identical seed, identical batch, identical resize history.
  `src/fortbo_trust_region.f90` carries the reference constants with their
  provenance, adapts by the counter rule, and records every resize event for
  the evidence trace. `test_trust_region` checks the length history against an
  independent counter model driven by a scripted outcome sequence, pins the
  reference constants so a future "tuning" edit has to argue with a test, and
  verifies the volume invariant at four hundred dimensions where a direct
  product of lengthscales would underflow. It caught a real defect: restarting
  an exhausted region — the canonical TuRBO restart — was not being counted.
- [x] Implement Thompson selection over `m` regions on seeded, splittable
  FortNum streams with common random numbers across regions.
  `src/fortbo_turbo.f90` generates candidates by perturbing the region center
  with probability `min(1, 20/d)` — Sobol supplies the magnitudes, the
  pseudorandom stream supplies the mask — and selects the batch by one arg-min
  per posterior realization across the concatenated regions, which is the
  bandit and the acquisition at once. `test_turbo` measures the sparsity
  claim directly (about twenty coordinates move per candidate at `d = 200`),
  checks selection against a brute-force arg-min, and checks the bandit
  behavior by consequence. The discrete arg-min carries an explicit derivative
  refusal. The quasi-random path now reaches the dimensions the method exists
  for: FortNum computes its Sobol primitive polynomials rather than tabulating
  them, so `d = 200` uses genuine low-discrepancy perturbations. Past the
  degree-13 enumeration limit it still refuses by name instead of silently
  substituting pseudorandom points.
- [x] Record a typed refusal for derivative products of the discrete Thompson
  argmin — it is not differentiable and must not pretend to be.
  `fortbo_thompson_gradient_refusal` explains that perturbing the candidates
  and reselecting measures which candidate happened to win, not a derivative.
- [ ] Reproduce the paper's qualitative ordering on Ackley-200, the 60D rover
  trajectory problem, and the 14D robot pushing problem against a pinned
  `uber-research/TuRBO` and the BoTorch `turbo_1` tutorial. The Sobol blocker
  is cleared — `fortnum_sobol` now covers dimensions into the hundreds with
  computed primitive polynomials, checked by exact equidistribution in all 200
  coordinates — so what remains is the harness and the pinned baselines.

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

`test_end_to_end` is the integration evidence for everything assembled so far:
a full Branin run over the search space, the gradient-carrying history, the
FortML surrogate chosen from the data, FortSym-derived expected improvement,
the TuRBO region and its candidate generator, and the benchmark's verified
optimum. Its oracle is the one that cannot be faked by a component test —
simple regret against a known optimum, beaten against random search from an
identical Sobol initial design on an identical evaluation budget. The
derivative-informed variant runs as a separate row rather than being folded
into the headline number.

- [x] Add Branin, Hartmann-3, Hartmann-6, Ackley, Rosenbrock, Levy, and the
  sphere with known optima and exact gradients. `src/fortbo_benchmarks.f90`
  carries each optimizer and optimal value as literature data, and every
  function supplies an analytic gradient so the derivative-observation path is
  exercised against a true adjoint rather than against differencing error.
  `test_benchmarks` refuses to take the recorded constants on trust: a dense
  grid over Branin's box and a local sweep around every other optimizer must
  fail to beat the recorded value, the gradient must vanish at each interior
  optimum, and Ackley's gradient is checked at the origin, which is
  simultaneously its optimum and the removable singularity where the natural
  expression divides by zero.
- [ ] Add constrained synthetic functions, noisy objectives, and
  multi-objective fixtures with known Pareto fronts or dense reference grids.
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
