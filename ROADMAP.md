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

## Resolved upstream defects

**FortML `gp_derivative_predict_input_jvp` overwrote the posterior variance
with the prior.** Found by the central-difference oracle in
`test_fortml_adapter`: `mean_dot` and `variance_dot` were both correct, but the
routine's `variance` output had been clobbered with `k(x,x)`, because the prior
covariance was written straight into `variance(j)` instead of a scratch scalar.
Chain-ruling the standard deviation with that value silently scaled every
acquisition gradient.

Fixed upstream (`f56cdcd`), with a regression test in
`test_derivative_gp_products` asserting that a JVP's primal outputs equal what
`predict` returns. The pre-existing tests checked only the tangents, which is
exactly how the defect survived; reverting the fix now fails that test.

**FortML had no second input derivative of the posterior, and its radial leaves
were wrong at coincidence.** DTuRBO mode 2 needs the curvature of *both*
posterior moments. The mean's was reachable indirectly, by predicting derivative
components and differentiating those once more, but the variance's was not: it
is not a predicted component but the quadratic form `k(x,x) - k_*^T K^-1 k_*`,
so its curvature needs genuine second input derivatives of both terms.

Added upstream as `gp_derivative_predict_input_hvp`, assembled from the exact
third-derivative machinery rather than by finite differences, which at a
near-stationary point lose most of their significant digits precisely where the
curvature matters. Building it exposed a real defect: the coincident-point
branch of the radial leaf zeroed the gradient *tangents* along with the gradient
itself. The gradient does vanish at `r = 0`, but its tangent is `phi''(0)` times
the direction, so the reported second-derivative product was exactly zero at
every training point — the one place a trust-region or Newton step is most
likely to ask for one. Matern 5/2 at an observed point returned `-3.79` where
the true curvature is `+5.47`. Fixed upstream (`83d088a`) with
`test_derivative_gp_input_hvp`, whose oracle is Richardson-extrapolated central
differences of `predict`, sharing no code with the analytic path.

Matern 3/2 has no third derivative at coincidence and continues to refuse rather
than return the limit of an expression that has none.

**This clears the blocker on DTuRBO mode 2**, which is now implementable in
FortBO without further upstream work.

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
- [x] Implement **knowledge gradient**. `src/fortbo_knowledge_gradient.f90`
  answers a different question from expected improvement: not how much better
  the next *observation* will be, but how much better the decision finally
  reported will be. The difference shows up whenever the point worth sampling
  is not the point worth reporting, which is why EI degrades under heavy noise
  and KG does not.

  Conditioning on one fantasized observation shifts every reference mean along
  a line in a single standard normal, so the inner expectation is the expected
  minimum of a family of affine functions of one scalar — computed exactly by
  envelope sweep rather than sampled, because sampling noise inside an
  acquisition turns a tie into a coin flip and breaks replay.

  Two real defects surfaced while testing it against Monte Carlo. First, the
  minimum of affine functions is *concave*, so sorting by increasing slope
  traverses its envelope right to left, while the maximum is convex and
  traverses left to right; the first version swept the minimum directly with
  the convex convention and kept lines that are never on the envelope at all.
  It is now computed as `-E[max(-lines)]`, so the sweep is written once. Second,
  the slope-tie ordering has to be *descending* by intercept: on an upper
  envelope only the highest of a set of parallel lines can appear, and
  ascending kept the wrong one. `test_knowledge_gradient` also checks the
  property that separates KG from EI — a candidate uncorrelated with every
  reference point is worth nothing however uncertain it is, where EI would
  chase its variance.
- [x] Implement **noisy expected improvement**. `fortbo_mc_noisy_ei_t` removes
  the observed value from the comparison entirely. Plain EI measures against
  the smallest value ever *observed*, which under noise is the minimum of a
  sample: biased low by roughly the noise scale, and worse the more points are
  evaluated. EI against it shrinks toward zero everywhere as a run proceeds,
  which is the failure usually reported as "EI stops exploring". Noisy EI's
  incumbent is the posterior's belief about the best *latent* value at the
  evaluated inputs, re-drawn inside each sample and sharing the sample index
  with the candidate — an honest comparison needs the draw in which the
  observed points look good to be the same draw in which the candidate is
  judged.

  `test_monte_carlo` checks it against its own definition from the frozen
  draws, checks that noiseless observations reduce it exactly to ordinary EI
  (so it is a generalization, not a different acquisition with the same name),
  and checks the property it exists for: it beats EI measured against a
  noise-depressed sample minimum. One assertion had to be corrected — an
  uncertain incumbent leaves *less* improvement available, not more, because
  the minimum of several random variables sits below the minimum of their
  means. Jensen decides that, not intuition.
- [x] Implement **max-value entropy search**. `src/fortbo_entropy.f90` asks how
  much the next observation will *tell us* about the optimum rather than how
  much better it will be. The two disagree whenever an evaluation is
  informative without being good — a point that will turn out mediocre but
  whose value rules out a region is worth an evaluation, and expected
  improvement cannot say so. `test_entropy` pins exactly that: relevance beats
  raw uncertainty, so a moderately uncertain point near the sampled optimum
  outranks a very uncertain one far from it.

  MES targets the distribution over the optimum's *value*, which is
  one-dimensional whatever the input dimension is, so the acquisition is closed
  form. FortBO minimizes, so `y*` is the minimum and conditioning truncates the
  posterior from *below*; the maximization form in the literature has the
  opposite truncation and the opposite sign on its second term, and
  transcribing it unchanged is the obvious way to get this wrong. The samples
  of `y*` are supplied by the caller for the same reason knowledge gradient's
  reference set is: which sampler produced them is a decision about the
  problem, and hiding it would make two runs incomparable without either saying
  why.

  The oracle is numerical integration of `-p log p` over the truncated density
  by Simpson's rule, applied both to the entropy itself and to the difference
  `H(f) - H(f | f >= y*)` that MES is defined as — the definition of entropy
  rather than a rearrangement of the closed form.
- [ ] Implement predictive entropy search, which targets the optimum's location
  rather than its value and needs the nested approximation MES avoids.
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
- [x] Implement batch **qEI/qNEI and qUCB** with deterministic seeded fixtures.
  `src/fortbo_batch.f90` scores a set of `q` points under one **joint**
  posterior draw, which is the only thing separating a batch acquisition from
  `q` separate ones. Correlated points share their randomness, so two nearby
  candidates rise and fall together and the second adds almost nothing to the
  maximum — the estimator penalizes redundancy with no explicit diversity term
  anywhere. Scoring the same batch under independent marginals would rate `q`
  copies of one point as `q` times as good as one, which is exactly backwards,
  and a posterior offering only marginal moments therefore refuses by
  capability rather than pretending the points are independent.

  `test_batch` anchors the estimators to the already-validated analytic EI via
  a batch of one, then measures the two behaviors that matter: a duplicate adds
  nothing, and a diverse pair beats a clustered one. The duplicate check
  surfaced something worth recording — two copies of a point agree only to
  about `sqrt(jitter)`, not to rounding, because a covariance with a repeated
  point is singular and the jitter that makes it factorizable is exactly what
  lets the copy wander, entering the draw through a Cholesky factor.
- [ ] Implement batch **qKG**, Thompson sampling as a batch rule, and fantasy
  observations. qKG needs the fantasy machinery over `q` simultaneous
  observations, which is a different construction from the three above.
- [x] Implement **constrained and cost-aware** acquisitions.
  `src/fortbo_constrained.f90` weights a base acquisition by the probability of
  feasibility, or divides it by cost to the power `alpha`. Both are weightings,
  and both are wrong in an instructive way if applied naively.

  Weighting by feasibility is only valid because expected improvement is
  non-negative. Applied to an acquisition that can go negative — UCB under a
  minimization convention — multiplying by a small probability moves the value
  *up*, so an almost certainly infeasible point outranks a feasible one and the
  ordering is inverted with nothing in the output saying so. A negative base
  value is therefore refused, and `test_constrained` asserts that refusal
  directly rather than trusting callers to notice.

  `alpha` exists because the raw improvement-per-cost ratio diverges as cost
  goes to zero, so a cheap and useless point beats an expensive and excellent
  one. It interpolates between ignoring cost and full per-unit accounting, and
  the test pins both endpoints. A non-positive cost is refused rather than
  clamped: it almost always means a cost model fitted in log space and never
  exponentiated, and clamping would hide that.

  Multi-objective acquisitions are already covered by `src/fortbo_pareto.f90`
  and preference acquisitions by `src/fortbo_preference.f90`.
- [x] Implement **multi-fidelity and risk-sensitive** acquisitions.
  `src/fortbo_risk.f90` offers mean-variance, value at risk, and conditional
  value at risk, all under the minimization convention. CVaR is the one that
  sees the tail it is protecting against and is coherent where VaR is not — VaR
  can *rise* when two risks are combined, which is not a property anyone wants
  in an objective. The ordering `CVaR >= VaR >= mean` is asserted rather than
  assumed, since it is the cheapest way to catch a swapped formula. A negative
  risk aversion is refused: risk *seeking* is coherent and almost never what
  was meant.

  The multi-fidelity weight is `base * rho^2 / cost^alpha`. `rho` is squared
  because information about a Gaussian scales with variance explained, not with
  correlation, so a fidelity with half the correlation must be four times
  cheaper to break even rather than twice — which is exactly what the test
  pins, and where my own arithmetic was inverted the first time.

  `test_risk` checks VaR and CVaR against Monte Carlo by *counting* rather than
  sorting. The obvious oracle — sort the draws and read off the quantile — was
  written first and timed out, since an insertion sort over 400000 draws is
  quadratic. Counting is both faster and a sharper check: VaR is defined by the
  fraction of outcomes below it, so measuring that fraction tests the
  definition instead of an artifact of how the sample was ordered.

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
- [x] Implement **TuRBO** to the specification below: lengthscale-rescaled
  hyperrectangle trust regions, success/failure counters, halve/double radius
  adaptation, restart on collapse, Thompson-sampling candidate selection, and
  the implicit multi-armed bandit across `m` simultaneous regions (TuRBO-1 and
  TuRBO-`m`). The regions, candidates, and Thompson selection each stand alone
  and are tested on their own; `src/fortbo_turbo_driver.f90` is what makes them
  a *method*. It is ask/tell rather than a callback loop, because FortBO does
  not own the objective and the asynchronous-worker item needs that shape
  anyway. Each region carries its own history and its own local surrogate;
  pooling the data would defeat the point of a local model. A collapsed region
  is left *unplaced* and draws a fresh design before being re-centered on the
  best point of it, since re-centering on an arbitrary point discards a design
  the run is about to pay for.

  **The bandit is implicit, and testing that took two attempts.** The first
  test let both regions explore freely and compared whichever happened to hold
  better values; that measured almost nothing, because both converge on the
  same basin and the result sat at chance whether the bandit worked or not.
  Constructing the asymmetry — one region seeded on a design at the optimum,
  the other in the far corner — made the measurement mean something, and
  immediately exposed a real defect: the driver sent 31 of 50 proposals to the
  *worse* region. A GP with a zero mean function reverts to zero away from its
  data, so a region whose observations are all bad extrapolates to a prior
  claiming zero is typical — better than anything it has actually seen — and
  its realizations undercut a region sitting on genuinely good values.
  Standardizing each region's observations before fitting and mapping the
  moments back fixes it: the prior then says "typical for here", which is what
  a trust region means. This is why TuRBO standardizes per region, and it is
  not cosmetic.
- [x] Implement **DTuRBO**, the derivative-enabled trust-region policy, in its
  three composable modes: derivative observations in the surrogate, posterior
  gradient/Hessian local quadratic models solved as bound-constrained
  subproblems, and gradient-based acquisition optimization inside the region.
  Modes 1 and 3 need no code of their own — mode 1 is the surrogate
  `fortbo_fit_from_history` already selects from the data, and mode 3 is what
  `fortbo_optimize_acquisition` already does inside the region bounds. Mode 2
  is `src/fortbo_dturbo.f90`, and it was blocked until FortML gained
  `predict_input_hvp`, because `hess sigma` existed nowhere in the stack.

  `lambda` is drawn from a standard normal truncated to the non-negative half
  line, by inverting that distribution's own CDF rather than by taking `abs` of
  a normal draw: the two give the same distribution but consume the stream
  differently, and replay against a recorded run would break. The truncation is
  not decorative — a negative `lambda` would subtract the standard deviation's
  gradient and steer the step *away* from uncertainty, which is the opposite of
  an acquisition. `test_dturbo` checks the sampled distribution against
  `2*Phi(x) - 1` with the tolerance taken from sampling error, and checks that
  `lambda = 0` recovers exactly a Newton step on the posterior mean.

  Radius adaptation uses the classical ratio test rather than the success and
  failure counters. The distinction is behavioral and is tested as such: a step
  that improves the objective while the model predicted ten times that
  improvement must *shrink* the region, because the model is badly wrong. A
  counter rule scores that as a success and would eventually expand. An
  indefinite Hessian is still not repaired; mode 2 relies on
  `fortbo_quadratic` following negative curvature to the boundary rather than
  working around it.
- [x] Add mixed-integer and categorical optimizers with typed derivative
  refusals for discrete coordinates and continuous products for the rest.
  `src/fortbo_mixed.f90` searches the discrete coordinates by neighbourhood —
  an integer to its two neighbours, a categorical to each of its other levels —
  and accepts a move only if it improves. Enumerating the discrete product is
  exponential and sampling it wastes most draws on combinations already
  rejected.

  A mixed space has no gradient. Not a gradient that is hard to compute: the
  acquisition is a step function of an integer coordinate and a finite table of
  a categorical one, so the derivative is undefined *everywhere* rather than
  awkward somewhere. The typed refusal exists so a caller gets a named error
  instead of a finite-difference estimate of a step function — zero almost
  everywhere and enormous exactly at the jumps, which carries no information
  and still looks like a derivative. The test checks the refusal fires on a
  mixed space and does *not* fire on a continuous one, since refusing there
  would break the ordinary gradient path.

  Neighbourhood moves are computed in the *decoded* space and re-encoded: a
  fixed step on the unit cube lands on a different integer for different ranges
  and on none at all for a range of one. `test_mixed` checks the neighbourhood
  against exhaustive enumeration, checks the walk reaches the exhaustive
  minimum from every start, and checks that a truncated walk reports
  `converged = .false.` rather than passing off a budget exhaustion as a local
  optimum.
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

- [x] Derive and generate the lengthscale-rescaling and volume-invariant
  kernels through FortSym, with an independent oracle for the invariant.
  FortSym's `app/gen_trust_region_leaf.f90` states the rescaling once,
  `length_j = exp(log ell_j - mean_k log ell_k) * L`, and derives it together
  with its derivatives in the log lengthscale, the log mean, and the base
  length. `region_side_lengths` now calls the generated leaf per dimension.

  **The remaining FortSym gap is narrower than it looked.** What was blocked is
  the *reduction*, and a mean is not a formula — it carries no mathematical
  content beyond "arithmetic mean". Everything that is an expression is scalar
  in `j` and derivable today; only the summation stays in Fortran. FortSym
  still cannot express a reduction, and that is recorded here rather than
  worked around silently, but nothing with mathematical content is transcribed
  by hand any longer.

  The volume invariant `product_j length_j = L^d` follows because deviations
  from a mean sum to zero, and is checked independently in `test_trust_region`,
  which also checks the generated derivatives against central differences and
  pins the statement that the normalization really is the geometric mean: a
  lengthscale sitting at the mean receives exactly the base length.

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
   batch and bandit rule. **Working end to end** — `test_end_to_end` runs this
   configuration on Branin against a real derivative-observation GP, with Sobol
   multistart inside the region bounds, and beats random search at equal
   budget. Reaching it required fixing the FortML variance-JVP defect recorded
   above; before that the surrogate could not supply a sound `moment_gradient`
   and this path was unreachable.

Mode 2 depends on a lengthscale prior that keeps posterior gradients from
vanishing in high dimension; the dimension-scaled log-normal prior
`log lambda_i ~ Normal(-4 + log(d)/2, 1)` (Hvarfner et al., 2024) is the default
and belongs to FortML's parameter registry, not to FortBO.

- [x] Define the derivative-observation posterior contract against FortML and
  fail loudly when a surrogate cannot supply `grad mu`, `grad sigma`,
  `hess mu`, `hess sigma`, or their device residency. The
  derivative-observation adapter declares and supplies `moment_gradient`,
  assembled from one query-input JVP per coordinate and checked against central
  differences of its own moments; the standard deviation's cusp at a training
  site reports the least-magnitude subgradient rather than an infinity. The
  value-only GP does **not** declare the capability and refuses by name,
  because FortML's `gp_predict_jvp` differentiates with respect to the
  parameters rather than the query — a recorded gap, not a choice.
  The mean's Hessian is also supplied, exactly and without waiting for
  FortSym's matrix milestone: a derivative-observation GP can *predict a
  derivative component*, so differentiating the prediction of `df/dx_j` with
  respect to the query gives `H(j,k)` from the same JVP the gradient uses. It
  is checked against central differences of the model's own reported gradient
  and symmetrized exactly. `FORTBO_CAP_MEAN_HESSIAN` is a separate bit from
  `FORTBO_CAP_MOMENT_HESSIAN` because a model can honestly have the mean's
  curvature and not the standard deviation's, and that is exactly the situation
  here.

  **The standard deviation's curvature is now supplied too**, once FortML
  gained `predict_input_hvp` (see *Resolved upstream defects*). It is not a
  predicted quantity but the square root's chain rule applied to the variance's
  curvature, `d2 sd = v_jk/(2 sd) - v_j v_k/(4 sd^3)`, so it needs the
  variance's gradient as well as its Hessian: the square root contributes a term
  that no accuracy in `v_jk` can supply. The pair costs `d` Hessian-vector
  products rather than the `d^2` JVPs the mean-only route needs, and
  `test_fortml_adapter` cross-checks the mean half against `mean_hessian`, which
  reaches the same matrix by a completely different route. At a training site of
  a noiseless model the cusp makes the curvature genuinely infinite, and it is
  refused rather than reported as a number of order 1e18 that a Newton step
  would follow silently. Reaching that state takes a deliberate construction:
  the jitter `fortbo_fit_from_history` adds holds the posterior standard
  deviation near 1e-5, so ordinary use never lands on the cusp — worth knowing,
  and worth testing anyway.
- [ ] Derive the posterior gradient and Hessian expressions for each supported
  kernel through FortSym and emit them; block-matrix work blocks on FortSym M9
  and must be fixed there.
- [ ] Verify every generated derivative kernel against a complex-step or
  Richardson-extrapolated finite-difference oracle and against FortAD, with
  the symmetry of the Hessian checked exactly.
- [x] Solve the bound-constrained quadratic subproblem through FortOpt with a
  documented fallback when the Hessian is indefinite; do not silently project
  to a positive-definite surrogate. `src/fortbo_quadratic.f90` minimizes
  `g'.s + 0.5 s'.H.s` over the box and leaves an indefinite `H` alone: negative
  curvature is information, and the right response is to follow it to the
  boundary, which the bound-constrained solve does. Indefiniteness is detected
  by a Cholesky attempt and reported through a flag; it never changes what is
  solved. A nonsymmetric Hessian is refused rather than averaged, because it
  means two derivations disagree. `test_quadratic` uses `-H^{-1} g` computed
  independently as the interior oracle, a dense grid when the box binds, and
  for the saddle case checks that the step reaches the exact boundary minimum —
  the value a positive-definite repair would fall short of.
- [ ] Benchmark all three modes plus their combinations against TuRBO on
  matched budgets, counting adjoint cost honestly, and report where derivative
  information does *not* pay for itself.

### BO4: experiment and decision policies

- [x] Implement multi-objective Pareto archives, hypervolume improvement, and
  scalarization policies. `src/fortbo_pareto.f90` keeps exactly the
  non-dominated set and computes hypervolume *exactly* by recursive dimension
  sweep rather than by sampling: an approximate indicator inside an acquisition
  turns a tie into a coin flip and breaks replay. `test_pareto` validates the
  sweep against Monte Carlo integration in three and four dimensions, with the
  tolerance taken from the binomial standard error, and against a hand-computed
  staircase area in two. Monotonicity is checked as the property that makes
  hypervolume a sound objective at all — a non-dominated addition strictly
  increases it, a dominated one leaves it exactly unchanged. The augmented
  Chebyshev scalarization's augmentation term is checked by the case it exists
  for: without it a weakly dominated point scores identically to the point that
  dominates it.
- [x] Implement preference learning and noisy dominance.
  `src/fortbo_preference.f90` covers both, because both reduce to one Gaussian
  tail probability. Preference learning fits a latent objective to pairwise
  judgements under the Thurstone-Mosteller model; noisy dominance asks the same
  question of a multi-objective posterior, combining the two standard
  deviations in quadrature. Deciding dominance by comparing posterior means
  instead discards exactly the information that says whether the comparison is
  trustworthy, which is how a front built from noisy observations fills up with
  points that were never good. Nothing is transcribed: FortSym's
  `app/gen_preference_leaf.f90` states the model once and derives the
  probability, its logarithm, and every first derivative. `test_preference`
  validates the probability against Monte Carlo simulation of the model's own
  generative story, the log-likelihood gradient against central differences,
  and the deep-tail asymptotic branch against the Gaussian tail written
  independently — that last check is what caught a missing `log(2)` in the
  branch, since a relative tolerance on a value near -3200 would have absorbed
  it. `fortbo_probability_non_dominated` assumes front members are independent,
  which they are not under a shared surrogate; the test asserts the direction
  the approximation errs in (it understates) rather than an equality that does
  not hold, and checks the one-member case where it is exact.
- [x] Implement active learning, level-set estimation, contour finding,
  feasibility search, and Bayesian calibration/design of experiments.
  `src/fortbo_active.f90` covers the goals that are *not* optimization. Using
  an optimization acquisition for any of them concentrates evaluations near the
  optimum, which is the wrong place when the answer lives elsewhere.

  The straddle rule for level sets is `|mu - t| - kappa sigma`, and the two
  terms genuinely disagree: one wants uncertainty, the other proximity to the
  threshold. `test_active` constructs the case where they conflict and shows
  that pure variance sampling picks a different point entirely — the failure
  the rule exists to avoid. Both degenerate limits are reachable and pinned:
  `kappa = 0` collapses to sampling nearest the contour and stops learning once
  one crossing is found, and large `kappa` forgets the contour altogether.

  Feasibility search is level-set estimation at `t = 0` on a constraint and
  gets no separate routine, because adding one would suggest it were a
  different thing. Maximin design is scored rather than sampled, since spread
  is a property of the design alone; it supplies the model-free starting design
  and the baseline any adaptive method has to beat.

  Every score follows the lower-is-better convention, and the test checks that
  rather than assuming it: a sign error would silently make every optimizer
  seek the *most* certain point and would still run cleanly.
- [x] Add stopping rules based on target value, stall, posterior uncertainty,
  acquisition magnitude, budget, cost, and wall time, with machine-readable
  diagnostics. `src/fortbo_stopping.f90` answers with a *reason*, not a
  boolean: stopping because the budget ran out and stopping because the
  posterior collapsed say opposite things about whether to trust the answer, so
  resource limits are tested first and the reason is recorded. A default rule
  never stops anything, so a caller who forgets to configure stopping does not
  silently get a truncated run, and a zero tolerance counts as enabled rather
  than as disabling. `test_stopping` drives every rule to both sides of its
  boundary and checks the priority order by constructing states where several
  rules would fire at once.

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
- [x] Record simple regret, cumulative regret, best feasible value, constraint
  violations, acquisition evaluations, gradient evaluations, ESS for sampled
  policies, memory, transfers, and wall time. `src/fortbo_metrics.f90` keeps one
  row per evaluation in run order. Three decisions separate an honest record
  from a flattering one, and each is tested rather than documented alone.

  An infeasible point never sets the incumbent, however good its objective
  value — letting it makes a constrained run report solving a problem it never
  solved. Before anything feasible exists the regret is reported *unavailable*
  rather than as a large number, because "we have not found a feasible point"
  and "we found a bad one" call for different responses, and a run with no known
  optimum gets best-value curves instead of regret against a guess.

  Cumulative regret charges every evaluation, not just improvements. The test
  runs two methods that reach the same answer at the same evaluation, one of
  which then keeps searching badly: simple regret cannot tell them apart and
  cumulative regret must.

  Acquisition and gradient evaluations are counted separately from objective
  evaluations, since one objective evaluation can hide thousands of acquisition
  evaluations and a regret-per-objective-evaluation plot hides that trade
  entirely. Wall time, memory, and transfers are recorded but belong to the
  machine, not the method.
- [x] Record the trust-region trace for every TuRBO/DTuRBO run: per-region
  radius history, success/failure counters, ratio-test values, restart events,
  and which region supplied each accepted batch point. A regret curve without
  the radius history is not evidence that the trust-region logic is correct.
  `src/fortbo_trace.f90` appends one row per batch in run order and never
  rewrites them, so a trace read back is the run as it happened. The ratio is
  computed inside `record` from the decreases it also stores, so the two can
  never drift apart, and a row carries `has_ratio` rather than a zero when no
  ratio test ran — TuRBO's counter rule has no ratio, and storing zero would be
  indistinguishable from a step that predicted well and delivered nothing,
  which is the opposite verdict about the model. `test_trace` drives a real
  region up and then down through `fortbo_dturbo_ratio_update` and recovers the
  whole story from the trace alone; writing it exposed that the trace had no
  row for the *starting* radius, so the history began at the already-expanded
  value and could not show the region growing at all.
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
