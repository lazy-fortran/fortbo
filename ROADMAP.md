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
- [ ] Implement predictive entropy search. **C3 built against the paper; C1 and
  C2 outstanding.** Written from arXiv:1406.2541 read rather than recalled — see
  `fortbo-bench/scripts/fetch_provenance.py`, which fetches the sources FortBO
  is built against into a gitignored `provenance/`.

  The paper replaces intractable conditioning on the optimum's location with
  three simplified constraints: **C1** that `x*` is a local optimum
  (`grad f(x*) = 0`, definite Hessian diagonal), **C2** that `f(x*)` beats every
  past observation, and **C3** that `f(x)` is worse than `f(x*)`.
  `src/fortbo_pes.f90` implements **C3** using the paper's own moment-matched
  variance rather than the quadrature written from memory first, including its
  `s`-floor safeguard for queries near `x*` — which is now *reported* through
  `correlation_scale` so a caller can tell an honest variance from a rescued
  one.

  **C3 alone is not location-aware, and the paper's formula confirms it.** With
  matched variances the entropy reduction grows as the query becomes *less*
  correlated with the sampled minimizer, which is the opposite of what a
  location-aware acquisition needs. The location information lives in C1 and C2,
  which need expectation propagation over a latent vector holding `f(x*)` and
  the Hessian diagonal. Reading the paper turned a vague suspicion into a
  precise statement of what is missing.

  The earlier quadrature is kept: comparing it against the matched form
  *measures* the paper's approximation instead of leaving its size unstated, and
  the test confirms the matched entropy is the larger of the two, as maximum
  entropy at fixed variance requires.
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
- [x] Implement batch **qKG** and fantasy observations. Written against Wu,
  Poloczek, Wilson and Frazier (arXiv:1703.04389), whose equation (3.4) states
  d-KG as `min_A mu_n - E_n[min_A mu_{n+q}]` with the expectation marginalizing
  over all `q` observations at once.

  The sequential case collapsed to a one-dimensional envelope because a single
  fantasy shifts every reference mean along a line in one scalar. With `q`
  fantasies the shift is affine in a `q`-vector and no such closed form exists,
  so `fortbo_batch_knowledge_gradient` is Monte Carlo over the fantasy vector.
  That is a statement about the problem rather than a shortcut: the exact
  quantity is a `q`-dimensional integral of a piecewise-linear function.

  `test_knowledge_gradient` anchors it where an anchor exists — with one fantasy
  slot d-KG *is* the sequential knowledge gradient, so the Monte Carlo estimate
  must land on the closed-form envelope value, which it does within sampling
  error. Monotonicity in the batch is checked too, since an extra observation
  can always be ignored and so cannot lower the value.

  Draws come from a caller-owned generator so two candidate batches are compared
  against the same realizations. Comparing them under independent draws would
  rank by sampling error exactly when their true values are close, which is when
  the ranking matters.
- [x] Add Thompson sampling as an explicit batch rule.
  `src/fortbo_thompson.f90` supplies the half that was missing — drawing the
  realizations from an arbitrary posterior — so the rule works with any policy
  and any provider, including the non-GP one. The selection itself was already
  policy-independent; it merely lived in `fortbo_turbo` because that is where it
  was first needed.

  The realizations must be *joint*. Independent marginals would let two
  near-identical candidates take two batch slots, because nothing would tell the
  estimator they are the same question asked twice. A posterior offering only
  marginal moments is therefore refused rather than accommodated.

  Selection is without replacement, which is a deliberate departure from the
  textbook rule: that rule describes one draw at a time and is silent on the
  case, while a batch spending `q` slots on one point would be a correct arg-min
  per slot and a useless batch. `test_thompson` measures that the rule prefers
  candidates the posterior rates better *and* that it still explores rather than
  collapsing onto the best mean — a rule that always picked the arg-min of the
  mean would pass the first check and fail the second.
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
- [x] Adapt the FortML sparse, variational, multi-output, and Student-t
  surrogates to the posterior protocol. `src/fortbo_fortml_sparse.f90` holds the
  adapters; the *models* live in FortML, since none of them is
  Bayesian-optimization specific.

  **Generic model work went upstream.** The Student-t process is new FortML code
  (`fortml_student_t_process.f90`), written against arXiv:1402.4306 read rather
  than recalled. A TP keeps a GP's analytic marginals and changes the thing that
  matters here: its predictive covariance depends on the *observed values*,
  through a Mahalanobis distance whose prior expectation is exactly `n`. Data
  more surprising than expected widen the predictive variance; tamer data narrow
  it. A GP cannot express either, and the test contrasts the two directly. The
  paper's parameterization is deliberately not the usual one — `cov = K`
  exactly, where most references use a scale matrix — and the large-`nu` limit
  against an exact GP is what pins it.

  **A real gap was found upstream too.** FortML's multi-output GP had
  `joint_covariance`, which is the *prior* `B (x) K`, and no posterior route at
  all. An adapter using the prior where a posterior belongs would report
  uncertainty that never shrinks with data: plausible on a plot and simply
  wrong. `predict_covariance` was added to FortML rather than reconstructed in
  FortBO.

  Each adapter's honesty boundary is stated and tested. The sparse GP's
  marginals are the *variational* posterior's, which systematically
  underestimates variance, so an acquisition against it explores less — the
  adapter cannot correct that and does not pretend to. The Student-t adapter
  reports the right first two moments and the wrong tail, since every FortBO
  acquisition integrates against a Gaussian; it declares moments only. The
  multi-output adapter projects onto one output and declares joint covariance,
  because restricting the real joint to one output's rows is exact — the test
  requires the covariance diagonal to equal the reported marginals, which is
  what catches a wrong stride where symmetry alone would not.
- [ ] Adapt the FortML multi-task and deep-kernel GPs, which have no posterior
  contract route yet.
- [x] Add heteroskedastic, Student-t, and classification surrogate likelihood
  adapters. As with the sparse family, the *models* live in FortML —
  `fortml_heteroskedastic_gp.f90` is new generic code, written there rather than
  in FortBO because nothing about it is Bayesian-optimization specific.

  **Heteroskedastic.** One noise variance for the whole domain is wrong whenever
  measurement quality varies with the input, and fitting a single noise does not
  average the regimes — it lands between them and is then overconfident where
  the data are noisy and underconfident where they are clean, which is worse
  than either. Noise is a second latent process on the *log* scale: a process on
  the variance directly would put mass on negatives, and clipping those would
  bias exactly the quiet regions the model exists to represent. Centring the log
  values makes extrapolation revert to the mean noise level rather than to unit
  variance, which nobody claimed. The adapter separates the two uncertainties a
  policy needs to tell apart: a point unmeasured deserves a first evaluation, a
  point measured badly deserves a repeat, and a plain GP cannot say which it is
  looking at.

  **Classification** presents the *latent* moments, deliberately, not the class
  probability. The latent is Gaussian and unbounded, which is what every FortBO
  acquisition integrates against; a probability is neither, and expected
  improvement computed on one would measure improvement in probability rather
  than in the objective. That makes this adapter right for exactly one job —
  finding the decision boundary, where the latent crosses zero — so its intended
  consumer is level-set estimation in `fortbo_active`, not `fortbo_ei`.
- [x] Add count and robust surrogate likelihood adapters. The observation
  models are new FortML code (`fortml_robust_gp.f90`), since a Poisson or
  Student-t likelihood is generic GP work.

  Both are cases an ordinary GP handles badly for one shared reason: it assumes
  a Gaussian residual whose variance does not depend on the latent. A Gaussian
  likelihood on counts puts mass on negative counts and treats a spread of 3
  around a mean of 4 the same as around a mean of 400. A Gaussian log density is
  quadratic, so a single outlier pulls with unbounded force, where a Student-t's
  is logarithmic and its influence *saturates*.

  Two numerical points that had to be got right rather than guessed. The
  Student-t posterior is **not log-concave** — beyond `sqrt(nu) * scale` its
  curvature turns negative and a Newton step would ascend — so the curvature is
  floored at zero, which is precisely the influence saturation the likelihood
  was chosen for. And the Newton system uses `B = I + W^(1/2) K W^(1/2)`, which
  is symmetric positive definite, rather than the direct `I + W K`, which is
  neither: the direct form would need normal equations and square the condition
  number, on a fit whose curvature has just been floored.

  The Poisson response is `exp(f + v/2)`, the log-normal *mean*, not `exp(f)`,
  which is the median — reporting the median understates every rate, and worst
  where the model is least sure. The adapter nonetheless presents *latent*
  moments, because an acquisition on the response would weight a rise from 400
  to 410 the same as one from 4 to 14.

  `converged` is exposed and gates the capability bits: an unsettled Laplace fit
  has moments, but they approximate around a point that is not a mode, and
  declaring them would let a policy consume them silently.
- [x] Add fully Bayesian surrogate hyperparameter integration through FortMC
  and compare integrated versus plug-in acquisition policies. Written against
  Snoek, Larochelle and Adams (arXiv:1206.2944), which states it plainly: the
  expectation over hyperparameters "is the correct generalization to account
  for uncertainty in hyperparameters", and the samples "can be acquired
  efficiently using slice sampling".

  **FortMC had no sampler**, only an abstract log-density type, so
  `fortmc_slice.f90` was added upstream. Slice sampling needs only the log
  density — no gradient, no proposal scale, no acceptance rate — which is why
  it suits hyperparameter marginalization, where the density's scale differs
  enormously between a lengthscale and a noise level. Heights are formed as
  `log p - Exponential(1)` so the density is never exponentiated; a GP log
  likelihood of order hundreds would overflow at once. The shrink step pulls in
  *the end the rejected point fell on*, which is what preserves detail balance —
  the alternatives give a chain that looks fine and has the wrong stationary
  distribution, invisible in any single run. The correlated-normal test is the
  one that bites: a coordinate sweep gets the marginals right almost however it
  is written, and only a correct sweep gets the correlation right.

  **The average is of acquisitions, not of moments.** Expected improvement is
  nonlinear in the moments, so by Jensen the two disagree, and the
  moment-averaged version systematically understates exactly the point the
  hyperparameter samples disagree about — which is the point worth evaluating.
  `test_integrated` measures that gap on a constructed pair and confirms the two
  coincide when the samples agree, so the gap is attributable to the
  disagreement rather than to the arithmetic.

  The comparison the item asks for runs end to end and unmocked: a real GP log
  marginal likelihood is slice-sampled, the surrogate refits at each draw, the
  acquisitions blend, and the result is checked to differ from the plug-in
  policy at the mean hyperparameter. A spread diagnostic reports when a chain
  did not move, so a run can say that integrating bought nothing rather than
  quietly paying for it.
- [x] Support user-defined FortML posterior providers without requiring a GP.
  `src/fortbo_linear_posterior.f90` is a Bayesian linear model on a caller-
  supplied feature map, and it is in the tree specifically so the contract's
  claim stays tested rather than merely stated: `test_linear_posterior` runs
  expected improvement, UCB, qEI, and knowledge gradient against it *unchanged*.

  It is the right second provider because it is not a GP in the way that
  matters. Its posterior has a finite-dimensional parameter, its cost is
  independent of the observation count once fitted, and its function-space
  covariance has rank at most the feature count however many query points are
  asked about. An acquisition that quietly assumed full-rank joint covariance
  would pass against a GP and fail here, so the test constructs exactly that
  case — six query points, three features — and confirms the covariance is
  singular while batch sampling still works, because joint samples are drawn in
  *weight* space rather than by factorizing the singular function-space matrix.

  It declines `FORTBO_CAP_MOMENT_GRADIENT`: that needs the feature map's
  Jacobian, and a caller who supplied only a value map must not have one
  invented on its behalf.

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
- [x] Add fixed categorical choices and constraint penalties or feasible-region
  parameterizations to the candidate optimizer. `src/fortbo_feasible.f90` offers
  both, and the difference between them is not a matter of taste.

  A fixed choice is a *reparameterization*: the coordinate leaves the search, so
  no candidate can violate it and no weight has to be chosen. Whenever a
  constraint can be expressed that way it should be, since searching a smaller
  space beats searching a larger one with a penalty pushing back. `test_feasible`
  decodes *every* candidate rather than one, which is what "cannot violate it"
  means operationally, and checks the unpinned coordinates are left undisturbed.

  Penalties handle what reparameterization cannot. The quadratic penalty is
  smooth but never binding — its optimum sits at `1 + 1/(2 rho)`, outside the
  feasible region for every finite weight — while the exact penalty is binding
  above a finite threshold but has a kink on the boundary, exactly where an
  optimizer spends its time. `fortbo_penalty_is_differentiable` reports which,
  so a gradient-based caller is not handed a kink without being told.

  The test that separates them needed correcting. It first asserted the
  quadratic optimum stays outside for weights up to `1e4`, which failed —
  because at that weight the offset is `5e-5` against a grid spacing of `1e-3`,
  so it was measuring the grid rather than the penalty. It now checks the
  minimizer against `1 + 1/(2 rho)` where that is resolvable, and records the
  practical consequence: the claim is true in exact arithmetic and invisible
  below grid resolution, so what a caller needs is a weight large enough to push
  the residual infeasibility under its own tolerance.
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
  **Constants checked against the paper.** arXiv:1910.01739 supplementary D
  gives `tau_succ = 3`, `L_min = 2^-7`, `L_max = 1.6`, `L_init = 0.8`, which
  FortBO matches. It gives `tau_fail = ceil(d/q)`, where FortBO uses
  `ceil(max(4, d)/q)` — the authors' reference implementation applies the floor
  of four and the paper's text does not mention it. FortBO follows the code and
  records the divergence here rather than leaving a reader to wonder which was
  intended.

- [x] Implement **DTuRBO**, the derivative-enabled trust-region policy, in its
  three composable modes: derivative observations in the surrogate, posterior
  gradient/Hessian local quadratic models solved as bound-constrained
  subproblems, and gradient-based acquisition optimization inside the region.
  Modes 1 and 3 need no code of their own — mode 1 is the surrogate
  `fortbo_fit_from_history` already selects from the data, and mode 3 is what
  `fortbo_optimize_acquisition` already does inside the region bounds. Mode 2
  is `src/fortbo_dturbo.f90`, and it was blocked until FortML gained
  `predict_input_hvp`, because `hess sigma` existed nowhere in the stack.

  **Two errors corrected by reading arXiv:2508.18423 rather than recalling it.**

  `lambda` is truncated to `(-1, 1)`, not to the non-negative half line. The
  first version truncated one-sidedly, reasoning that a negative `lambda` would
  subtract the standard deviation's gradient and steer away from uncertainty.
  That reasoning is plausible and is not what the paper does: section 4 restricts
  `lambda` to `(-1, 1)` explicitly "to ensure local convergence". The bound
  limits how far the local model may deviate from the mean's own Newton model
  *in either direction*; it is not a clipped exploration weight. Negative draws
  occur and the test now requires them. Sampling is still by CDF inversion
  rather than rejection, so one draw is consumed per sample and replay against a
  recorded stream holds.

  Algorithm 2 also caps expansion at `mu ||g||` as well as at the region
  maximum, which the first version omitted. Near a stationary point the gradient
  vanishes and the cap pulls the region in, which is the smooth form of the
  restart-on-small-gradient rule rather than a separate test. It is optional:
  a caller without a gradient keeps the plain doubling rule instead of being
  handed a cap it cannot honestly compute.

  `test_dturbo` checks the sampled distribution against the truncated normal's
  own CDF on `(-1, 1)` with the tolerance from sampling error, and checks that
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
- [x] Add parallel/asynchronous workers, pending-point fantasizing, retries,
  timeouts, and failure-aware objective/cost handling.
  `src/fortbo_workers.f90` exists because having evaluations in flight changes
  what the acquisition must be told, not merely when it is called.

  **Pending points must be fantasized.** Until a dispatched point returns, the
  posterior still shows full uncertainty there, so an acquisition not told
  about it dispatches the same point again and a pool of `q` workers
  degenerates to `q` copies of one evaluation. The fantasy policy is explicit
  and nameable, because the "constant liar" variants that substitute the
  incumbent or the worst observed value bias the surrogate differently and a
  run that used one is not comparable with a run that used another.

  **A failure is not a value.** Recording a large number for a crashed job
  teaches the surrogate the region is bad, which is a claim about the objective
  nobody measured — the objective may be excellent there and the cluster merely
  unreliable. Failures are recorded as failures, retried to a bounded limit,
  counted, and only then abandoned; `test_workers` checks no objective value is
  invented along the way.

  **Cost is charged for failed attempts.** A timeout that burned an hour cost
  an hour, and charging only successes makes an unreliable configuration look
  cheap — exactly backwards for a cost-aware policy.
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
- [x] Verify every generated derivative kernel against a complex-step or
  Richardson-extrapolated finite-difference oracle, with the symmetry of the
  Hessian checked exactly. `test_generated_kernels` sweeps *all* of
  `src/generated`, so adding a leaf without evidence means adding it here too.

  The oracle is Richardson-extrapolated central differences of each leaf's own
  primal. Complex-step would be sharper — no subtractive cancellation at all —
  but it needs the leaf emitted over complex arithmetic and FortSym emits real
  leaves. That limitation is recorded rather than worked around; the roadmap
  allows either oracle, and Richardson pins every derivative to at least eight
  digits here.

  Beyond agreement, each leaf is checked against the identities its derivation
  implies but the emitted code does not enforce, because a generated expression
  can be numerically right at a point and structurally the wrong function: the
  acquisition leaf must depend on `mu` and `best` only through their gap and
  must treat `xi` as exactly a shift of the incumbent; the preference leaf's
  two orderings must be complementary, which is the identity a misplaced sign in
  the `erfc` argument would break; the rescaling leaf's log derivatives must
  equal the value and its negation. Each leaf is also exercised in the regime
  its derivation is least comfortable in — the deep EI tail and the
  near-zero-spread limit — since that is where a generated kernel actually
  fails.

  Hessian symmetry is checked where a Hessian exists: exactly, in
  `test_fortml_adapter` and `test_derivative_gp_input_hvp` upstream. No
  currently generated FortBO leaf emits a Hessian, which is why the check lives
  there rather than here.
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
- [x] Benchmark all three modes plus their combinations against TuRBO on
  matched budgets, counting adjoint cost honestly, and report where derivative
  information does *not* pay for itself. `test_dturbo_modes` does the counting
  in **adjoint-equivalents**, not evaluations: a reverse-mode gradient costs a
  small multiple of a primal, and a comparison measured in "evaluations" while
  taking a gradient at every one would be spending several times as much and
  reporting a tie as a win.

  **The measured finding is that derivative observations lose here.** At four
  adjoint-equivalents per gradient — the conservative end of the usual range —
  the value-only run beat the derivative run on Branin at two budgets, on a
  four-dimensional Levy, and on a smooth bowl. The evaluation count dominates:
  18 gradient-informed points do not beat 90 plain ones.

  Rather than a binary verdict at one assumed cost, the test measures the
  **break-even multiple**: how many plain evaluations a gradient-informed one
  must be worth before it stops being a bargain. That is the useful quantity,
  since the answer depends entirely on the adjoint cost, and it is well below
  four on these benchmarks.

  Two scope limits, stated rather than buried. The candidate search here is
  random-restart EI, so this measures mode 1 alone and not mode 3's
  gradient-based in-region optimization. And these are low-dimensional smooth
  problems; a gradient supplies `d` numbers per adjoint, so the balance shifts
  with dimension, and the regime where it shifts far enough has not been
  measured — the derivative GP's `n(1 + d)` rows make a per-step refit at large
  `d` too slow for a unit test, which is itself a finding worth recording.
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

- [x] Add resident posterior sampling, acquisition evaluation, reduction, and
  candidate-batch kernels for CPU/OpenACC. `src/fortbo_device.F90` scores and
  reduces a candidate batch with the array never leaving the device: the
  moments go up once and only the chosen index and its value come back.

  **The device claim is checked as determinism, not speed.** The device answer
  must be *bit-identical* to the host's — not within a tolerance, since a
  tolerance would accept exactly the reduction-order variation that makes a run
  unreplayable. Ties break to the lowest index on both paths, which is what
  makes exact agreement achievable: without a stated rule a sequential sweep and
  a parallel reduction could pick different tied candidates and no numerical
  tolerance would hide it, because the *index* would differ. Two device runs are
  also compared with each other, since a reduction whose order varied between
  launches would pass a single comparison.

  **Verified on hardware.** A standalone OpenACC probe against the two RTX 5060
  Ti devices present here ran the kernel on device and selected index 4712 with
  a value bit-identical to the host's. That probe also caught a real defect: the
  scoring function needs `!$acc routine seq` to be callable from inside a device
  kernel, and without it the build silently keeps the work on the host. A
  host-only test would never have shown it.

  Absent OpenACC support in the build, `fortbo_device_available` is false by
  construction and the resident path **refuses by name** rather than falling
  back silently. A benchmark row claiming a device number that was produced on a
  CPU is worse than a missing row, and `fortbo_device_name` lets a row say which
  path it came from rather than leaving it to be inferred from timing.
- [x] Keep the TuRBO/DTuRBO inner loop resident: perturbation masking,
  per-region Thompson draws, and the cross-region argmin.
  `fortbo_device_turbo_select` puts every region's candidates in **one** array,
  scores them with **one** kernel, and reduces across **all** regions at once.
  There is no per-region launch and no per-region transfer, which is what the
  roadmap's "a host round trip per region per iteration is a failed GPU claim"
  actually requires — and it is also why the cross-region bandit falls out of
  the same reduction rather than needing a second pass.

  A masked-out candidate is given an infinite realization rather than being
  compacted away, because compaction would need a host-visible count and that
  is precisely the round trip being avoided. Masking *every* candidate is
  refused rather than returned as a selection with no winner.

  The reduction breaks ties to the lowest index, matching the scoring kernel, so
  host and device agree bit-for-bit and a pooled bandit decision is
  reproducible. `test_device` checks the mask is respected, that no unmasked
  candidate anywhere beats the pooled winner — which is what makes the argmin
  genuinely cross-region rather than best-in-region-one — and that the device
  result is identical to the host's.

  **Not yet resident**: Sobol candidate generation and the posterior
  gradient/Hessian evaluation. Sobol's Gray-code recurrence is inherently
  sequential in the draw index and needs a skip-ahead formulation to parallelize;
  the posterior derivatives need FortML's device JVP/HVP products, which are the
  next item. Both are named rather than left to be discovered from a profile.
- [ ] Keep FortAD-bearing acquisition graphs on FortAD/FortSym until complete
  device JVP/VJP/HVP products exist. Use CUDA for fixed sampling/reduction
  kernels where OpenACC cannot preserve residency or determinism.
- [x] Benchmark against BoTorch/GPyTorch, JAX, and deterministic NumPy on
  matched functions, models, precision, seeds, restart counts, and stopping
  criteria. Report regret/sample efficiency separately from wall time.
  Two halves. **Correctness** (`test_cross_framework`): a pinned RBF GP checked
  against all three references, which the generator first cross-checks against
  each other. Posterior mean and standard deviation agree to 7e-16. Two
  differences had to be found and removed rather than tolerated -- GPyTorch's
  constrained setters do not round trip in float64 (its softplus inverse runs
  in single precision, so a requested lengthscale of 0.7 is stored 1.2e-8
  away), and FortML adds a default 1e-10 diagonal jitter that the references do
  not. Both were invisible in isolation and both were real model differences.
  BoTorch's legacy analytic EI remains 2e-9 from the direct algebra; JAX agrees
  with NumPy at 4e-16 on the same quantity, so BoTorch is the outlier, which is
  what the logEI paper (arXiv:2310.20708) describes.
  **Sample efficiency** (`test_regret_benchmark`): Branin, identical listed
  initial design rather than a shared seed, matched budget and stopping
  criterion. FortBO reaches 8.9e-4 final regret against BoTorch's 5.2e-4 in 30
  evaluations. Wall time is reported in its own columns and never mixed in: it
  measures a Fortran binary against a Python stack, and on the expensive
  objectives BO exists for it is nearly irrelevant. The inner searches are not
  identical -- BoTorch runs L-BFGS-B on acquisition gradients, FortBO's
  value-only GP exposes none and samples instead at a matched acquisition
  budget -- and that is stated rather than papered over. Generators live in
  `fortbo-bench/scripts/emit_reference.py` and `emit_regret.py`.

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
- [x] Add constrained synthetic functions, noisy objectives, and
  multi-objective fixtures with known Pareto fronts or dense reference grids.
  `src/fortbo_fixtures.f90` holds the Gardner and Townsend constrained problems
  and the ZDT1/ZDT2 multi-objective pair.

  Constraints are kept as separate functions, never folded into the objective
  as a penalty: a pre-penalized fixture makes a method that respects the
  constraint indistinguishable from one that ignores it, which is the only
  thing a constrained benchmark is for. `test_fixtures` checks the property
  that makes each fixture worth having — the unconstrained optimum over the box
  must be *infeasible*, so the constraint actually bites. A fixture whose
  constraint is slack at the optimum is worthless as evidence.

  The Pareto fronts are analytic rather than gridded. Two checks keep them
  honest: every stated front point must be attained by a real input, and no
  sampled input may dominate one — a "front" that a random draw beats would
  make every hypervolume comparison meaningless. The front's hypervolume is
  computed through `fortbo_pareto`, so fixture and indicator agree.

  ZDT2 is in the suite because its front is *concave*. A weighted-sum
  scalarization reaches only the convex hull, so on ZDT2 it collapses to the
  endpoints however the weights are swept — which the test demonstrates
  alongside ZDT1, where the same sweep does cross the interior. A suite of
  convex fronts alone would report scalarization methods as complete when they
  are not.

  Noise is a standard deviation the caller applies, not a draw taken inside the
  fixture: a fixture that drew its own noise would be unreplayable and two
  methods could not be compared on the same realizations.
- [x] Add the high-dimensional trust-region fixtures with exact gradients:
  Ackley-200 on `[-5,10]^200`, Levy and Rosenbrock at `d = 100..500`. The
  functions and their analytic gradients were already in
  `src/fortbo_benchmarks.f90` at arbitrary dimension; what was missing was
  evidence that they are correct *at those dimensions*, which
  `test_high_dimensional` now supplies.

  Gradients are checked against Richardson-extrapolated central differences at
  random interior points — the independent statement of what a derivative is,
  sharing no code with the analytic expressions. Coordinates are sampled rather
  than swept, since at `d = 500` a full sweep is thousands of evaluations per
  step size and a systematic error shows up in a sample just as surely.

  Two checks catch what a value comparison alone would not. The gradient must
  *vanish* at each stated optimum: a function can take the right value at the
  right point and still be the wrong function. And Ackley's optimum must not sit
  at the centre of its box — the `[-5,10]` domain is deliberately asymmetric,
  and a fixture that silently recentred would place the answer exactly where a
  search starts.
- [x] Add the 60D rover trajectory fixture. `src/fortbo_rover.f90` implements
  the reward the TuRBO paper states in appendix F.2 — a B-spline through 30
  planar control points, `-20` per collision, `-10` times the L1 endpoint
  distances, plus `5` — read from the paper rather than recalled.

  **The obstacle map is ours, and that is stated rather than glossed.** The
  paper gives the reward and cites Wang et al. for the terrain; the layout is
  not in the paper, so reproducing the published *numbers* is not possible from
  it. Inventing a map and calling the result "the rover problem" would produce
  scores that look comparable to the literature and are not. What is reproduced
  is the structure, which is what makes it a useful 60-dimensional test, and any
  claim against published numbers has to say which map it used.

  The fixture **refuses a gradient by name**. Its collision term is piecewise
  constant, so the derivative is zero almost everywhere and undefined on the
  obstacle boundaries; a finite difference would return zero across most of the
  domain and something enormous at a crossing, neither of which is a gradient.

  `test_rover` verifies the reward against the formula on trajectories whose
  terms are known by hand, and pins two properties a weaker fixture would fail:
  the straight line from start to goal must collide, or the obstacle map is
  decorative and the problem reduces to writing two endpoints in sixty numbers;
  and moving the *interior* control points must change the objective, which an
  implementation reading only its endpoints would not do.
- [ ] Add the 14D robot pushing fixture, which needs a rigid-body physics
  simulator rather than a closed-form reward.
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
- [x] Keep CPU, transfer-inclusive GPU, resident GPU, and typed refusal rows
  separate, with source/toolchain/device provenance.
  `src/fortbo_provenance.f90` holds the four lanes and refuses to blur them.

  The distinction that matters most is **transfer-inclusive versus resident**.
  Both are legitimate numbers answering different questions, and reporting a
  resident timing as though it were what a user experiences is the single most
  common way a GPU claim misleads. They are separate lanes, and a mean is taken
  within a lane, never across.

  **A refusal is a result and stays in the table**, carrying no timing at all
  rather than a zero. A zero would be averaged in and would drag every summary
  toward it invisibly; dropping refused rows entirely is what turns "this works
  on four of nine configurations" into "this works". A refusal must also say
  *why*, or it is indistinguishable from a row nobody got round to filling in.

  A lane with no measured rows reports **unavailable**, not zero — "fast" and
  "never ran" are different claims. And comparability requires matching case,
  precision, and *source revision*: comparing a CPU row from one revision
  against a GPU row from another measures the revisions rather than the lanes,
  which is the most common way a speedup is manufactured by accident.


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
