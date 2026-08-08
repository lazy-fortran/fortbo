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

The active cross-repository FortBO reproduction and performance campaign is
documented in [FortBO reproduction and performance campaign](#fortbo-reproduction-and-performance-campaign)
near the end of this roadmap.

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
- [x] Implement predictive entropy search. **C1.1, C1.2, C2 and C3 all built
  against the paper.** Written from arXiv:1406.2541 read rather than recalled — see
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

  **C1.2 and C2 are now implemented too**, by expectation propagation, in
  `fortbo_pes_latent_constraints`. The latent vector is the paper's
  `z = [f(x*); diag(grad^2 f(x*))]`. C1.1 needs no code: conditioning on a
  vanishing gradient and known off-diagonal Hessian entries is linear-Gaussian
  and is already folded into the caller's prior. What remains is the
  non-Gaussian part — one truncation factor per Hessian diagonal entry for
  C1.2, and the paper's *soft* maximum `Phi((z_1 - ymax)/sigma)` for C2, soft
  because the observations are noisy and a hard constraint would require
  inference on the latent values behind them.

  The paper's convention is kept rather than silently flipped: it maximizes,
  so the Hessian diagonal is negative and `f(x*)` exceeds `ymax`, and a
  minimizing caller passes negated values. Translating inside would have
  hidden the one place a sign error is unrecoverable.

  **A real error, caught by an exact oracle.** The paper writes the site
  update as `v_site <- 1/beta - v_cavity`, a relation between *variances*.
  The first implementation read it as a relation between precisions
  (`beta - cavity_precision`) and produced marginals wrong by order a hundred.
  Because the prior is diagonal and each constraint touches one coordinate,
  the true posterior factorizes and every marginal has a closed form, so EP
  must reproduce it *exactly* — which is what made an error of that size
  visible instead of looking like approximation error. The module now works
  from tilted moments in natural parameters rather than the paper's
  `kappa`/`beta` shorthand, whose two factor types differ only in the sign
  before `1/kappa`.

  `test_pes_constraints` checks against the closed forms *and* against direct
  numerical integration of the constraint definitions, which shares no algebra
  with the module — the closed-form check alone would verify the EP
  bookkeeping but not the moments. Mutation testing confirms it: flipping one
  sign in the truncation moments trips four assertions. The tail behaviour is
  pinned separately, because at `alpha = -40` both the normal density and its
  integral underflow while their ratio stays near 40, and a `NaN` there would
  propagate through every coordinate rather than stay local.

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
- [x] Adapt the FortML multi-task and deep-kernel GPs, which have no posterior
  contract route yet. `fortbo_structured.f90` is that route; the modelling
  stays in FortML where it is generic.

  The deep-kernel GP did not exist upstream at all, so it was written first:
  `fortml_deep_kernel_gp.f90`, from Wilson et al., *Deep Kernel Learning*
  (arXiv:1511.02222), fetched into fortml-bench provenance and read. Equation
  (5) composes a base kernel with a neural feature map; the weights are
  learned jointly with the base hyperparameters through the marginal
  likelihood, and `weight_gradient` implements the paper's own factorization
  -- equation (7) contracted with dK/dg, then backpropagated -- rather than
  differencing. The finite-difference oracle immediately caught a real error:
  the first version accumulated only the first-argument derivative, on the
  wrong reasoning that the transposed pair would supply the rest, and was
  exactly half the true gradient. KISS-GP is deliberately not implemented, so
  the model is exact and carries a dense GP's cubic cost.

  **Multi-task needs a decision the model does not make.** A multi-output GP
  is a posterior over a vector and an acquisition is a function of a scalar,
  so `target_output` is required rather than defaulted -- a default would
  silently optimize the first output of a three-output model and look
  entirely normal. The variance reported is that output's own marginal from
  the joint covariance diagonal; what multi-task buys is a sharper posterior
  on the target from the other tasks' data, not a different notion of
  uncertainty. Neither adapter claims moment gradients, so a gradient-based
  search gets a refusal by name and can fall back to
  `fortbo_search_acquisition`.
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
- [x] Reproduce the paper's qualitative ordering on Ackley-200, the 60D rover
  trajectory problem, and the 14D robot pushing problem against a pinned
  `uber-research/TuRBO` and the BoTorch `turbo_1` tutorial. The Sobol blocker
  is cleared — `fortnum_sobol` now covers dimensions into the hundreds with
  computed primitive polynomials, checked by exact equidistribution in all 200
  coordinates — and the harness plus pinned baselines are now recorded.

  **Harness done, all three problems wired up.**
  `test/turbo_ordering_harness.f90` with `test_turbo_ordering_push_slow`,
  `_rover_slow` and `_ackley_slow`: three seeds each, matched budgets, matched
  initial designs, medians rather than single runs.

  On **push-14** the paper's central claim holds and is asserted: both TuRBO
  variants beat quasi-random search decisively (-3.37 and -2.73 against
  -1.63). On **rover-60** and **ackley-200** the comparison is recorded but
  *not* asserted, because the budget that fits inside `fo`'s five-minute cap
  cannot test the ordering. Rover-60 at 22 evaluations has TuRBO-1 scoring
  1331 against random search's 1190 — it loses, and it should: a GP fitted to
  a couple of dozen points in sixty dimensions carries almost no information,
  so the trust region contracts around an arbitrary point while undirected
  search still covers the space. The paper runs thousands of evaluations.
  Asserting its ordering at these budgets would mean either tuning until it
  appeared or shipping a test that fails for a correct implementation.

  One real defect was found and is now enforced rather than merely fixed.
  TuRBO-`m` spends `m * n_initial` evaluations before any region has a usable
  surrogate; when the budget does not clear that, the run emits nothing but
  initial-design points, and because those come from the same seeded uniform
  stream the random arm draws from, TuRBO-`m` and random search return
  **bit-identical** values. That looked exactly like a plumbing bug. The
  harness now checks `budget > 2 * n_regions * n_initial` at run time, because
  the failure is silent and produces numbers that look like a tie.

  Also recorded: the third arm is quasi-random search and is named as such,
  not dressed up as the paper's global-BO baselines, which are out of reach at
  these dimensions.

  **The pinned baseline now runs.**
  `fortbo-bench/scripts/run_turbo_baselines.py` drives `uber-research/TuRBO`
  *unmodified*, at the commit `fetch_provenance.py` records, on Ackley-200.
  Ackley alone, deliberately: it is stated in closed form and both sides use
  the same `[-5, 10]` box, so the two implementations are provably optimizing
  the identical function. Comparing on the rover or the pushing problem would
  measure our fixtures — structurally faithful but numerically ours — rather
  than the optimizers, while looking like a TuRBO comparison.

  At FortBO's own budget (16 evaluations, 5 initial, 200 dimensions) the
  reference's TuRBO-1 reaches 13.69 against random search's 13.94, and
  `test_turbo_ordering_ackley_slow` checks FortBO lands in the same region.
  Agreement *in kind* is what is asserted, not in value: the reference fits GP
  hyperparameters with Adam every step while FortBO runs a fixed lengthscale,
  which is a difference in the surrogate rather than in the trust-region logic
  under comparison, so demanding equal numbers would demand that two different
  models agree.

  **The reference corroborates the negative result above.** Given room to run
  (60 evaluations, 10 initial) the authors' own code reaches 12.56 with
  TuRBO-1 against 13.51 with TuRBO-5, with TuRBO-5 barely ahead of random's
  13.53 — the same ordering FortBO shows, from the same cause. That is direct
  evidence that declining to assert TuRBO-m over TuRBO-1 at small budgets was
  right, and that tuning until the paper's ordering appeared would have been
  fitting the test to an expectation the reference implementation does not
  meet either at these budgets.

  **The BoTorch `turbo_1` tutorial is also recorded.**
  `fortbo-bench/scripts/run_botorch_turbo.py` runs its independent
  implementation on the same Ackley-200 budgets and seeds. At the matched
  budget its median is 13.38, while the pinned authors' implementation is
  13.69 and FortBO is 13.65; at the roomy budget the corresponding medians are
  12.77 and 12.56 for the two Python references. The three-way fixture keeps
  the models' fitting differences explicit rather than treating close values
  as proof of identical surrogate state.

  The rover and pushing rows remain explicitly fixture-local: the pinned
  repositories do not provide the same obstacle map or rigid-body simulator.
  The benchmark records that non-comparability instead of presenting scores
  from different objectives as an external baseline. Ackley is the direct
  pinned cross-implementation comparison; all three FortBO fixtures have
  independent seeded ordering tests and regenerable records.

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
- [x] Derive the posterior gradient and Hessian expressions for each supported
  kernel through FortSym and emit them; block-matrix work blocks on FortSym M9
  and must be fixed there.

  **The M9 dependency was recorded more broadly than it is.** A GP posterior
  gradient *looks* like block-matrix work and is not. Writing

      m(x) = k(x)^T alpha,        alpha = K^-1 y
      v(x) = k(x,x) - k(x)^T K^-1 k(x)

  the derivatives are `dm/dx_i = (dk/dx_i)^T alpha` and
  `dv/dx_i = dk(x,x)/dx_i - 2 (dk/dx_i)^T K^-1 k(x)`, and every matrix in
  them — `alpha`, `K^-1 k(x)` — is a *numeric* vector from a runtime solve
  against data. No symbolic inverse, determinant or eigenvalue appears
  anywhere. So Bareiss, Dixon lifting and symbolic eigenvalues are not on this
  path, and the item was never actually blocked. This is the second time in
  this roadmap a recorded FortSym blocker turned out to be narrower than
  written; both were resolved by asking what the expression actually needs.

  The symbolic content is two things, and both are emitted. **Per-kernel input
  derivatives**: FortSym now carries `gen_rbf_derivatives`, `gen_matern12_hvp`,
  `gen_matern32_hvp` and `gen_matern52_hvp`, covering every kernel FortBO's
  posteriors use — 5/2 was the gap, and it is the default precisely because it
  is twice differentiable, so a posterior Hessian over it exists where one over
  3/2 does not at coincident points. **The variance-to-standard-deviation
  chain rule**: `gen_posterior_moment_leaf` derives

      s = sqrt(v),  s' = v'/(2 sqrt v),  s'' = v''/(2 sqrt v) - v'^2/(4 v^{3/2})

  and emits it as `src/generated/fortbo_generated_posterior_moment_leaf.f90`.
  The `v^(-3/2)` term is exactly `s^(-3)`, which is where
  `FORTBO_SD_HESSIAN_FLOOR` comes from — at a standard deviation of 1e-6 that
  term already reaches about 1e18. Deriving it makes the bound something the
  generator produces rather than something a comment claims.

  `test_generated_kernels` gains the matching check, against
  Richardson-extrapolated differences of a *stated* variance profile rather
  than of the leaf's own output, and sweeps down to small variances where the
  `v^(-3/2)` term dominates. Halving that term's coefficient trips it.
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
- [x] Keep FortAD-bearing acquisition graphs on FortAD/FortSym until complete
  device JVP/VJP/HVP products exist. Use CUDA for fixed sampling/reduction
  kernels where OpenACC cannot preserve residency or determinism.
  `src/fortbo_placement.f90` enforces the rule rather than leaving it as an
  intention.

  **The precondition is not met today, and the module states that from what
  exists rather than assuming it.** FortML carries device JVP and VJP products
  for the multi-output GP, the variational classifiers and the ELBO, and
  device HVP products only for the linear pipelines. The exact GP regression
  FortBO's posteriors are built on has no device derivative products at all.
  So every derivative-bearing acquisition graph over a GP surrogate belongs on
  the host now.

  The word that carries the weight is *complete*. A partial product set is
  worse than an empty one because it **runs**: a graph placed on a device with
  a forward product but no reverse one falls back for the missing piece and
  produces derivatives from two code paths with no record of which. So all
  seven incomplete combinations are refused as firmly as the empty one, and
  the test sweeps all seven rather than checking the empty case alone.

  Refusals are by name and name the missing products, since "incomplete" is
  not actionable. A refused request never returns a quiet host placement —
  that is how a benchmark table acquires a device row that never ran on a
  device, which `fortbo_provenance` exists to prevent one layer down. Whether
  a graph is derivative-bearing is read from the posterior's own capabilities
  rather than a caller-supplied flag, so it cannot drift from what the model
  can actually do. Value-only graphs are *not* blocked: they never needed the
  products, and blocking them would have made the value-only device kernels in
  `fortbo_device` unreachable.

  **CUDA is deliberately not used, and that is a measurement.** The roadmap
  permits dropping to CUDA where OpenACC cannot preserve residency or
  determinism. It can: `fortbo_device` pools every region's candidates into
  one array, runs one kernel and takes one reduction with an index tie-break,
  and `test_device` checks the result is *bit identical* to the host across
  repeated launches. Writing CUDA for kernels OpenACC already handles
  deterministically would add a second code path to keep in agreement for no
  measured gain.
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

- [x] Inventory every FortBO source module in the benchmark matrix, including
  direct external lanes, published-policy lanes, independent reference-only
  lanes, and device/contract refusal lanes. The complete row set is generated
  by `fortbo-bench/scripts/run_feature_matrix.py`; the benchmark repository
  owns the JSON/CSV evidence and refuses to call an unavailable or
  non-comparable lane a speed win.

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
- [x] Add the 14D robot pushing fixture, which needs a rigid-body physics
  simulator rather than a closed-form reward. `src/fortbo_push.f90`.

  The TuRBO paper does not define this problem; its README points at
  `zi-w/Ensemble-Bayesian-Optimization` and lists the three changes it made.
  That repository is now in `fortbo-bench` provenance and was read. Every
  number comes from it: the box, the two objects' starts and goals, the
  shapes and densities, the friction joints capped at force 5 / torque 2 for
  objects and 2 / 2 for hands, the proportional hand controller
  `F = m (v* - v) * 30`, the 1/100 timestep, the `int(10 x)` step
  quantization, the 100 settling steps, and the reward. None of it is
  derivable.

  **What is ours is named.** Box2D is not reproduced, so published numbers
  for this problem are not comparable: contact shapes are a capsule per hand
  and a disc per object, which loses the square's corners, though masses and
  inertias come from the true shapes. What is reproduced is the structure
  that makes the problem hard -- fourteen interacting parameters, an
  objective that is genuinely *discontinuous* at contact changes and at the
  quantized duration, flat regions where a hand misses entirely, and two
  coupled sub-problems sharing one table. The gradient is refused by name.

  Two test-design errors are worth recording because both were caught by
  mutation testing rather than reasoning. Asserting each coordinate changes
  the objective under a single probe failed on the push-duration coordinate
  for entirely correct physics -- extending a push after contact is lost does
  nothing -- so the claim is now that no coordinate is inert under *every*
  probe. And the suite claimed to pin that contact impulses act at the
  contact point; removing that coupling left it passing, because the torque
  coordinates also steer the hand's orientation. There is now no check for
  it, and the reason is stated: the reference's own friction caps suppress
  object rotation so completely (measured final spin 1.6e-15 for a
  deliberately off-centre strike) that the application point is not
  observable in the final state.
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

## FortBO reproduction and performance campaign

This is the execution plan for the cross-repository experiment requested in
August 2026: reproduce the published Landreman TuRBO configuration and the
Glas--Padidar--Kellison--Bindel DTuRBO configuration, then replace only the
optimizer with FortBO and measure parity, cost, and performance. It extends
the completed derivative-free campaign below; it does not retroactively label
the earlier literature audit as an implementation or performance result.

The campaign repository is this checkout,
`/mnt/storage/code/simsopt-dfo`, at
`git@gitlab.tugraz.at:D461BDE997455AF1/simsopt-dfo.git`. Authenticated GitLab
search found no `simsopt-dfo` project below `plasma/proj` or
`plasma/proj/stel`; the existing owner-namespace project is therefore reused.
Moving it to a group later is a GitLab administration task, not a reason to
duplicate the checkout or change the experiment. The FortBO library remains
at `/mnt/storage/code/lazy-fortran/fortbo`; generic package benchmarks remain
in `/mnt/storage/code/lazy-fortran/fortbo-bench`. Large source archives,
environments, scratch directories, and numerical ledgers stay outside Git.

### 1. Frozen provenance

Every run must begin by recording the following values in a manifest. A
manifest with a missing digest is not a reproduction.

| Item | Frozen value |
| --- | --- |
| This campaign repository | `D461BDE997455AF1/simsopt-dfo`, `main` |
| Landreman Zenodo source archive | `/home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar` |
| Landreman archive SHA-256 | `7037bb0abbaaa7ccc4bc7b9f5434e41b18ecdf97af04cf8ae244ea2ae20c428f` |
| Landreman driver | `software/alpha_opt/scripts/driver_turbo_PCA_unconstrained.py` inside that archive |
| Landreman PCA data SHA-256 | `745548e503beda2f8794b169b8a8abd55adeddfa4acc71c2a76045b61acaac7c` |
| Landreman VMEC input SHA-256 | `88318d8b2ab17741110a11bc5141ecfbbd862eb5ff02b47f808bc527c6bf263e` |
| Bindel/Glas paper | `/home/ert/proj/stellopt-talk/literature/glas2022_coil-dturbo.pdf` |
| Bindel/Glas paper SHA-256 | `24cc2600e8b20b74b80b96ce294286c66bea19c2e96808e516abae5f960f8d0b` |
| Bindel/Glas local harvest | `/home/ert/data/simsopt-dfo-harvest/coil-dturbo-paper-2110.07464/2110.07464.tar` |
| Bindel/Glas harvest SHA-256 | `61e1dc8912ddb4825b6ac5ad5d26c2a0d86280fb71d86f2ef3991dfb5c40a693` |
| FortBO at this audit | `/mnt/storage/code/lazy-fortran/fortbo`, commit `b094bc1daa8b10eaee2ad5f8211cd7d09264fcff` |
| Simsopt-dfo at this audit | commit `d69565dadcb8c2eaf47527e61a9c89155ce2e650` |

The commands to verify the archive-level pins are:

~~~
sha256sum \
  /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar \
  /home/ert/proj/stellopt-talk/literature/glas2022_coil-dturbo.pdf \
  /home/ert/data/simsopt-dfo-harvest/coil-dturbo-paper-2110.07464/2110.07464.tar

tar -xOf /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar \
  20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt/data/20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5 | sha256sum
tar -xOf /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar \
  20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt/data/input.vmec | sha256sum
~~~

The current `simsopt-dfo` B5 and B6 campaigns are useful controls but are not
silently substituted for the two requested papers:

- B5 is the ConStellaration simple-to-build QI problem at upstream commit
  `112b20ae07193910d467d26033fe51022e641b9f`, with existing five-seed
  BoTorch TuRBO-1 and TuRBO-m ledgers. It is the first real-physics place to
  test a value-only FortBO substitution after the synthetic gates.
- B6 is the Bindel/Landreman/Padidar alpha-loss case at
  `alpha_particle_opt` commit `b04ad48c22a32e8c8d5561f9b025e5360f3a122a`.
  It is a separate fast-ion comparator and does not reproduce the Glas et al.
  coil DTuRBO paper.
- The Glas et al. paper and harvest contain figures and manuscript material,
  but no FOCUS source, W7-X input, perturbation covariance file, or optimizer
  ledger. A numerical claim on that case is blocked until those inputs are
  recovered or a separately labeled reimplementation is accepted.

### 2. The exact upstream configurations

#### 2.1 Landreman data-informed PCA TuRBO

The source is the archived script, not a newly written approximation. Extract
it and inspect the source before every replay:

~~~
export LANDREMAN_ARCHIVE=/home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar
export LANDREMAN_RUN=/home/ert/data/landreman-reproduction/20260514-01
mkdir -p $LANDREMAN_RUN
tar -xf $LANDREMAN_ARCHIVE -C $LANDREMAN_RUN
export LANDREMAN_ROOT=$LANDREMAN_RUN/20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt
sed -n '1,760p' $LANDREMAN_ROOT/scripts/driver_turbo_PCA_unconstrained.py
~~~

The original contract to preserve is:

- aspect ratio `6.0`; `minor_radius = 3.1 / aspect_ratio**0.38`;
  `major_radius` derived from it; vacuum disabled; `max_B_target=12.0`; one
  max-B iteration;
- data-informed `SurfaceWeightedPCA`, `dim_x=20`, unit box `[0,1]^20`, with
  the archived PCA file above; no alternate PCA fit or reordering;
- `n_particles=25000`, `t_max=1e-1`, `tau=0.1`, `maxloss=0.02`,
  `t_block=1e-3`, `tol=1e-6`, `min_dt=1e-9`, and the exact
  `compute_alpha_loss`/VMEC++ evaluator;
- failure-as-value semantics exactly as the source: VMEC failures become
  `fail_val=5.5`, the DMerc failure value is `-0.5`, and no failed result is
  dropped from the training ledger;
- `num_evals=10000`, `batch_size=1`, five minutes of wall-clock reserve in a
  three-hour job, save frequency one, and CPU-only GP fitting on rank zero;
- the manager uses MPI rank zero plus `size-1` workers. The initial design is
  `max(2*dim, 2*(size-1))` scrambled Sobol points with `seed=0`, and only
  completed observations enter the GP;
- the trust state is `length=0.8`, `length_min=0.5**7`, `length_max=1.6`,
  success tolerance `10`, and failure tolerance
  `ceil(max(4/batch_size, dim/batch_size))`;
- the local candidate count is `min(5000, max(2000, 200*dim))`, therefore
  `2000` at `dim=20`; the TS mask probability is `min(20/dim,1)` and every
  candidate is forced to perturb at least one coordinate;
- the archived production path uses qEI (`acqf="ei"`), `q=batch_size=1`,
  `num_restarts=10`, `raw_samples=512`, with the commented TS path retained
  as a separate ablation, not mixed into the primary result;
- the GP is BoTorch `SingleTaskGP` with a `MaternKernel(nu=2.5,
  ard_num_dims=20)` inside a `ScaleKernel`, `GaussianLikelihood` noise
  interval `[1e-8,1e-3]`, lengthscale interval `[0.005,4.0]`, standardized
  observations, `ExactMarginalLogLikelihood`, and unlimited Cholesky size;
- the manager's completion order, Sobol stream, source update order, and
  current state-update behavior are part of the historical replay. A clean
  reimplementation may correct a source bug only in a named ablation and must
  preserve the unmodified replay row.

The unmodified upstream smoke command is:

~~~
cd $LANDREMAN_ROOT
OMP_NUM_THREADS=12 mpiexec -n 9 python scripts/driver_turbo_PCA_unconstrained.py
~~~

The `9` is a placeholder for the original allocation only if the source run
manifest proves that allocation; otherwise use the recorded original MPI size.
Do not claim an exact run by changing the hard-coded PCA path, resolution,
failure values, or worker count without recording that as a deviation. The
portable reproduction copies the two archived data files to a run-local path,
changes only the path binding, and records both original and resolved paths and
their digests.

The FortBO replacement must expose the same normalized `x`, objective values,
failure values, Sobol seed, initial points, completion-driven `ask`/`tell`
schedule, qEI or TS choice, trust-state transitions, GP fit data, and stopping
condition. It must not call the alpha evaluator from Fortran directly until a
separately tested C/Python or file-based ABI exists; the first parity driver
may keep the evaluator in Python and use FortBO as a library policy.

#### 2.2 Glas--Padidar--Kellison--Bindel DTuRBO/AdamCV

The paper's W7-X-like configuration is:

- five distinct modular coils per field period, stellarator symmetry and five
  field periods, 50 physical coils;
- Fourier order `NF=6`, `N=3*5*(2*6+1)=195` free coefficients for one field
  period; `Nseg=64`, `Ntheta=Nzeta=64`;
- perturbations `U ~ N(0,C)` from the periodic GP/Fourier covariance;
  perturbation amplitudes `p=5 mm` and `p=10 mm` are separate experiments;
- `omega_B=100`, `omega_L=0.5`, target coil length `8.0`, separation
  `epsilon_c=0.23 m`, quadratic penalty `lambda=100`, and alpha-quasimax
  `alpha=10000`;
- the stage-one box is centered at circular coils of radius `1.5 m` and has
  `lb,ub=x0 +/- delta*sqrt(Var[U])`, with `delta` chosen so the translational
  mode has half-width `1.5/2 m`;
- DTuRBO stage one has `200` initial evaluations, batch size `100`, and a
  maximum of `100000` objective evaluations;
- stage two is AdamCV, at most `2000` iterations, gradient batch size `10`,
  `eta=0.001`, `gamma=0.01`, `beta1=beta2=0.95`, and `epsilon_A=1e-10`;
- the separate local control uses `50000` gradient evaluations, ten gradient
  evaluations per step, `eta=0.04`, `gamma=0.1`, the same beta values and
  epsilon, and the paper's SAA/BFGS comparison. It must not be pooled with
  the global DTuRBO result.

Published DTuRBO is not merely “TuRBO with a gradient flag”: it uses a
stochastic variational GP with inducing structure and conditions on paired
function/gradient observations. The current FortBO derivative-observation GP
is an exact dense derivative GP, and `fortbo_dturbo` currently implements a
Newton-BO-style posterior-derivative trust-region step. Those are valuable
components but are not yet paper-level DTuRBO parity. Until the gates below
pass, report three distinct rows: paper DTuRBO (literature or recovered
source), FortBO exact derivative mode, and value-only FortBO/TuRBO.

The paper-level run command is intentionally a required interface, not a
pretence that missing FOCUS inputs already exist:

~~~
export COIL_ROOT=/home/ert/data/simsopt-dfo-harvest/coil-dturbo-paper-2110.07464
test "$(sha256sum $COIL_ROOT/2110.07464.tar | awk '{print $1}')" = \
  61e1dc8912ddb4825b6ac5ad5d26c2a0d86280fb71d86f2ef3991dfb5c40a693
# After the FOCUS source, W7-X input, covariance, and driver are recovered:
cd $COIL_ROOT/recovered-focus
mpiexec -n 14 python run_dturbo_adamcv.py \
  --nf 6 --nseg 64 --ntheta 64 --nzeta 64 \
  --coils-per-period 5 --field-periods 5 --dimension 195 \
  --omega-b 100 --omega-l 0.5 --length-target 8.0 \
  --separation 0.23 --lambda-penalty 100 --alpha-quasimax 10000 \
  --perturbation-mm 5 --global-budget 100000 --global-batch 100 \
  --global-initial 200 --local-iterations 2000 --local-batch 10 \
  --eta 0.001 --gamma 0.01 --beta1 0.95 --beta2 0.95 --epsilon 1e-10 \
  --seed SEED --output results/bindel/dturbo-5mm-SEED.json
~~~

The recovered driver must also be run with `--perturbation-mm 10` and with
the local SAA/BFGS control. `SEED` is not invented: it comes from the
recovered source or the paper's experiment manifest. If no source and seed
ledger are recovered, the result is literature-only and the FortBO run is a
matched reimplementation, never an “exact reproduction.”

### 3. FortBO feature-parity gates

The FortBO substitution is admitted to a real-physics comparison only after
these independent behavioral gates pass. Each gate gets a small synthetic
fixture with an oracle independent of the FortBO implementation and a JSON
record containing the requested and actual calls.

1. **Space and mapping.** Reproduce the same `[0,1]^20` PCA chart, the same
   195-coefficient coil chart, bounds, coordinate ordering, clipped trust
   region, and physical-to-normalized chain rule. Round-trip and boundary
   tests must agree with Python/NumPy to `1e-13` in float64.
2. **Randomness.** Match Sobol direction, scramble, seed, point numbering,
   candidate mask, minimum-one-coordinate rule, TS draw ordering, and region
   seed splitting. If exact Sobol bits cannot be shared across languages,
   freeze a common candidate file and compare policy behavior on that file;
   do not compare two independently generated random streams as if they were
   identical.
3. **Trust state.** Match initial radius, success/failure counters, relative
   improvement threshold, expansion/shrink factors, minimum-radius restart,
   and multi-region placement. Replay a recorded answer stream and compare
   every radius and counter, not only the final best value.
4. **Surrogate.** Match standardized-output convention, Matern-5/2 ARD
   covariance, likelihood noise bounds, lengthscale bounds, exact-vs-sparse
   inference choice, fit tolerance, Cholesky policy, and tie handling. First
   compare posterior mean, variance, derivative blocks, and sampled paths on a
   frozen training set against BoTorch/GPyTorch. This is a numerical parity
   gate, not a test that merely reads FortBO's own output.
5. **Acquisition.** Match qEI and TS as separate methods. For qEI compare the
   objective, gradient, optimizer bounds, restarts, raw samples, and q=1
   selection. For TS compare joint rather than independent posterior draws,
   candidate pooling, no-replacement selection, and the perturbation mask.
6. **Scheduling.** Reproduce the original completion-driven behavior: initial
   points are submitted, the next point is generated only from completed
   observations, the tell is associated with the originating region, and
   worker completion order is preserved in the ledger. Compare a deterministic
   delay fixture with one, two, eight, and 32 workers.
7. **Failures.** Preserve failure-as-value for Landreman and structured
   failure rows for the SIMSOPT cases. A timeout, process loss, retry, solver
   failure, non-finite value, and user cancellation must remain distinct. The
   charged budget counts the truth attempt exactly once; infrastructure retry
   counts separately.
8. **Derivatives/DTuRBO.** For the Glas comparison, implement the published
   inducing-point variational derivative GP, paired value-plus-gradient action,
   Thompson sampling policy, and its bound/trust-region rules, or explicitly
   mark the row as FortBO exact-derivative mode rather than DTuRBO. Validate
   kernel value/value, value/gradient, gradient/value, and gradient/gradient
   blocks against an independent finite-difference or automatic-differentiation
   oracle at ordinary and coincident points.
9. **Restart and persistence.** Stop and resume after every initial wave and
   after a worker failure. The resumed ledger, posterior training set, random
   state, trust state, and final result must match an uninterrupted run for a
   deterministic evaluator.
10. **Objective ABI.** The physics evaluator must be callable without changing
    its code, compiler flags, resolution, tolerances, or failure semantics.
    For the first integration the Python or MPI evaluator may remain the
    worker; only candidate proposal and tell state move to FortBO. Benchmark
    the ABI itself so wrapper overhead is not hidden inside physics time.

The parity result is a table of maximum absolute/relative differences for
points, posterior quantities, acquisition quantities, state traces, and
ledger rows. “It found a similar minimum” is not feature parity.

### 4. Exact commands in this repository

The existing BoTorch controls and common physics ledgers are the starting
point. Run them at a clean commit and publish results outside the checkout:

~~~
export DFO=/mnt/storage/code/simsopt-dfo
export RESULT_ROOT=/home/ert/data/simsopt-dfo-fortbo/results/$(git -C $DFO rev-parse HEAD)
cd $DFO
uv sync --frozen --extra bayesopt --extra mhd --inexact
uv lock --check
uv run pytest tests/test_bayesopt.py tests/test_async_turbo.py \
  tests/test_async_turbo_m.py tests/test_b5_constellaration.py \
  tests/test_b5_async_turbo.py tests/test_b6_alpha_particle.py
uv run ruff check .
~~~

The existing value-only controls are:

~~~
# B5: raw and data-informed TuRBO-1; five paired seeds, 256 truth calls, 8 workers.
SIMSOPT_DFO_CLUSTER=acluster ./scripts/submit_acluster_b5_async_turbo.sh 1,2,3,4,5 1

# B5: data-informed TuRBO-m; four regions, 40 initial points per region,
# 256 truth calls, the same five seeds and eight workers.
SIMSOPT_DFO_CLUSTER=acluster ./scripts/submit_acluster_b5_async_turbo.sh 1,2,3,4,5 4

# B6: existing matched TuRBO control, five seeds and the frozen alpha oracle.
SIMSOPT_DFO_CLUSTER=acluster ./scripts/submit_acluster_b6_matched_turbo.sh 101,102,103,104,105
~~~

The exact B5 configuration is in `scripts/run_b5_async_turbo.py` and
`src/simsopt_dfo/async_turbo.py`: budget `256`, workers `8`, initial points
`2*d`, length `0.8`, minimum `0.5**7`, maximum `1.6`, success tolerance `10`,
failure tolerance `d`, and completion-driven TS. These existing scripts are
controls for the new FortBO runner; their configuration must not be changed
to make a FortBO result look better.

The new runner shall have one command-line contract for both implementations:

~~~
uv run python scripts/run_fortbo_reproduction.py \
  --case landreman-pca-turbo \
  --implementation fortbo \
  --config configs/landreman-pca-turbo.json \
  --seed SEED --workers WORKERS --budget 10000 \
  --scratch $RESULT_ROOT/landreman/fortbo/seed-SEED \
  --output $RESULT_ROOT/landreman/fortbo/seed-SEED.json

uv run python scripts/run_fortbo_reproduction.py \
  --case b5-constellaration \
  --implementation fortbo \
  --config configs/b5-data-informed-turbo.json \
  --seed SEED --workers 8 --budget 256 \
  --scratch $RESULT_ROOT/b5/fortbo/seed-SEED \
  --output $RESULT_ROOT/b5/fortbo/seed-SEED.json
~~~

The script and configs are a work item below, not files that may be silently
assumed to exist today. Before they land, use the existing B5 scripts and a
small synthetic FortBO driver to validate the ABI. The production script must
accept `--implementation botorch` as a control and write the same schema for
both implementations, so comparison code cannot infer method identity from
different ledgers.

For exact Landreman replay the command must additionally accept
`--upstream-script`, `--pca-file`, `--vmec-input`, `--mpi-size`, and
`--acquisition {ei,ts}` and record the source script digest. The FortBO row
uses the same arguments and evaluator, with only the policy implementation
changed. The source's qEI configuration is primary; TS is an explicit second
row.

For the Bindel/Glas case the corresponding command must accept
`--derivative-mode {value-only,exact-derivative,svd-dturbo}` and refuse
`svd-dturbo` until the variational derivative model and FOCUS inputs have
passed the gates. A refusal is preferable to silently labeling the current
Newton-style `fortbo_dturbo` implementation as published DTuRBO.

### 5. What is measured

Every row contains a machine-readable run manifest and candidate ledger. At
minimum record:

- objective, best-so-far, best feasible objective, constraint violation,
  target reached, and time/calls to each fixed checkpoint;
- requested budget, successful calls, failed calls, timeout/retry counts,
  initial calls, gradient calls, derivative components, and out-of-sample
  validation calls;
- wall time, truth/evaluator time, model-fit time, acquisition-search time,
  serialization/ABI time, queue wait, worker busy time, idle-worker time,
  peak and mean concurrency, and rank/thread layout;
- CPU model, GPU model and UUID if allocated, memory high-water mark, power
  or energy if the scheduler exposes it, compiler, MPI, BLAS, Python/Fortran
  runtime versions, package lock or commit IDs, and Slurm job/step IDs;
- exact input digests, objective settings, coordinate chart, seeds, random
  stream state/checkpoints, trust-region trace, model hyperparameters, and
  candidate ordering;
- for DTuRBO, value and gradient work separately, number of derivative rows,
  inducing-point count, variational fit time, posterior predictive error, and
  cost per paired value/gradient observation.

The primary statistical report uses paired seeds and fixed truth-call
checkpoints `32, 64, 128, 256, 512, 1000, 2000, 5000, 10000` for Landreman
and `32, 64, 128, 256` for B5. Bindel stage one additionally reports `200,
1000, 5000, 10000, 50000, 100000` and stage-two combined cost up to `116000`
charged-equivalent calls. Plot median and fixed-seed paired bootstrap 90%
intervals, probability of beating the control, time to target, failure
fraction, and performance profiles. Never rank by wall time without reporting
truth-call and gradient-call work; never rank by objective without reporting
the cost used to obtain it.

Performance claims require the same evaluator and a two-part comparison:

1. **Policy cost:** run the two policies against a frozen candidate/evaluator
   trace and measure fit, acquisition, memory, and proposal throughput with
   physics removed.
2. **End-to-end cost:** run the same physical problem with the same allocation,
   evaluator, failure semantics, and charged budget. Report total wall time,
   useful physics time, and utilization separately.

The claim “FortBO is faster” is accepted only if it wins a predeclared metric
with a paired interval excluding zero, or if it achieves the same objective
target with fewer charged calls and no statistically significant increase in
failure or constraint violation. One favorable seed is not a claim.

### 6. Cluster execution protocol

Use `/home/ert/data/simsopt-dfo-fortbo` for archives, environments, run
directories, and results. Use a commit-addressed detached worktree on a
compute host; never run from a dirty home checkout. Keep source and result
roots distinct so a failed job cannot modify the repository. Every submission
must capture `scontrol show job -dd`, `sacct`, `lscpu`, `nvidia-smi` when a GPU
is allocated, source status, package freeze, linked libraries, and the full
stdout/stderr.

Current toolchain/resource facts from the August 2026 audit:

| Host | Use | Toolchain and resource rule |
| --- | --- | --- |
| aCluster | first smoke and CPU controls; node 34 was the only observed node with an unallocated T4 | CUDA `11.8`/`nvcc`, `gfortran 12.2`, Intel `ifx` under `/opt/intel/oneapi/compiler/`; no usable `nvfortran`/`nvc`; request `--gres=gpu:1` and never assume a GPU on a login node |
| sCluster | preferred GPU campaign when a GPU frees; much more idle CPU capacity, but all observed GPUs were allocated | CUDA `13.1` at `/usr/local/cuda`, also 12.8/12.9; `gfortran 12.2`; no `nvfortran`/`nvc`; request the Blackwell GPU GRES explicitly |
| faepmac1 | SSH proxy/login only | do not benchmark or build a physics result there |
| faepkub4 | source/bootstrap and inspection only | current environment exposes GCC and Intel modules, but no NVIDIA compiler or CUDA compiler |

The current FortBO Fortran build therefore uses `gfortran` or Intel `ifx`; an
NVIDIA Fortran compiler is not available on the audited hosts. GPU speedups
must use the existing CUDA/OpenACC-capable Fortran path and be reported as
GPU runs only after `nvidia-smi`, device placement, and kernel residency are
captured. A CPU-only FortBO policy comparison is valid and should be done
first.

The standard aCluster smoke route is:

~~~
export HOST=acluster
export CHECKOUT=/home/ert/proj/simsopt-dfo-fortbo-checkouts/$(git -C $DFO rev-parse HEAD)
ssh $HOST 'mkdir -p /home/ert/data/simsopt-dfo-fortbo /home/ert/proj'
ssh $HOST "git clone --no-checkout git@gitlab.tugraz.at:D461BDE997455AF1/simsopt-dfo.git /home/ert/proj/simsopt-dfo-fortbo || true"
ssh $HOST "git -C /home/ert/proj/simsopt-dfo-fortbo fetch origin main && git -C /home/ert/proj/simsopt-dfo-fortbo worktree add --detach $CHECKOUT $(git -C \"$DFO\" rev-parse HEAD)"

# CPU policy smoke: no GPU is needed for the first parity gate.
ssh $HOST "cd $CHECKOUT && sbatch --parsable --partition=compute --nodes=1 \
  --ntasks=1 --cpus-per-task=32 --time=04:00:00 \
  --wrap='export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
  uv run python scripts/run_fortbo_reproduction.py --case b5-constellaration \
  --implementation fortbo --config configs/b5-data-informed-turbo.json \
  --seed 1 --workers 8 --budget 32 --scratch /home/ert/data/simsopt-dfo-fortbo/smoke \
  --output /home/ert/data/simsopt-dfo-fortbo/smoke.json'"
~~~

The above is a submission template and becomes executable only after the
runner/config work item lands. For the existing B5 controls, use the checked
in `scripts/submit_acluster_b5_async_turbo.sh` route, which already pins the
source, external ConStellaration commit, environment, result root, and test
set. Do not use an untracked ad-hoc `srun` command for a published result.

The sCluster route is the same, with `SIMSOPT_DFO_CLUSTER=scluster`, an
explicit Blackwell GPU request in a dedicated sbatch file, and a recorded
CUDA device. Use sCluster for full Landreman GPU workers only when the GRES is
actually available; the current Landreman source uses rank-zero CPU model
fitting and one GPU-capable worker per MPI rank, so the allocation must be
designed around the actual worker count rather than inferred from nominal
cluster size. Never request more physics workers than allocated GPUs.

Do not launch a 100,000-call Bindel reproduction until the FOCUS source,
covariance, and input are recovered and a short one-call value/gradient smoke
matches an independent oracle. The full run is a later reservation, not a
placeholder job to occupy the cluster.

### 7. Work packages and acceptance gates

- [ ] **F0: source/config freeze.** Add `configs/landreman-pca-turbo.json`,
  `configs/bindel-dturbo.json`, archive/source digests, and a manifest schema.
  Add a checker that rejects changed physics, bounds, seeds, dimensions,
  evaluator settings, or worker semantics. Gate: two clean source extractions
  produce identical normalized configuration documents.
- [ ] **F1: common runner and independent replay oracle.** Add
  `scripts/run_fortbo_reproduction.py` and a Python reference adapter that
  can run the original BoTorch policy and the FortBO policy with identical
  `ask`/`tell` traces. Gate: deterministic delay and analytic objectives agree
  on every candidate and state row; tests do not merely compare FortBO output
  to itself.
- [ ] **F2: FortBO value-only TuRBO parity.** Implement the Landreman qEI and
  TS configuration, including exact GP fit settings, candidate generation,
  trust-state rules, Sobol behavior, and q=1 completion scheduling. Gate:
  posterior/acquisition/state tolerances and recorded schedule pass before
  any real physics run.
- [ ] **F3: B5 real-physics substitution.** Run five paired seeds for raw and
  data-informed TuRBO-1, then five paired data-informed TuRBO-m controls, at
  the existing 256-call/8-worker budget. Gate: common schema, zero source
  drift, B5 start parity, complete ledgers, and utilization report.
- [ ] **F4: Landreman end-to-end replay.** Run the unmodified source and
  FortBO at the original worker count and at a resource-matched one-GPU smoke.
  Gate: same start/objective/failure semantics, exact initial-design digest,
  and paired metrics at 32 through 10,000 calls. A different MPI size is a
  scaling row, not an exact reproduction.
- [ ] **F5: published DTuRBO model.** Recover FOCUS and the covariance/input
  artifacts, implement or bind the variational derivative GP and paired
  function/gradient action, then compare its posterior and policy trace to an
  independent reference. Gate: value/gradient covariance, trust region,
  Thompson selection, and gradient-cost accounting pass.
- [ ] **F6: Bindel/Glas two-stage run.** Run 5 mm and 10 mm with the exact
  100,000/200/100 global and 2,000/10 local settings, plus local SAA/BFGS.
  Gate: recovered source and seeds, 14-core allocation or a labeled matched
  allocation, out-of-sample stochastic validation, and all 195-coordinate
  ledgers. If inputs remain unavailable, close F6 as blocked/literature-only;
  do not fabricate a reproduction.
- [ ] **F7: performance analysis.** Run policy-only and end-to-end controls at
  identical node/CPU/GPU layouts, repeat at least five paired seeds for each
  stochastic real case, and publish tables/plots under `results/fortbo/`.
  Gate: fixed-checkpoint paired bootstrap, time-to-target, call efficiency,
  fit/acquisition breakdown, concurrency, memory, and energy where available.
- [ ] **F8: release and integration decision.** Add a concise report linking
  all manifests, jobs, ledgers, parity tables, negative results, and blockers.
  Only after F0--F7 pass may the result claim feature parity or superiority.

The existing “external method reproduction” checkboxes further down remain
historical context. These F0--F8 checkboxes are the active FortBO campaign and
must be updated one at a time with a behavioral test, evidence, commit, and
push.
