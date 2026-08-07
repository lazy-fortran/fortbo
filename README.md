# fortbo

Differentiable Bayesian optimization for modern Fortran.

`fortbo` owns sequential design and acquisition optimization. FortML supplies
the surrogate/model and posterior contracts, FortOpt solves local constrained
acquisition problems, FortNum supplies arrays, RNG, and linear algebra,
FortAD supplies differentiable acquisition products, and FortSym generates or
proves compact fixed kernels. FortMC is an optional companion for fully
Bayesian surrogate sampling and Thompson-style policies.

The package is intentionally model-agnostic at the acquisition boundary: a
surrogate provides a posterior over queried points and, when gradient-based
optimization is requested, differentiable reparameterized samples or analytic
posterior derivatives.

The pre-1.0 API may change without compatibility shims. See
[`ROADMAP.md`](ROADMAP.md) for the implementation, GPU, and benchmark plan.
