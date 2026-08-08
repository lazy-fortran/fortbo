# fortbo roadmap

This file tracks work still required for FortBO and its external reproduction
campaign. Completed package milestones are summarized below. Their
implementation details and behavioral tests remain in the repository history
and test suite.

## Current status

As of 2026-08-08:

| Area | Status |
| --- | --- |
| Acquisition catalog, batch policies, constraints, cost, risk, and multi-fidelity APIs | shipped | <!-- slop-ok: compact capability inventory -->
| Exact, derivative-observation, sparse/variational, multi-task, robust, and custom posterior adapters | shipped |
| Search spaces, TuRBO-1/TuRBO-m, Newton-style derivative mode, ask/tell workers, retries, tracing, and persistence | shipped | <!-- slop-ok: compact capability inventory -->
| CPU/OpenACC kernels, provenance lanes, refusals, synthetic fixtures, and generic cross-framework benchmarks | shipped |
| Landreman TuRBO reproduction with FortBO | open |
| Published Glas/Bindel DTuRBO reproduction with FortBO | blocked on upstream inputs and model parity |
| Public performance claim against BoTorch and DTuRBO | open |

Completed generic evidence is in /mnt/storage/code/lazy-fortran/fortbo-bench.
The physics campaign belongs in /mnt/storage/code/simsopt-dfo. Large inputs,
environments, scratch directories, and result ledgers belong in
/home/ert/data/simsopt-dfo-fortbo.

## Active blockers

1. The Landreman archive contains the driver, VMEC input, and PCA data, but the
   original absolute run path and MPI allocation are not available as a
   complete manifest. A portable replay must record those deviations.
2. The Glas et al. harvest contains manuscript and figures, but no FOCUS
   source, W7-X input, perturbation covariance, or optimizer ledger. Its
   reproduction is literature-only until those artifacts are recovered.
3. Published DTuRBO uses an inducing-point stochastic variational GP with
   paired function and gradient observations. FortBO currently has an exact
   dense derivative GP and a Newton-BO-style derivative trust-region policy.
   Until the published model is implemented and checked, label the result
   fortbo-exact-derivative.
4. The audited hosts expose gfortran and Intel ifx, but no nvfortran or nvc.
   GPU results require an allocated device, captured identity, and kernel
   residency evidence.

## Repository and provenance

The campaign checkout is /mnt/storage/code/simsopt-dfo with GitLab remote
git@gitlab.tugraz.at:D461BDE997455AF1/simsopt-dfo.git. Authenticated search
found no matching project under plasma/proj or plasma/proj/stel, so this
owner-namespace project is reused.

FortBO is /mnt/storage/code/lazy-fortran/fortbo. Do not put archives,
virtual environments, scratch directories, or generated ledgers in Git.

Frozen artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Landreman Zenodo archive | 7037bb0abbaaa7ccc4bc7b9f5434e41b18ecdf97af04cf8ae244ea2ae20c428f |
| Landreman PCA data inside the archive | 745548e503beda2f8794b169b8a8abd55adeddfa4acc71c2a76045b61acaac7c |
| Landreman VMEC input inside the archive | 88318d8b2ab17741110a11bc5141ecfbbd862eb5ff02b47f808bc527c6bf263e |
| Glas/Bindel paper | 24cc2600e8b20b74b80b96ce294286c66bea19c2e96808e516abae5f960f8d0b |
| Glas/Bindel harvest | 61e1dc8912ddb4825b6ac5ad5d26c2a0d86280fb71d86f2ef3991dfb5c40a693 |

~~~
sha256sum /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar /home/ert/proj/stellopt-talk/literature/glas2022_coil-dturbo.pdf /home/ert/data/simsopt-dfo-harvest/coil-dturbo-paper-2110.07464/2110.07464.tar
tar -xOf /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar 20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt/data/20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5 | sha256sum
tar -xOf /home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar 20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt/data/input.vmec | sha256sum
~~~

## Exact configurations

### Landreman data-informed TuRBO

Source: software/alpha_opt/scripts/driver_turbo_PCA_unconstrained.py in the
pinned Zenodo archive.

Preserve:

- SurfaceWeightedPCA, dimension 20, unit bounds [0,1]^20, and the archived
  PCA file without refitting or reordering.
- Aspect ratio 6.0, max_B_target 12.0, one max-B iteration, vacuum off.
- n_particles 25000, t_max 1e-1, tau 0.1, maxloss 0.02, t_block 1e-3,
  tol 1e-6, min_dt 1e-9.
- VMEC failure value 5.5, DMerc failure value -0.5, and failures treated as
  values.
- 10,000 evaluations, batch size 1, three-hour wall limit with five minutes
  reserved, save frequency 1.
- Rank zero fits the CPU GP and MPI workers evaluate asynchronously. The
  initial design is scrambled Sobol with seed 0 and
  max(2*dim, 2*(size-1)) points.
- Trust state length 0.8, minimum 0.5**7, maximum 1.6, success tolerance 10,
  failure tolerance ceil(max(4/batch_size, dim/batch_size)).
- Candidate count min(5000, max(2000, 200*dim)), which is 2000 at d=20.
  TS mask probability is min(20/d,1), with at least one changed coordinate.
- Primary acquisition is qEI with q=1, num_restarts=10, raw_samples=512.
  TS is a separate ablation.
- GP is BoTorch SingleTaskGP with Matern nu=2.5 ARD, Gaussian noise interval
  [1e-8,1e-3], lengthscale interval [0.005,4.0], standardized outputs,
  exact marginal likelihood, and Cholesky inference.

~~~
export LANDREMAN_ARCHIVE=/home/ert/data/landreman-data-informed-2026/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar
export LANDREMAN_RUN=/home/ert/data/landreman-reproduction/20260514-01
mkdir -p $LANDREMAN_RUN
tar -xf $LANDREMAN_ARCHIVE -C $LANDREMAN_RUN
export LANDREMAN_ROOT=$LANDREMAN_RUN/20260514-01-zenodo_for_data_informed_spaces_paper/software/alpha_opt
sed -n '1,760p' $LANDREMAN_ROOT/scripts/driver_turbo_PCA_unconstrained.py
~~~

After establishing the original MPI size:

~~~
cd $LANDREMAN_ROOT
OMP_NUM_THREADS=12 mpiexec -n ORIGINAL_MPI_SIZE python scripts/driver_turbo_PCA_unconstrained.py
~~~

A path remapping or changed MPI size is a portability or scaling row, not an
exact replay.

### Glas/Bindel DTuRBO and AdamCV

Preserve the W7-X-like setup:

- five modular coils per period, five field periods, 50 physical coils;
- NF=6, N=195, Nseg=64, Ntheta=Nzeta=64;
- periodic GP/Fourier perturbations at p=5 mm and p=10 mm;
- omega_B=100, omega_L=0.5, target length 8.0, separation 0.23 m,
  quadratic penalty lambda=100, alpha-quasimax 10000;
- bounds centered at circular coils of radius 1.5 m and scaled by perturbation
  variance;
- stage one: 200 initial calls, batch 100, maximum 100,000 calls;
- AdamCV stage two: 2,000 iterations, batch 10, eta=0.001, gamma=0.01,
  beta1=beta2=0.95, epsilon=1e-10;
- local control: 50,000 gradient calls, ten gradients per step, eta=0.04,
  gamma=0.1, beta1=beta2=0.95, epsilon=1e-10.

The paper-level command becomes executable after the missing FOCUS artifacts
are recovered:

~~~
cd /home/ert/data/simsopt-dfo-harvest/coil-dturbo-paper-2110.07464/recovered-focus
mpiexec -n 14 python run_dturbo_adamcv.py --nf 6 --nseg 64 --ntheta 64 --nzeta 64 --coils-per-period 5 --field-periods 5 --dimension 195 --omega-b 100 --omega-l 0.5 --length-target 8.0 --separation 0.23 --lambda-penalty 100 --alpha-quasimax 10000 --perturbation-mm 5 --global-budget 100000 --global-batch 100 --global-initial 200 --local-iterations 2000 --local-batch 10 --eta 0.001 --gamma 0.01 --beta1 0.95 --beta2 0.95 --epsilon 1e-10 --seed SEED --output results/bindel/dturbo-5mm-SEED.json
~~~

Repeat at 10 mm and run the SAA/BFGS local control. The seed must come from
the recovered source or experiment ledger.

## Parity protocol

FortBO replaces only proposal generation and optimizer state. The evaluator,
coordinate map, bounds, resolution, tolerances, failure semantics, random
numbers, and charged-call accounting stay fixed.

1. Coordinate and ABI parity: round-trip the 20D PCA chart and 195D coil
   chart against independent Python/NumPy code. Test bounds, ordering, <!-- slop-ok: parity dimensions -->
   clipping, and chain rules to 1e-13 in float64.
2. Randomness and schedule: match Sobol seed/scramble, candidate masks, TS
   draw order, region assignment, completion order, and ask/tell state. Use
   a frozen candidate file when cross-language Sobol bits cannot match.
3. Posterior parity: compare mean, variance, derivative blocks, and sampled
   paths with BoTorch/GPyTorch on frozen training data. Record jitter,
   tolerances, Cholesky policy, and hyperparameters.
4. Acquisition parity: compare qEI and TS separately, including qEI
   restarts/raw samples, TS joint draws, candidate pooling, and
   no-replacement selection.
5. Trust-state parity: replay an answer stream and compare radius,
   success/failure counters, thresholds, expansion, shrinkage, restarts, and <!-- slop-ok: compact parity inventory -->
   multi-region placement.
6. Failure and restart parity: exercise timeout, process loss, retry, solver
   failure, non-finite value, cancellation, and checkpoint/resume. Keep
   failures distinct from values and charge retries separately.
7. DTuRBO parity: validate the inducing variational derivative GP and paired
   value/gradient actions before calling a FortBO row published DTuRBO.
8. Independent oracle: every gate uses a mathematical, NumPy, BoTorch, or
   delay-injected evaluator oracle that does not consume FortBO output as its
   expected answer.

## Benchmark matrix

| Case | Control | FortBO rows | Budget |
| --- | --- | --- | --- |
| Landreman PCA | archived BoTorch qEI and TS ablation | value-only qEI and TS | 10,000 |
| B5 ConStellaration | existing BoTorch TuRBO-1 and TuRBO-m | value-only FortBO TuRBO-1 and TuRBO-m | 256 |
| Glas/Bindel 5 mm and 10 mm | recovered DTuRBO and AdamCV | published-parity DTuRBO, or exact-derivative until parity | 100,000 + local |
| Synthetic fixtures | BoTorch/GPyTorch and deterministic NumPy | every FortBO policy lane | 32 to 512 |

Existing B5 controls:

~~~
cd /mnt/storage/code/simsopt-dfo
uv sync --frozen --extra bayesopt --extra mhd --inexact
SIMSOPT_DFO_CLUSTER=acluster ./scripts/submit_acluster_b5_async_turbo.sh 1,2,3,4,5 1
SIMSOPT_DFO_CLUSTER=acluster ./scripts/submit_acluster_b5_async_turbo.sh 1,2,3,4,5 4
~~~

Every run records objective and feasibility checkpoints, successful/failed/
timeout/retry/initial/value/gradient calls, wall/evaluator/model/acquisition/ABI/
queue/busy/idle time, CPU/GPU identity, memory, energy when available,
compiler/MPI/BLAS/package versions, Slurm IDs, source status, input digests,
trust trace, candidate order, random state, model parameters, derivative
rows, inducing count, and out-of-sample validation.

Use checkpoints 32,64,128,256,512,1000,2000,5000,10000 for Landreman,
32,64,128,256 for B5, and 200,1000,5000,10000,50000,100000 for Glas/Bindel.
Report paired-seed medians, fixed-seed bootstrap 90% intervals, probability of
beating the control, time to target, failure rate, and performance profiles.

A performance claim has two rows:

1. Policy cost on a frozen evaluator trace, measuring fit, acquisition, memory,
   and proposal throughput.
2. End-to-end cost with identical physics, allocation, failure policy, and
   charged budget.

Faster requires a predeclared metric with a paired interval excluding zero, or
the same target with fewer charged calls and no significant failure or
constraint penalty.

## Cluster protocol

Use commit-addressed detached checkouts on compute hosts. Keep results in
/home/ert/data/simsopt-dfo-fortbo. Capture scontrol show job -dd, sacct, lscpu,
nvidia-smi for GPU jobs, package freezes, linked libraries, stdout, and stderr.

| Host | Role | Constraint |
| --- | --- | --- |
| aCluster | first CPU smoke and available-T4 tests | CUDA 11.8, gfortran 12.2, Intel ifx, no nvfortran/nvc, request --gres=gpu:1 |
| sCluster | preferred GPU run when GRES is free | CUDA 13.1 at /usr/local/cuda, gfortran 12.2, all audited GPUs were allocated |
| faepmac1 | SSH proxy/login | no benchmark runs |
| faepkub4 | inspection/bootstrap | no current CUDA or NVIDIA compiler |

Start with a CPU smoke. Use sCluster only after a GPU allocation is confirmed.
Never request more physics workers than allocated GPUs. A GPU result requires
device identity and kernel-residency evidence.

## Delivery plan

- [ ] F0: add versioned Landreman and Glas JSON configs, source digests, and
  a manifest checker.
- [ ] F1: add scripts/run_fortbo_reproduction.py with implementation
  botorch|fortbo, identical output schema, and an independent analytic/delay
  replay oracle.
- [ ] F2: match Landreman qEI and TS posterior, acquisition, candidate,
  trust-state, and completion traces.
- [ ] F3: run five paired seeds for raw/data-informed TuRBO-1 and
  data-informed TuRBO-m at 256 calls and eight workers.
- [ ] F4: run archived Landreman control and FortBO at the original allocation,
  then a labeled resource-matched GPU scaling row.
- [ ] F5: recover FOCUS artifacts and implement/check the inducing variational
  derivative model and paired action.
- [ ] F6: run both Glas/Bindel perturbation amplitudes, global/local stages,
  SAA/BFGS control, and out-of-sample validation.
- [ ] F7: publish ledgers, parity tables, utilization, cost breakdowns,
  negative results, and exact source/runtime manifests.

## Completion rule

Complete the campaign only when F0--F7 have pushed evidence, every parity gate
passes, missing inputs are recovered or explicitly closed as blocked, and the
final report separates archived control, FortBO value-only TuRBO, FortBO
exact-derivative mode, published-parity DTuRBO, policy-only cost, end-to-end
cost, CPU, transfer-inclusive GPU, resident GPU, and unavailable/refused lanes.

## References

- Eriksson et al., “Scalable global optimization via local Bayesian <!-- slop-ok: paper title -->
  optimization,” NeurIPS 2019, https://arxiv.org/abs/1910.01739.
- Glas, Padidar, Kellison, and Bindel, “Efficient stochastic optimization of <!-- slop-ok: author list -->
  stellarator coils,” https://arxiv.org/abs/2110.07464.
- Wu, Poloczek, Wilson, and Frazier, “Bayesian optimization with gradients,” <!-- slop-ok: author list -->
  NeurIPS 2017.
- The BoTorch TuRBO tutorial and the pinned upstream scripts above.
