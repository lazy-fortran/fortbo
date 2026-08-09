# fortbo roadmap

This file tracks work still required for FortBO and its external reproduction
campaign. Completed package milestones are summarized below. Their
implementation details and behavioral tests remain in the repository history
and test suite.

## Current status

As of 2026-08-09, 18:59 CEST:

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

The local ARD lane now retains every usable value and automatically appends
complete derivative observations when a history contains them;
`ignore_gradients` is the explicit value-only escape hatch. Replay adapters
also accept frozen initial designs and region-partitioned candidate pools, with
independent coordinate and trust-state gates. These are local parity
prerequisites, not evidence that the archived Landreman runtime or published
DTuRBO model has been reproduced.

The archived Landreman source contract now checks the trust-state update,
minimum-length restart, Thompson candidate masking, no-replacement sampling,
and qEI trust bounds in addition to the pinned constants. The independent
NumPy replay reference records the same restart transition. A live archived
MPI/physics trace is still required for F2.

FortBO now also has a fixed-hyperparameter inducing variational derivative
posterior reachable through `fortbo_fit_from_history` and the TuRBO driver
configuration. It retains all usable value rows, appends every complete
gradient row, exposes covariance/joint-sample/moment-gradient capabilities,
and is checked against an independent scalar variational oracle. The published
stochastic training procedure and published paired DTuRBO action remain
unchecked; the local ask/tell path now carries paired gradients into this
model when configured.

The external B5 control ledgers are now materialized from Git LFS and pass the
FortBO-side audit: ten TuRBO-1 ledgers (raw and data-informed) plus ten
data-informed four-region TuRBO-m ledgers, all at 256 calls and eight workers.
FortBO's five raw TuRBO-1 rows and two data-informed TuRBO-1 rows are now
recorded and independently audited. The seed-2 data-informed row is the first
one with a complete concurrent BoTorch oracle pair; the remaining three
data-informed TuRBO-1 rows and five data-informed TuRBO-m rows remain pending,
so the control set is complete but the FortBO campaign is not.
At 16:02 CEST, the current `check_fortbo_b5.py` independently re-audited all
five archived raw TuRBO-1 rows and the completed data-informed seed-1 row; all
six passed and no ledger was modified.
At 18:47 CEST, that six-row audit was repeated together with the archived
seed-2 pair check; all checks passed and no ledger was modified.

FortBO's TuRBO driver now accepts an explicit success mask: failed truth calls
remain in the history and trust accounting as imputed worst cases, but are
excluded from surrogate training and cannot become incumbents. The
`fortbo_b5_protocol` and `fortbo_b5_completion_protocol`, together with
`scripts/run_fortbo_b5.py`, provide both the legacy batched and the
completion-driven eight-worker external-evaluator bridges. Completion-driven
asks are one point at a time, keep pending points out of subsequent proposals
with posterior-mean fantasies, accept out-of-order tells, and preserve failed
rows without inventing objective values. Clean eight-call smokes through the
completion-driven bridge recorded eight raw truth failures and eight
successful data-informed truth calls (272.1 seconds), both at peak concurrency
eight with nontrivial completion order; both rows passed the FortBO-side
audit. These are bridge checks, not F3 rows. The first full 256-call
data-informed TuRBO-1 FortBO row completed on `faepkub4` without starting a
duplicate physics run here. Its final ledger is archived at
`/home/ert/data/simsopt-dfo-fortbo/b5-completion-20260809/remote-faepkub4/`
and passes `check_fortbo_b5.py` independently. Seed 1 records 256 truth calls,
68 failed upstream evaluations retained under the failure mask, peak
concurrency eight, and 14102.512474124 seconds wall time. Its pinned sources
are FortBO `cd71e9b9f2f71d0966d3b5c8792f213d78eca1d4`, simsopt-dfo
`2a3ce7b71ea81f659e8910dbd38ea3e99ad9dff4`, and ConStellaration
`112b20ae07193910d467d26033fe51022e641b9f`; the ledger SHA-256 is
`d4cf49c8cb8ac0cb4837c6e346aaff17de949edcf1edaba8219ee1219d7427e1`. This is
one FortBO data-informed TuRBO-1 row, but it is not an oracle-paired row; the
seed 2 below is the first paired row. `/var/tmp/ert` had 84 GB free at the
snapshot.
The pre-existing seed-1 BoTorch control at
`/home/ert/data/simsopt-dfo/b5-turbo-c1845a5/frozen/data-informed-seed-1.json`
does share the transform digest and ConStellaration revision, but it was not
launched by the concurrent pair runner and has no pair manifest; it remains a
standalone comparator rather than a paired F3 oracle.

The concurrent data-informed seed-2 pair then completed on `faepkub4`. The
pair and both child ledgers are archived under
`/home/ert/data/simsopt-dfo-fortbo/b5-oracle-pairs/seed-2-retry3/`; the current
transform-aware `check_oracle_pair.py` passes after deriving FortBO's aggregate
best from its successful evaluation rows. The original control reached best
value `5.602708596031043` with 11 failed evaluations in 9571.621494071005
seconds; FortBO reached `7.252038876891312` with eight failed evaluations in
5169.303276091989 seconds. The FortBO-minus-oracle delta is
`1.6493302808602692`. Both child processes returned zero; the pair SHA-256 is
`a9bc51e8acc41111d2ec24e653e4a85cf905b84debbe2c112c27c75c26118ac2`. The
shared simsopt-dfo revision is
`2a3ce7b71ea81f659e8910dbd38ea3e99ad9dff4`, FortBO is
`9197d89b736535dea905ae0d11d0a80515721149`, and ConStellaration is
`112b20ae07193910d467d26033fe51022e641b9f`. This is the first fully
oracle-paired data-informed TuRBO-1 F3 row.

The public-source provenance lane is now scripted but deliberately not run from
this workstation. `configs/reproduction/source-downloads.json` pins the
Landreman Zenodo record `20733437`, the Glas/Bindel arXiv source, FOCUS
`develop/e4bb49b`, the ConStellaration checkout, and the three B5 Hugging Face
Parquet shards. `scripts/fetch_reproduction_sources.py` reuses an existing
artifact when its environment-variable path is set, verifies SHA-256 digests,
pins Git revisions, rejects unsafe archive paths, and refuses a download or
unpack when the configured free-space reserve would be violated. It does not
copy the existing 7.5 GB Landreman archive merely to make a second mirror.

The same fetch script was run on `faepkub4` at 15:22 CEST for the small
Glas/Bindel and FOCUS artifacts. The arXiv harvest and PDF passed their pinned
SHA-256 digests (`61e1dc89...40a693` and `24cc2600...60f8d0b`), and the FOCUS
checkout is clean at revision
`e4bb49b0632c650e326616912e274feb7781a60d` with tree digest
`f67ad973...1969a7f38`. They remain under
`/var/tmp/ert/fortbo-reproduction-sources/` on the remote host; this is source
acquisition evidence, not an F6 physics reproduction.

The three pinned B5 dataset shards were subsequently staged on `faepkub4` and
passed the manifest size/SHA-256 checks at 15:39 CEST. They remain remote at
`/var/tmp/ert/fortbo-reproduction-sources/constellaration-data/`:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `train-00000-of-00003.parquet` | 251687471 | `e48dada8775f9c86820372cdfe34fd4b181635284bb254854bbd0f09514838c6` |
| `train-00001-of-00003.parquet` | 203988362 | `c574dcbc02f29417a7f5cc088dde6d38f1f165280ab166089a2b1752f3579b9b` |
| `train-00002-of-00003.parquet` | 154525812 | `92e816bdd9898fc7f481040279084dc71ae2130e974a49df0dc2336a693fc084` |

`scripts/run_b5_oracle_pair.py` launches the pinned simsopt-dfo BoTorch control
and `scripts/run_fortbo_b5.py` concurrently with the same B5 mode, seed, budget,
workers, evaluator commit, and coordinate map. The control ledger is recorded
as the independent oracle; `scripts/check_oracle_pair.py` checks both ledgers'
behavioral accounting and compares FortBO's best value against the oracle.
Both Python interpreters are explicit (`--original-python` and
`--fortbo-python`) so the pair cannot silently mix the evaluator and FortBO
environments. The launcher also refuses non-empty run roots and pre-existing
pair outputs, preflights the simsopt-dfo/BoTorch and ConStellaration imports,
checks every recursive relative path dependency declared by FortBO's
`fpm.toml` manifests, and injects the selected simsopt-dfo source into
`PYTHONPATH`, preventing a later seed from overwriting campaign evidence or
failing after workers have already started.
The independent pair checker additionally requires matching B5 case, mode,
dimension, transform file, and transform SHA-256 before reporting a comparison.
The launcher now also marks a pair complete only when both child ledgers exist
and report `passed: true`.
This pair must be launched on `faepkub4` for CPU work or inside an allocated
aCluster/sCluster Slurm job for GPU work. No physics reproduction is to run on
the workstation. The runner stops both children if the run filesystem falls
below its disk reserve.

The completed paired data-informed seed-2 launch ran on `faepkub4`
with the original BoTorch control and FortBO sharing the pinned ConStellaration
evaluator. Earlier seed-2 launch attempts failed before a valid evaluator pair
was started (remote `fo` path, missing BoTorch, and evaluator-environment
selection); each failed pair document is retained and none is counted as F3.
The completed pair used 256 truth calls and eight workers on each side and left
83 GB free on `/var/tmp/ert`. At that checkpoint no next CPU campaign or GPU
Slurm job had started; the workstation was not used for physics work.

The next commit-addressed seed-3 attempt was prepared on `faepkub4` from
FortBO `59c6e35` under `/var/tmp/ert/fortbo-cpu-59c6e35/`, but stopped before
physics. The fresh checkout's `fo build` failed because
`fortnum_kinds.mod` was unavailable; FortBO therefore ended before its first
`ASK`, the paired process returned 1, and the pair is not an F3 row. The failed
manifest is retained at
`/var/tmp/ert/fortbo-cpu-59c6e35/runs/oracle-pairs/data-informed-seed-3.json`.
No retry was launched; the latest remote disk check left 82 GB free, and the
workstation remained unused for physics. The failed attempt left the clean
dependency/build context as the next campaign gate.

That build context is now repaired on `faepkub4`: the FortBO `59c6e35`
checkout has clean detached sibling worktrees for its path dependencies, with
FortNum `7ced2f7aa272`, FortOpt `883aa7ee0a3f`, FortAD `d71cdf724cd8`, FortML
`970508656825`, FortMC `e5e42a0ac1d4`, FortFront `488c49e14700`, FortGen
`ea422bb282ba`, and FortSparse `7ad20738b3f8`. Remote `fo build` completed
818/818, and `FO_TEST_TIMEOUT=60 fo test` passed all 48 tests in 81.1 seconds.
The default 10-second test timeout still reports 45/48 because the three slow
tests exceed that harness limit; this is why the explicit timeout is recorded.
This was a software-only gate, and the workstation was not used.

After that gate, the data-informed TuRBO-1 seed-3 retry-1 pair was launched on
`faepkub4` with the pinned simsopt-dfo/ConStellaration environments, 256 calls,
and eight workers. Its FortBO checkout is `59c6e35`; its fresh run root is
`/var/tmp/ert/fortbo-cpu-59c6e35/runs/oracle-pair-data-informed-seed-3-retry1/`
and its pair output is
`/var/tmp/ert/fortbo-cpu-59c6e35/runs/oracle-pairs/data-informed-seed-3-retry1.json`.
The launcher was alive after startup with no ledger yet and 82 GB free on
`/var/tmp/ert`. At the 18:59 CEST checkpoint, the parent and both child
evaluator processes were still alive after 01:38:58. The oracle scratch root
had 120 dispatched requests and 103 response files; the FortBO root had 123
requests and 85 responses, with the pair ledger still absent. This is an
in-progress attempt, not an F3 row. The workstation remains unused for
physics.

The Landreman exact-tool path is now portable: the manifest resolves
`LANDREMAN_ARCHIVE`, `scripts/run_landreman_original.py` extracts only the
archived `software/alpha_opt` tree, verifies the archive and pinned member
digests, records the one `/pscratch` PCA remap, and emits the historical
5-rank/4-GPU `srun` command. Preparation is safe outside an allocation;
`--execute` refuses to run without `SLURM_JOB_ID`. This is replay plumbing and
source-level evidence, not yet a live Landreman control/FortBO result. The
`slurm/landreman_exact_replay.sbatch` wrapper now captures Slurm, CPU, GPU,
compiler/runtime, and FortBO commit metadata before downloading/reusing the
archive and executing that control on the allocation.

The exact-tool preflight was completed on `faepkub4` at 15:00 CEST. The
7,529,635,840-byte Landreman archive was downloaded by
`fetch_reproduction_sources.py` and passed its pinned SHA-256
(`7037bb0a...20c428f`). `run_landreman_original.py` extracted 26,508,541 bytes
into a new run root and recorded the portable PCA remap in
`replay.json`; the execution-manifest and archived-driver contract checks both
passed. The archived job confirms one node, `regular`, `gpu&hbm40g`, account
`m4505`, five ranks, 13 CPUs per task, and four GPUs. The preparation evidence
is retained under
`/home/ert/data/simsopt-dfo-fortbo/landreman-exact-replay/prepared-630ae10/`;
the large archive remains on `faepkub4` rather than being copied to the
workstation. The current `aCluster` and `sCluster` inventories expose one GPU
per node, so no exact one-node/four-GPU allocation was submitted; the live
control/FortBO trace remains open.
The read-only 18:44 CEST Slurm inventory still reports one `gpu:tesla_t4:1`
per visible aCluster node and one `gpu:nvidia_rtx_pro_6000...:1` on the idle
sCluster node; no compatible four-GPU node was visible and no GPU job was
submitted by this campaign.

The missing FortBO side of F4 is now staged in
`scripts/run_landreman_fortbo.py`: rank zero owns the completion-driven
ask/tell protocol and four MPI worker ranks each own an independent archived
VMEC/PCA objective. `slurm/landreman_fortbo.sbatch` preserves the historical
one-node/four-GPU shape, captures runtime metadata, refuses an existing output,
and performs source preparation/check-only gates before launching.
`scripts/check_landreman_fortbo.py` independently audits candidate IDs, unit
box bounds, statuses, completion order, counts, pinned revisions, and the
best-value reduction. The bridge/checker and wrapper checks pass (13 focused
tests, Ruff, and shell syntax); no physics run has been claimed for this
bridge yet.

At `bf7c665`, a clean canonical checkout with clean sibling dependencies passes
all 48 Fortran tests and all 40 Python tests. The current pushed checkout passes
all 48 Fortran tests and all 64 Python tests. A raw 256-call attempt was stopped
after 71 completed failures because several upstream evaluator subprocesses
ran for multiple minutes; it produced no ledger and is not counted as an F3
row. The external evaluator has an explicit 600-second worker timeout policy.
The latest simsopt-dfo commit `2a3ce7b` starts each worker in its own process
group and kills that group on timeout, so descendant physics processes cannot
keep a worker future or pipe alive. Its focused B5 contract suite passes 7
tests and Ruff passes. The completed remote row below used the earlier
`a9b6e2b` source; the process-group fix was not yet present.

The first bounded local retry was stopped at 243 dispatched requests, before
writing a ledger, to keep the workstation cool; its scratch is retained but is
not evidence. Future CPU campaigns are staged under `/var/tmp/ert` on
`faepkub4`. GPU campaigns use Slurm on `aCluster` or `sCluster`, with the
NVIDIA compiler/environment selected inside the allocation; no long physics
campaign runs on the workstation.

The bridge was refreshed against those current commits: the raw completion
smoke recorded eight failures in 15.38 seconds, and the data-informed smoke
recorded eight successes in 223.63 seconds. Both rows reached peak concurrency
eight, had nontrivial completion order, and passed `check_fortbo_b5.py`; they
remain bridge evidence rather than 256-call F3 rows.

The CPU lane was then reproduced on `faepkub4` from the commit-addressed
`/var/tmp/ert/fortbo-cpu-3c3bc26` workspace: its raw eight-call row recorded
eight failures in 43.52 seconds at peak concurrency eight and passed the same
audit. The remote Fortran gate built successfully and passed the ARD and driver
tests; two unrelated 10-second test-harness timeouts occurred in the full
48-test host run.

The first complete FortBO F3 row was then run on `faepkub4`, not the
workstation: raw TuRBO-1, seed 1, 256 calls, eight workers, and completion-
driven ask/tell. The ledger at
`/home/ert/data/simsopt-dfo-fortbo/b5-completion-20260809/remote-faepkub4/`
passes `check_fortbo_b5.py` and records 37 successful and 219 failed truth
calls, including 46 structured 600-second timeouts, peak concurrency eight,
nontrivial completion order, and 5449.87 seconds wall time. Its pinned source
commits are FortBO `3c3bc26`, simsopt-dfo `a9b6e2b`, and ConStellaration
`112b20a`. This is one of the five required raw TuRBO-1 rows; the remaining
F3 FortBO rows are still open.

The second complete raw TuRBO-1 row used the current pushed sources on the
same host: seed 2, 256 calls, eight workers, and FortBO `8a51512` with
simsopt-dfo `2a3ce7b`. Its ledger also passes `check_fortbo_b5.py` and records
30 successful and 226 failed truth calls, including 39 structured worker
timeouts, peak concurrency eight, all 256 unique completion IDs, and
4580.55 seconds wall time. Two of the five raw TuRBO-1 FortBO rows are now
recorded; the other F3 policy/seed rows remain open.

The third complete raw TuRBO-1 row used FortBO `69e67e4` and simsopt-dfo
`2a3ce7b`: seed 3, 256 calls, eight workers, and the same pinned
ConStellaration source. Its ledger passes `check_fortbo_b5.py` and records 46
successful and 210 failed truth calls, including 25 structured worker
timeouts, peak concurrency eight, all 256 unique completion IDs, and
3490.09 seconds wall time. Three of the five raw TuRBO-1 FortBO rows are now
recorded; the remaining raw seeds and data-informed rows are open.

The fourth complete raw TuRBO-1 row used FortBO `c263e46` and simsopt-dfo
`2a3ce7b`: seed 4, 256 calls, eight workers, and the same pinned
ConStellaration source. Its ledger passes `check_fortbo_b5.py` and records 26
successful and 230 failed truth calls, including 25 structured worker
timeouts, peak concurrency eight, all 256 unique completion IDs, and
3528.25 seconds wall time. Four of the five raw TuRBO-1 FortBO rows are now
recorded; the final raw seed and data-informed rows remain open.

The fifth complete raw TuRBO-1 row used FortBO `0fc3dd2` and simsopt-dfo
`2a3ce7b`: seed 5, 256 calls, eight workers, and the same pinned
ConStellaration source. Its ledger passes `check_fortbo_b5.py` and records 29
successful and 227 failed truth calls, including 40 structured worker
timeouts, peak concurrency eight, all 256 unique completion IDs, and
4904.78 seconds wall time. All five raw TuRBO-1 FortBO rows are now recorded;
the data-informed rows remain open.

The FOCUS source was recovered from its public Git repository and pinned at
`e4bb49b0632c650e326616912e274feb7781a60d`, with the stochastic source and
W7-X high-mirror example files present. An isolated GNU Fortran/OpenMPI/HDF5
build produced `xfocus`; a bounded W7-X initialization reached surface
initialization but then failed in an upstream GNU Fortran format string. This
recovers source/build evidence, not the paper's exact W7-X input, the frozen
covariance/realization ledger, optimizer implementation, or seed ledger. The
published periodic-kernel equations are now implemented in
`scripts/glas_covariance.py` and checked against direct numerical integration;
that generator does not claim to recover the paper's random-number schedule.

The current local GPU audit on `mailuefterl` found `nvfortran`/`nvc` 26.5-0 and
two NVIDIA GeForce RTX 5060 Ti devices (driver 610.43.03, 16311 MiB each).
`src/fortbo_device.F90` now keeps the score and pooled TuRBO realization arrays
on the device through the reduction, returning only the selected scalar state.
A standalone probe compiled with `nvfortran -acc -O2 -Minfo=accel` passed on
both devices; `PGI_ACC_TIME=1` reported NVIDIA kernels and reduction kernels
with no per-candidate array copyout. This is kernel/residency evidence, not a
package benchmark: `FC=nvfortran fo build --flag "-acc"` currently stops in the
external FortAD `fortad_reverse.f90` with an nvfortran internal compiler error
(`Deferred-length character symbol must have descriptor`, line 5166).

## Active blockers

1. The Landreman archive and its historical Slurm job are now pinned in the
   manifest, including the 5-rank allocation (one manager plus four workers)
   and four GPUs. The archived driver still names an absolute `/pscratch` data
   path; `run_landreman_original.py` applies the declared remap in a generated
   copy and refuses non-Slurm execution. The source and driver preflight now
   pass, but the archived control/FortBO pair has not yet been launched: the
   the FortBO-side MPI bridge is ready, but current aCluster and sCluster
   inventories provide one GPU per node while
   the historical job requires four GPUs on one node. A compatible Tu Graz
   allocation is therefore still required for the live MPI/physics trace
   needed for F2.
2. The Glas et al. harvest contains manuscript and figures, but no paper-
   specific W7-X input, covariance/realization ledger, or optimizer ledger.
   FOCUS source is now pinned separately and builds in isolation; the
   published periodic covariance formula is implemented locally, but its
   bundled high-mirror example is not evidence of the paper's exact inputs,
   and its GNU Fortran initialization smoke currently hits an upstream
   format-string failure. The reproduction remains literature-only until the
   remaining artifacts and runtime parity are recovered.
3. Published DTuRBO uses an inducing-point stochastic variational GP with
   paired function and gradient observations. FortBO now has a checked local
   fixed-hyperparameter inducing derivative posterior, plus an exact dense
   derivative GP and Newton-BO-style derivative trust-region policy. The
   published stochastic training and paired action are still unchecked; until
   those gates pass, label the result `fortbo-exact-derivative` or
   `fortbo-variational-derivative` as appropriate.
4. The current audit host has nvfortran/nvc and two visible RTX 5060 Ti
   devices, and the standalone FortBO device probe now has captured identity
   and kernel-residency evidence. A whole-package OpenACC build is still
   blocked by the external FortAD nvfortran compiler error recorded above, so
   the package-level GPU benchmark and end-to-end performance profile remain
   open.

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
| Landreman Ax environment list | 99ea0eb5f3d920804cdcad0505485a6d6b213b259bb74f211ff9ef765b9af228 |
| Landreman PCA data inside the archive | 745548e503beda2f8794b169b8a8abd55adeddfa4acc71c2a76045b61acaac7c |
| Landreman VMEC input inside the archive | 88318d8b2ab17741110a11bc5141ecfbbd862eb5ff02b47f808bc527c6bf263e |
| Glas/Bindel paper | 24cc2600e8b20b74b80b96ce294286c66bea19c2e96808e516abae5f960f8d0b |
| Glas/Bindel harvest | 61e1dc8912ddb4825b6ac5ad5d26c2a0d86280fb71d86f2ef3991dfb5c40a693 |
| FOCUS source tree at develop/e4bb49b | de28fd32e76c4deb5baddee2369696c6a085c97ed54d7343aa27c809fdd1ad97 |
| B5 TuRBO-1 control comparison | f22030f45d8c6e98e247a6f63a5af0bc812d18cb3ee4205462376f55c170c4b9 |
| B5 TuRBO-m control comparison | 4d6f2df3785f350a8b87fb1cfed443225109e1c657b1b9b139bffc65768d1dcd |
| B5 data-informed transform JSON | 951c2b6f8e0f8dd1dee0297aca91645900ce4044f104e3f5132fbad523e68340 |
| B5 data-informed transform arrays | 23efce87449de554ff8a0c9025cdf44e301a14f48c511fb907f6fdeb6fce7953 |

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
export LANDREMAN_ARCHIVE=/var/tmp/ert/reproduction-sources/landreman/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar
python scripts/fetch_reproduction_sources.py --artifact landreman_archive --check-only
python scripts/run_landreman_original.py \
  --run-root /var/tmp/ert/landreman-reproduction/20260809-exact
~~~

Inside the allocated 5-rank/4-GPU Slurm job, repeat the preparation command
with `--execute`. The launcher performs the declared `/pscratch` PCA remap and
uses the archived driver with `srun --ntasks=5 --cpus-per-task=13`; it does not
silently substitute a different MPI size or a local workstation path.

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

The paper-level command remains non-executable because the paper-specific
W7-X input, frozen covariance/realization convention, optimizer source, and
seed ledger are still missing. The recovered FOCUS checkout is a source/build
reference, and `scripts/glas_covariance.py` implements the published
single-period covariance formula; neither is the missing experiment ledger:

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
| mailuefterl | local OpenACC kernel probe | nvfortran/nvc 26.5-0, two RTX 5060 Ti devices; whole-package build currently stops in external FortAD |
| aCluster | first CPU smoke and available-T4 tests | CUDA 11.8, gfortran 12.2, Intel ifx, no nvfortran/nvc, request --gres=gpu:1 |
| sCluster | preferred GPU run when GRES is free | CUDA 13.1 at /usr/local/cuda, gfortran 12.2, all audited GPUs were allocated |
| faepmac1 | SSH proxy/login | no benchmark runs |
| faepkub4 | inspection/bootstrap | no current CUDA or NVIDIA compiler |

Start with a CPU smoke. Use sCluster only after a GPU allocation is confirmed.
Never request more physics workers than allocated GPUs. A GPU result requires
device identity and kernel-residency evidence.

## Delivery plan

- [x] F0: add versioned Landreman and Glas JSON configs, source digests, and
  a manifest checker.
- [x] F1: add scripts/run_fortbo_reproduction.py with implementation
  botorch|fortbo, identical output schema, and an independent analytic/delay
  replay oracle.
- [x] F1.5: add public source pins, checksum/revision-aware downloads with a
  free-space guard, and the concurrent B5 original-control/oracle pair
  launcher. The scripts are cluster-portable; no workstation physics run is
  counted.
- [ ] F2: match Landreman qEI and TS posterior, acquisition, candidate,
  trust-state, and completion traces.
- [ ] F3: run five paired seeds for raw/data-informed TuRBO-1 and
  data-informed TuRBO-m at 256 calls and eight workers; control ledgers are
  audited, all five raw TuRBO-1 FortBO rows and two data-informed TuRBO-1
  FortBO rows are now recorded, and seed 2 is the first complete oracle-paired
  data-informed row. The remaining data-informed rows are pending; seed 1 is
  still a standalone comparator.
- [ ] F4: run archived Landreman control and FortBO at the original allocation,
  then a labeled resource-matched GPU scaling row. The exact archive, source
  digest, portable preparation, contract checks, and FortBO MPI bridge pass;
  launch only on the Tu Graz allocation once four GPUs are available on one
  node, and retain the archived control as the independent oracle.
- [ ] F5: recover FOCUS artifacts and implement/check the inducing variational
  derivative model and paired action. The local fixed-hyperparameter model,
  adapter/driver path, independent oracle, and pinned/build-checked FOCUS
  source are shipped; the paper-specific inputs, stochastic training parity,
  and published paired-action gate remain open.
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
