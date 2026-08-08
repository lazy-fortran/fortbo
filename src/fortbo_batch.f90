module fortbo_batch
    !! Batch acquisitions: qEI, qNEI, qUCB (ROADMAP BO1).
    !!
    !! A batch acquisition scores a *set* of `q` points evaluated together, and
    !! the only thing that makes it different from `q` separate acquisitions is
    !! that the points are scored under one **joint** posterior draw:
    !!
    !!     qEI(X)  = E[ max_i max(best - f_i, 0) ]
    !!     qUCB(X) = E[ max_i (mu_i + beta |f_i - mu_i|) ]
    !!
    !! with `f` drawn jointly, so correlated points share their randomness. That
    !! sharing is the whole mechanism: two nearby candidates rise and fall
    !! together, so the second adds almost nothing to the maximum, and the
    !! estimator penalizes the redundancy without any explicit diversity term.
    !! Scoring the same batch under independent marginals would rate `q` copies
    !! of one point as `q` times as good as one copy, which is exactly backwards.
    !!
    !! The `max` inside the expectation is what makes these batch quantities and
    !! also what makes them non-separable: `E[max]` is not `max E`, so a batch
    !! acquisition cannot be assembled from per-point values however they are
    !! combined. That is why these live here rather than as a reduction over the
    !! sequential acquisitions.
    !!
    !! Samples are drawn once and frozen, as elsewhere in FortBO, so a batch
    !! comparison is deterministic under replay and two candidate batches are
    !! compared against the same realizations rather than against separate
    !! noise.
    !!
    !! qKG is not here. It needs the fantasy machinery over `q` simultaneous
    !! observations, which is a different construction from these three, and
    !! folding it in would have hidden that difference behind a shared name.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_JOINT_SAMPLE
    implicit none
    private

    public :: fortbo_batch_samples_t
    public :: fortbo_qei
    public :: fortbo_qnei
    public :: fortbo_qucb

    !! Frozen joint draws over one batch, shaped `(point, sample)`.
    type :: fortbo_batch_samples_t
        real(dp), allocatable :: draws(:, :)
    contains
        procedure, public :: generate => samples_generate
        procedure, public :: n_points => samples_n_points
        procedure, public :: n_samples => samples_n_samples
    end type fortbo_batch_samples_t

contains

    !! Draw and freeze `n_samples` joint realizations of the posterior over
    !! `points`. The posterior must support joint sampling; marginal moments are
    !! not enough, and a batch acquisition built on them would be wrong in the
    !! specific way that matters — it would reward duplicates.
    subroutine samples_generate(self, posterior, points, n_samples, generator, &
            status)
        class(fortbo_batch_samples_t), intent(out) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: n_samples
        type(rng_t), intent(inout) :: generator
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: block(:, :)
        integer :: q, s

        q = size(points, 1)
        if (q < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: a batch needs at least one point")
            return
        end if
        if (n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: sample count must be positive")
            return
        end if
        if (.not. posterior%supports(FORTBO_CAP_JOINT_SAMPLE)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo batch: posterior cannot draw joint samples")
            return
        end if

        allocate (self%draws(q, n_samples))
        allocate (block(q, 1))
        do s = 1, n_samples
            call posterior%joint_sample(points, generator, block, status)
            if (status%code /= FORTNUM_OK) then
                deallocate (self%draws)
                return
            end if
            self%draws(:, s) = block(:, 1)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine samples_generate

    pure integer function samples_n_points(self) result(q)
        class(fortbo_batch_samples_t), intent(in) :: self

        q = 0
        if (allocated(self%draws)) q = size(self%draws, 1)
    end function samples_n_points

    pure integer function samples_n_samples(self) result(n)
        class(fortbo_batch_samples_t), intent(in) :: self

        n = 0
        if (allocated(self%draws)) n = size(self%draws, 2)
    end function samples_n_samples

    !! `E[ max_i max(best - xi - f_i, 0) ]`.
    subroutine fortbo_qei(samples, best, xi, value, status)
        type(fortbo_batch_samples_t), intent(in) :: samples
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: threshold, total, best_in_sample
        integer :: s, i

        value = 0.0_dp
        if (.not. allocated(samples%draws)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: samples have not been generated")
            return
        end if

        threshold = best - xi
        total = 0.0_dp
        do s = 1, samples%n_samples()
            best_in_sample = 0.0_dp
            do i = 1, samples%n_points()
                best_in_sample = max(best_in_sample, threshold - samples%draws(i, s))
            end do
            total = total + best_in_sample
        end do
        value = total/real(samples%n_samples(), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_qei

    !! Batch noisy expected improvement.
    !!
    !! The threshold is the best latent value at the already-evaluated points
    !! under the *same* draw index as the batch, for the same reason the
    !! sequential version couples them: a sample in which the observed points
    !! look good must be the sample in which the batch is judged.
    subroutine fortbo_qnei(samples, observed_draws, value, status)
        type(fortbo_batch_samples_t), intent(in) :: samples
        !! `observed_draws(observed_point, sample)` — latent values at the
        !! already-evaluated inputs, one column per batch sample.
        real(dp), intent(in) :: observed_draws(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: total, incumbent, best_in_sample
        integer :: s, i, j

        value = 0.0_dp
        if (.not. allocated(samples%draws)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: samples have not been generated")
            return
        end if
        if (size(observed_draws, 1) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: at least one observed point is required")
            return
        end if
        if (size(observed_draws, 2) /= samples%n_samples()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: observed draws must match the sample count")
            return
        end if

        total = 0.0_dp
        do s = 1, samples%n_samples()
            incumbent = huge(1.0_dp)
            do j = 1, size(observed_draws, 1)
                incumbent = min(incumbent, observed_draws(j, s))
            end do
            best_in_sample = 0.0_dp
            do i = 1, samples%n_points()
                best_in_sample = max(best_in_sample, incumbent - samples%draws(i, s))
            end do
            total = total + best_in_sample
        end do
        value = total/real(samples%n_samples(), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_qnei

    !! `E[ max_i (mu_i + beta |f_i - mu_i|) ]`, the reparameterized form.
    !!
    !! FortBO minimizes, so a *lower* score is better and the quantity returned
    !! is negated relative to the maximization convention: `fortbo_qucb` returns
    !! the expected best (smallest) optimistic value in the batch, and a caller
    !! minimizes it. The absolute value is what makes this an optimistic bound
    !! rather than a mean — without it the beta term would average away.
    subroutine fortbo_qucb(samples, mean, beta, value, status)
        type(fortbo_batch_samples_t), intent(in) :: samples
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: beta
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: total, best_in_sample, optimistic
        integer :: s, i

        value = 0.0_dp
        if (.not. allocated(samples%draws)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: samples have not been generated")
            return
        end if
        if (size(mean) /= samples%n_points()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: mean must have one entry per batch point")
            return
        end if
        if (beta < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo batch: beta must not be negative")
            return
        end if

        total = 0.0_dp
        do s = 1, samples%n_samples()
            best_in_sample = huge(1.0_dp)
            do i = 1, samples%n_points()
                optimistic = mean(i) - beta*abs(samples%draws(i, s) - mean(i))
                best_in_sample = min(best_in_sample, optimistic)
            end do
            total = total + best_in_sample
        end do
        value = total/real(samples%n_samples(), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_qucb

end module fortbo_batch
