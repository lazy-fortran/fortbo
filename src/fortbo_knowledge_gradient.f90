module fortbo_knowledge_gradient
    !! Knowledge gradient (ROADMAP BO1).
    !!
    !! Expected improvement asks how much better the *next observation* will be.
    !! Knowledge gradient asks a different and usually more useful question: how
    !! much better will the decision I finally make be, if I sample here first?
    !! Formally, for minimization over a reference set `A`,
    !!
    !!     KG(x) = min_a mu_n(a) - E[ min_a mu_{n+1}(a) | sample at x ].
    !!
    !! The difference matters whenever the point worth sampling is not the point
    !! worth reporting: KG will happily pay for an observation at a place it
    !! would never recommend, because that observation sharpens the posterior
    !! somewhere it would. Expected improvement cannot express that, which is
    !! why it degrades badly under heavy noise while KG does not.
    !!
    !! Conditioning on one fantasized observation at `x` shifts every reference
    !! mean along a line in a single standard normal `Z`:
    !!
    !!     mu_{n+1}(a) = mu_n(a) + sigma_tilde(a) Z,
    !!     sigma_tilde(a) = cov_n(a, x) / sqrt(sigma_n^2(x) + noise).
    !!
    !! So the inner expectation is the expected minimum of a finite family of
    !! affine functions of one scalar normal. That has a closed form, and this
    !! module computes it exactly by Frazier's construction: sort the lines by
    !! slope, discard the ones that are nowhere on the lower envelope, and
    !! integrate the envelope piece by piece. The result is exact to rounding,
    !! deterministic, and free of the sampling noise that would otherwise turn a
    !! tie between two candidates into a coin flip and break replay.
    !!
    !! The reference set is supplied by the caller rather than inferred. KG is
    !! defined relative to the set of points the run is willing to *report*, and
    !! that is a decision about the problem, not about the acquisition.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_normal
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE
    implicit none
    private

    public :: fortbo_knowledge_gradient_value
    public :: fortbo_expected_minimum_of_lines
    public :: fortbo_batch_knowledge_gradient

    !! Two slopes closer than this are the same line for envelope purposes.
    !! Distinguishing them would divide by their difference when computing an
    !! intersection, turning rounding into an arbitrary breakpoint.
    real(dp), parameter, public :: FORTBO_KG_SLOPE_TOLERANCE = 1.0e-300_dp

contains

    !! `E[ min_i (intercept_i + slope_i Z) ]` for a standard normal `Z`, exactly.
    !!
    !! Computed as `-E[ max_i (-intercept_i - slope_i Z) ]`. The detour is
    !! deliberate. The minimum of affine functions is *concave*, so ordering the
    !! lines by increasing slope traverses its envelope from right to left,
    !! while the maximum is convex and traverses left to right. Writing the
    !! sweep once, for the convex case, and negating is the difference between
    !! one correct routine and two nearly-identical ones with opposite
    !! inequalities — which is exactly the kind of pair that ends up
    !! inconsistent. An earlier version swept the minimum directly with the
    !! convex convention and silently kept lines that are never on the envelope.
    !!
    !! With the survivors sorted by increasing slope and breakpoints
    !! `c_1 < ... < c_{m-1}` between consecutive ones, line `k` is on the
    !! envelope for `Z` in `(c_{k-1}, c_k)` and
    !!
    !!     E[max] = sum_k a_k (Phi(c_k) - Phi(c_{k-1}))
    !!            + sum_k (b_{k+1} - b_k) phi(c_k),
    !!
    !! using `integral z phi(z) = -phi(z)` on each piece, with `Phi(c_0) = 0`
    !! and `Phi(c_m) = 1`.
    subroutine fortbo_expected_minimum_of_lines(intercepts, slopes, expectation, &
            status)
        real(dp), intent(in) :: intercepts(:)
        real(dp), intent(in) :: slopes(:)
        real(dp), intent(out) :: expectation
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: work_intercept(:), work_slope(:)
        real(dp), allocatable :: keep_intercept(:), keep_slope(:), breakpoint(:)
        integer, allocatable :: order(:)
        real(dp) :: crossing, lower_cdf, upper_cdf, total
        integer :: n, i, kept, k

        expectation = 0.0_dp
        n = size(intercepts)
        if (n < 1 .or. size(slopes) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: line arrays must agree and be nonempty")
            return
        end if

        ! Negate into the convex problem.
        allocate (order(n), work_intercept(n), work_slope(n))
        call sort_by_slope_then_intercept(-slopes, -intercepts, order)
        do i = 1, n
            work_slope(i) = -slopes(order(i))
            work_intercept(i) = -intercepts(order(i))
        end do

        allocate (keep_intercept(n), keep_slope(n), breakpoint(n))
        kept = 0
        do i = 1, n
            ! Equal slopes: only the *higher* intercept can be on an upper
            ! envelope, and the sort has put it first among its ties.
            if (kept >= 1) then
                if (work_slope(i) - keep_slope(kept) <= FORTBO_KG_SLOPE_TOLERANCE) &
                    cycle
            end if
            do
                if (kept == 0) exit
                crossing = (keep_intercept(kept) - work_intercept(i))/ &
                    (work_slope(i) - keep_slope(kept))
                if (kept == 1) exit
                ! The new line takes over at or before the point where the
                ! current top took over, so the top never appears at all.
                if (crossing > breakpoint(kept - 1)) exit
                kept = kept - 1
            end do
            kept = kept + 1
            keep_slope(kept) = work_slope(i)
            keep_intercept(kept) = work_intercept(i)
            if (kept > 1) breakpoint(kept - 1) = crossing
        end do

        if (kept == 1) then
            expectation = -keep_intercept(1)
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        total = 0.0_dp
        do k = 1, kept
            if (k == 1) then
                lower_cdf = 0.0_dp
            else
                lower_cdf = normal_cdf(breakpoint(k - 1))
            end if
            if (k == kept) then
                upper_cdf = 1.0_dp
            else
                upper_cdf = normal_cdf(breakpoint(k))
            end if
            total = total + keep_intercept(k)*(upper_cdf - lower_cdf)
        end do
        do k = 1, kept - 1
            total = total + (keep_slope(k + 1) - keep_slope(k))* &
                normal_pdf(breakpoint(k))
        end do
        expectation = -total
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_expected_minimum_of_lines

    pure real(dp) function normal_cdf(z) result(value)
        real(dp), intent(in) :: z

        value = 0.5_dp*(1.0_dp + erf(z/sqrt(2.0_dp)))
    end function normal_cdf

    pure real(dp) function normal_pdf(z) result(value)
        real(dp), intent(in) :: z

        value = exp(-0.5_dp*z*z)/sqrt(8.0_dp*atan(1.0_dp))
    end function normal_pdf

    !! Knowledge gradient of sampling at `candidate`, relative to `reference`.
    !!
    !! `noise_variance` is the observation noise the fantasized measurement would
    !! carry. It belongs here rather than in the posterior because it describes
    !! the instrument, not the belief, and because a run that can pay for a
    !! quieter measurement changes it without refitting anything.
    subroutine fortbo_knowledge_gradient_value(posterior, candidate, reference, &
            noise_variance, value, status)
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: candidate(:)
        !! `reference(point, coordinate)` — the set the run would report from.
        real(dp), intent(in) :: reference(:, :)
        real(dp), intent(in) :: noise_variance
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: joint(:, :), covariance(:, :)
        real(dp), allocatable :: mean(:), variance(:), tilde(:)
        real(dp) :: denominator, current_best, expected_best
        integer :: m, d, i

        value = 0.0_dp
        m = size(reference, 1)
        d = size(candidate)
        if (m < 1 .or. d < 1 .or. size(reference, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: reference set shape is invalid")
            return
        end if
        if (noise_variance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: noise variance must not be negative")
            return
        end if
        if (.not. posterior%supports(FORTBO_CAP_MOMENTS) .or. &
            .not. posterior%supports(FORTBO_CAP_COVARIANCE)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo knowledge gradient: posterior lacks joint covariance")
            return
        end if

        ! One joint query: the reference set with the candidate appended, so the
        ! covariances and the moments come from the same evaluation and cannot
        ! disagree.
        allocate (joint(m + 1, d))
        joint(:m, :) = reference
        joint(m + 1, :) = candidate
        allocate (covariance(m + 1, m + 1), mean(m + 1), variance(m + 1))
        call posterior%moments(joint, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call posterior%covariance(joint, covariance, status)
        if (status%code /= FORTNUM_OK) return

        denominator = sqrt(max(variance(m + 1), 0.0_dp) + noise_variance)
        current_best = minval(mean(:m))
        if (denominator <= 0.0_dp) then
            ! A noiseless observation at a point the posterior already knows
            ! exactly teaches nothing, so the decision cannot improve.
            value = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        allocate (tilde(m))
        do i = 1, m
            tilde(i) = covariance(i, m + 1)/denominator
        end do

        call fortbo_expected_minimum_of_lines(mean(:m), tilde, expected_best, status)
        if (status%code /= FORTNUM_OK) return

        ! Non-negativity is a theorem, not a clamp: information cannot make the
        ! best decision worse in expectation. Rounding can still produce a tiny
        ! negative, and reporting that would let a comparison prefer a candidate
        ! for a reason that is pure noise.
        value = max(current_best - expected_best, 0.0_dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_knowledge_gradient_value

    !! Batch knowledge gradient over `q` simultaneous fantasized observations.
    !!
    !! Written against Wu, Poloczek, Wilson and Frazier, *Bayesian Optimization
    !! with Gradients* (arXiv:1703.04389), whose equation (3.4) states
    !!
    !!     d-KG = min_A mu_n - E_n[ min_A mu_{n+q} | batch ],
    !!
    !! with the expectation marginalizing over all `q` observations at once.
    !!
    !! The sequential case collapsed to a one-dimensional envelope because a
    !! single fantasy shifts every reference mean along a line in one scalar.
    !! With `q` fantasies the shift is affine in a `q`-vector, and the minimum
    !! of affine functions of a multivariate normal has no such closed form.
    !! This is therefore Monte Carlo over the fantasy vector — and that is a
    !! statement about the problem, not a shortcut: the exact quantity is a
    !! `q`-dimensional integral of a piecewise-linear function.
    !!
    !! `loading(reference, slot)` is the caller's factorization of how each
    !! fantasy moves each reference mean, the batch analogue of `sigma_tilde`.
    !! Supplying it rather than deriving it here keeps the posterior's
    !! factorization where the posterior is, and lets a caller reuse one
    !! factorization across many candidate batches.
    !!
    !! Draws are taken from a caller-owned generator and can be frozen by
    !! seeding it, so two candidate batches are compared against the same
    !! realizations rather than against separate noise. Comparing batches under
    !! independent draws would rank them by sampling error whenever their true
    !! values are close, which is exactly when the ranking matters.
    subroutine fortbo_batch_knowledge_gradient(reference_mean, loading, &
            n_samples, generator, value, status)
        real(dp), intent(in) :: reference_mean(:)
        !! `loading(reference, slot)`.
        real(dp), intent(in) :: loading(:, :)
        integer, intent(in) :: n_samples
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: draw(:), shifted(:)
        real(dp) :: current_best, total, sample_min
        integer :: m, q, s, i, j

        value = 0.0_dp
        m = size(reference_mean)
        q = size(loading, 2)
        if (m < 1 .or. size(loading, 1) /= m) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: loading must match the reference set")
            return
        end if
        if (q < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: a batch needs at least one point")
            return
        end if
        if (n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo knowledge gradient: sample count must be positive")
            return
        end if

        allocate (draw(q), shifted(m))
        current_best = minval(reference_mean)
        total = 0.0_dp
        do s = 1, n_samples
            do j = 1, q
                call rng_normal(generator, draw(j))
            end do
            shifted = reference_mean
            do j = 1, q
                do i = 1, m
                    shifted(i) = shifted(i) + loading(i, j)*draw(j)
                end do
            end do
            sample_min = minval(shifted)
            total = total + sample_min
        end do

        ! Non-negativity is a theorem: information cannot make the best
        ! decision worse in expectation. Sampling error can still produce a
        ! small negative, and reporting it would let a batch be preferred for
        ! a reason that is pure noise.
        value = max(current_best - total/real(n_samples, dp), 0.0_dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_batch_knowledge_gradient

    !! Ascending by slope, and *descending* by intercept within equal slopes.
    !!
    !! The descending tie-break is what lets the sweep dispose of parallel lines
    !! with a single `cycle`: on an upper envelope only the highest of a set of
    !! parallel lines can ever appear, so putting it first means every later
    !! member of the tie is skipped without computing an intersection that would
    !! divide by a zero slope difference. Ascending would keep the wrong one.
    !!
    !! Insertion sort: the reference sets here are small, and a stable order
    !! keeps the result reproducible.
    pure subroutine sort_by_slope_then_intercept(slopes, intercepts, order)
        real(dp), intent(in) :: slopes(:)
        real(dp), intent(in) :: intercepts(:)
        integer, intent(out) :: order(:)
        integer :: i, j, moving

        do i = 1, size(order)
            order(i) = i
        end do
        do i = 2, size(order)
            moving = order(i)
            j = i - 1
            do while (j >= 1)
                if (slopes(order(j)) < slopes(moving)) exit
                if (slopes(order(j)) == slopes(moving) .and. &
                    intercepts(order(j)) >= intercepts(moving)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = moving
        end do
    end subroutine sort_by_slope_then_intercept

end module fortbo_knowledge_gradient
