module fortbo_integrated
    !! Integrated acquisitions: marginalizing surrogate hyperparameters
    !! (ROADMAP BO2).
    !!
    !! Written against Snoek, Larochelle and Adams, *Practical Bayesian
    !! Optimization of Machine Learning Algorithms* (arXiv:1206.2944), fetched
    !! by `fortbo-bench/scripts/fetch_provenance.py`. The paper states it
    !! plainly: for probability of improvement and expected improvement, the
    !! expectation over hyperparameters "is the correct generalization to
    !! account for uncertainty in hyperparameters", and the samples "can be
    !! acquired efficiently using slice sampling".
    !!
    !! **The plug-in estimate is not a simplification, it is a different
    !! acquisition.** Fitting hyperparameters by maximum likelihood and then
    !! acting as though they were known makes a run overconfident exactly when
    !! it can least afford to be — early, with few observations, when the
    !! lengthscale is barely determined and the choice of it dominates where the
    !! acquisition points. Averaging the *acquisition* over hyperparameter
    !! samples is what the definition asks for.
    !!
    !! **The average is of acquisitions, not of moments.** Averaging the
    !! posterior means and variances first and then computing one acquisition
    !! would be a different and wrong quantity: expected improvement is a
    !! nonlinear function of the moments, so by Jensen the two disagree, and the
    !! moment-averaged version systematically understates the value of a point
    !! the hyperparameter samples disagree about — which is precisely the point
    !! worth evaluating. `fortbo_integrated_acquisition` therefore takes an
    !! acquisition value per sample and averages those.
    !!
    !! Sampling belongs to FortMC and refitting belongs to the caller, so this
    !! module holds only the blending and the diagnostics that say whether the
    !! blend meant anything.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_integrated_acquisition
    public :: fortbo_hyperparameter_spread
    public :: fortbo_integration_is_informative

    !! Below this relative spread across hyperparameter samples, the integrated
    !! acquisition and the plug-in one cannot differ meaningfully and the extra
    !! cost buys nothing. Reported rather than acted on: whether to keep paying
    !! for a chain is the caller's decision.
    real(dp), parameter, public :: FORTBO_INTEGRATION_SPREAD_FLOOR = 1.0e-6_dp

contains

    !! Average an acquisition over hyperparameter samples.
    !!
    !! `values(point, sample)` is the acquisition at each point under each
    !! sampled hyperparameter vector, which the caller produced by refitting.
    subroutine fortbo_integrated_acquisition(values, integrated, status, weights)
        real(dp), intent(in) :: values(:, :)
        real(dp), intent(out) :: integrated(:)
        type(fortnum_status_t), intent(out) :: status
        !! Optional per-sample weights, for an importance-weighted chain. A
        !! plain MCMC chain is already distributed correctly and needs none.
        real(dp), intent(in), optional :: weights(:)
        real(dp), allocatable :: normalized(:)
        real(dp) :: total
        integer :: n, m, i, k

        integrated = 0.0_dp
        n = size(values, 1)
        m = size(values, 2)
        if (n < 1 .or. m < 1 .or. size(integrated) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo integrated: value and output shapes disagree")
            return
        end if

        allocate (normalized(m))
        if (present(weights)) then
            if (size(weights) /= m) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo integrated: weights must match the sample count")
                return
            end if
            if (any(weights < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo integrated: weights must not be negative")
                return
            end if
            total = sum(weights)
            if (total <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo integrated: weights must not all be zero")
                return
            end if
            normalized = weights/total
        else
            normalized = 1.0_dp/real(m, dp)
        end if

        do i = 1, n
            do k = 1, m
                integrated(i) = integrated(i) + normalized(k)*values(i, k)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_integrated_acquisition

    !! Relative spread of each hyperparameter across the chain.
    !!
    !! A diagnostic, not a gate. A chain whose lengthscale samples all agree
    !! says the data have determined it, and integrating is then honest but
    !! pointless; one whose samples disagree wildly says the opposite, and a
    !! plug-in acquisition on it would be a confident answer to a question
    !! nobody can yet answer.
    subroutine fortbo_hyperparameter_spread(samples, spread, status)
        !! `samples(parameter, draw)`.
        real(dp), intent(in) :: samples(:, :)
        real(dp), intent(out) :: spread(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: mean, variance
        integer :: d, m, j

        spread = 0.0_dp
        d = size(samples, 1)
        m = size(samples, 2)
        if (d < 1 .or. m < 2 .or. size(spread) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo integrated: at least two draws are needed for a spread")
            return
        end if

        do j = 1, d
            mean = sum(samples(j, :))/real(m, dp)
            variance = sum((samples(j, :) - mean)**2)/real(m - 1, dp)
            ! Relative to the mean's magnitude, so a lengthscale of 100 and one
            ! of 0.01 are comparable. Falling back to the absolute deviation
            ! when the mean is near zero, where a ratio says nothing.
            if (abs(mean) > FORTBO_INTEGRATION_SPREAD_FLOOR) then
                spread(j) = sqrt(variance)/abs(mean)
            else
                spread(j) = sqrt(variance)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_hyperparameter_spread

    !! Whether integrating could have changed anything.
    !!
    !! False when every hyperparameter is pinned across the chain, in which case
    !! the integrated acquisition equals the plug-in one to rounding and the
    !! chain was wasted effort. Saying so lets a run report the fact rather than
    !! quietly paying for a marginalization that did nothing.
    pure logical function fortbo_integration_is_informative(spread) result(useful)
        real(dp), intent(in) :: spread(:)

        useful = .false.
        if (size(spread) < 1) return
        useful = maxval(spread) > FORTBO_INTEGRATION_SPREAD_FLOOR
    end function fortbo_integration_is_informative

end module fortbo_integrated
