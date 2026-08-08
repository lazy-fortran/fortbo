program test_risk
    !! BO1: risk-sensitive and multi-fidelity acquisitions.
    !!
    !! Oracles:
    !!
    !!   * VaR and CVaR are checked against Monte Carlo simulation of the very
    !!     posterior they summarize, by counting rather than sorting: VaR is
    !!     defined by the fraction of outcomes below it, so measuring that
    !!     fraction tests the definition directly;
    !!   * the ordering CVaR >= VaR >= mean holds by construction and is
    !!     asserted, because it is the cheapest way to catch a swapped formula;
    !!   * the multi-fidelity weighting is checked as an ordering, and
    !!     specifically for the error it exists to avoid: a weakly correlated
    !!     cheap fidelity must not outrank a strongly correlated one until it is
    !!     cheap by the *square* of the correlation ratio.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_risk, only: fortbo_mean_variance, fortbo_value_at_risk, &
        fortbo_conditional_value_at_risk, fortbo_multi_fidelity_acquisition
    implicit none

    integer :: failures

    failures = 0
    call check_tails_against_monte_carlo(failures)
    call check_risk_ordering(failures)
    call check_mean_variance(failures)
    call check_multi_fidelity(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_risk: PASS"
    else
        print *, "test_risk: FAIL", failures
        error stop 1
    end if

contains

    !! Simulate the posterior and check the two summaries against it directly,
    !! without sorting.
    !!
    !! An earlier version sorted the draws to read off the quantile. That is the
    !! obvious oracle and it was a mistake: an insertion sort over 200000 draws
    !! is quadratic and the test timed out. Counting is not only faster, it is a
    !! sharper check — VaR is *defined* by the fraction of outcomes below it, so
    !! measuring that fraction tests the definition rather than an artifact of
    !! how the sample was ordered.
    subroutine check_tails_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        real(dp) :: mean(1), sd(1), var_value(1), cvar_value(1)
        real(dp) :: draw, sample, tail_total, fraction, standard_error
        real(dp) :: empirical_cvar
        integer :: s, below, tail_count
        real(dp), parameter :: alpha = 0.9_dp

        mean(1) = 1.25_dp
        sd(1) = 0.75_dp

        call fortbo_value_at_risk(mean, sd, alpha, var_value, status)
        call expect(status%code == FORTNUM_OK, "value at risk computes", failures)
        call fortbo_conditional_value_at_risk(mean, sd, alpha, cvar_value, status)
        call expect(status%code == FORTNUM_OK, &
            "conditional value at risk computes", failures)

        call rng_seed(generator, int(717171, int64), status)
        below = 0
        tail_total = 0.0_dp
        tail_count = 0
        do s = 1, n_samples
            call rng_normal(generator, draw)
            sample = mean(1) + sd(1)*draw
            if (sample <= var_value(1)) then
                below = below + 1
            else
                tail_total = tail_total + sample
                tail_count = tail_count + 1
            end if
        end do

        ! The defining property: exactly alpha of the mass sits below VaR.
        fraction = real(below, dp)/real(n_samples, dp)
        standard_error = sqrt(alpha*(1.0_dp - alpha)/real(n_samples, dp))
        call expect(abs(fraction - alpha) < 5.0_dp*standard_error, &
            "the reported quantile has exactly alpha of the mass below it", &
            failures)

        ! Given that, CVaR is the mean of what is above it.
        empirical_cvar = tail_total/real(tail_count, dp)
        standard_error = sd(1)/sqrt(real(tail_count, dp))
        call expect(abs(cvar_value(1) - empirical_cvar) < 5.0_dp*standard_error, &
            "conditional value at risk matches the simulated tail mean", failures)
    end subroutine check_tails_against_monte_carlo

    !! CVaR >= VaR >= mean for alpha above one half, by construction.
    subroutine check_risk_ordering(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(3), sd(3), var_value(3), cvar_value(3)
        integer :: k
        logical :: ordered

        mean = [0.0_dp, -1.0_dp, 2.5_dp]
        sd = [1.0_dp, 0.25_dp, 3.0_dp]

        call fortbo_value_at_risk(mean, sd, 0.95_dp, var_value, status)
        call fortbo_conditional_value_at_risk(mean, sd, 0.95_dp, cvar_value, status)
        ordered = .true.
        do k = 1, 3
            if (.not. (cvar_value(k) >= var_value(k))) ordered = .false.
            if (.not. (var_value(k) >= mean(k))) ordered = .false.
        end do
        call expect(ordered, &
            "conditional value at risk is at least value at risk is at least the mean", &
            failures)

        ! A certain outcome carries no risk, so all three coincide.
        sd = 0.0_dp
        call fortbo_value_at_risk(mean, sd, 0.95_dp, var_value, status)
        call fortbo_conditional_value_at_risk(mean, sd, 0.95_dp, cvar_value, status)
        call expect(maxval(abs(var_value - mean)) == 0.0_dp .and. &
            maxval(abs(cvar_value - mean)) == 0.0_dp, &
            "a certain outcome has no risk premium", failures)

        ! At the median the quantile is the mean, which pins the direction of
        ! the inverse CDF.
        sd = [1.0_dp, 0.25_dp, 3.0_dp]
        call fortbo_value_at_risk(mean, sd, 0.5_dp, var_value, status)
        call expect(maxval(abs(var_value - mean)) < 1.0e-12_dp, &
            "the median quantile is the mean", failures)

        ! More uncertainty must mean more risk, never less.
        call fortbo_conditional_value_at_risk([0.0_dp], [1.0_dp], 0.9_dp, &
            var_value(1:1), status)
        call fortbo_conditional_value_at_risk([0.0_dp], [2.0_dp], 0.9_dp, &
            cvar_value(1:1), status)
        call expect(cvar_value(1) > var_value(1), &
            "a wider posterior carries more tail risk", failures)
    end subroutine check_risk_ordering

    subroutine check_mean_variance(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(2), variance(2), value(2)

        mean = [0.0_dp, 0.0_dp]
        variance = [0.1_dp, 4.0_dp]

        call fortbo_mean_variance(mean, variance, 0.0_dp, value, status)
        call expect(maxval(abs(value - mean)) == 0.0_dp, &
            "lambda zero recovers the posterior mean", failures)

        call fortbo_mean_variance(mean, variance, 0.5_dp, value, status)
        call expect(value(1) < value(2), &
            "of two equal means the certain one is preferred", failures)

        ! Variance penalizes both tails, which is the honest limitation of this
        ! utility and is worth pinning: a point with a better mean can still
        ! lose to a worse but tighter one.
        call fortbo_mean_variance([-1.0_dp, 0.0_dp], [9.0_dp, 0.0_dp], 0.5_dp, &
            value, status)
        call expect(value(1) > value(2), &
            "mean-variance can reject a better mean for being uncertain", failures)
    end subroutine check_mean_variance

    subroutine check_multi_fidelity(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: base(3), correlation(3), cost(3), value(3)

        base = 1.0_dp

        ! Same cost: the better-correlated fidelity must win.
        correlation = [1.0_dp, 0.5_dp, -1.0_dp]
        cost = 1.0_dp
        call fortbo_multi_fidelity_acquisition(base, correlation, cost, 1.0_dp, &
            value, status)
        call expect(status%code == FORTNUM_OK, "the multi-fidelity weight computes", &
            failures)
        call expect(value(1) > value(2), &
            "a better-correlated fidelity wins at equal cost", failures)
        call expect(abs(value(1) - value(3)) < 1.0e-14_dp, &
            "a perfectly anti-correlated fidelity is just as informative", &
            failures)

        ! The squaring is the point. A fidelity with half the correlation
        ! carries a quarter of the information, so it must need to be four times
        ! cheaper to break even, not twice.
        base = 1.0_dp
        correlation = [1.0_dp, 0.5_dp, 0.5_dp]
        cost = [1.0_dp, 2.0_dp, 0.25_dp]
        call fortbo_multi_fidelity_acquisition(base, correlation, cost, 1.0_dp, &
            value, status)
        call expect(value(2) < value(1), &
            "half the correlation at twice the price is not a bargain", failures)
        call expect(abs(value(3) - value(1)) < 1.0e-14_dp, &
            "half the correlation breaks even at a quarter of the price", failures)

        ! A fidelity that says nothing is worth nothing at any price.
        call fortbo_multi_fidelity_acquisition([1.0_dp], [0.0_dp], [1.0e-9_dp], &
            1.0_dp, value(1:1), status)
        call expect(value(1) == 0.0_dp, &
            "an uncorrelated fidelity is worthless however cheap", failures)
    end subroutine check_multi_fidelity

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: value(2)

        call fortbo_mean_variance([0.0_dp, 0.0_dp], [1.0_dp, 1.0_dp], -0.5_dp, &
            value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a risk-seeking lambda is refused rather than accepted silently", &
            failures)

        call fortbo_mean_variance([0.0_dp, 0.0_dp], [-1.0_dp, 1.0_dp], 0.5_dp, &
            value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative variance is refused", failures)

        call fortbo_value_at_risk([0.0_dp, 0.0_dp], [1.0_dp, 1.0_dp], 1.0_dp, &
            value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "alpha at one is refused, since the tail would be empty", failures)

        call fortbo_conditional_value_at_risk([0.0_dp, 0.0_dp], [1.0_dp, 1.0_dp], &
            0.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "alpha at zero is refused", failures)

        call fortbo_multi_fidelity_acquisition([1.0_dp, 1.0_dp], [1.5_dp, 0.5_dp], &
            [1.0_dp, 1.0_dp], 1.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a correlation outside [-1,1] is refused", failures)

        call fortbo_multi_fidelity_acquisition([1.0_dp, 1.0_dp], [0.5_dp, 0.5_dp], &
            [0.0_dp, 1.0_dp], 1.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero fidelity cost is refused", failures)
    end subroutine check_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_risk
