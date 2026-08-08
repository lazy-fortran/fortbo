program test_constrained
    !! BO1: constrained and cost-aware acquisitions.
    !!
    !! Oracles:
    !!
    !!   * the feasibility probability is checked against Monte Carlo simulation
    !!     of the constraint beliefs it models, with the tolerance taken from the
    !!     binomial standard error;
    !!   * the weightings are checked as *orderings*, which is all an acquisition
    !!     is ever used for: a feasible point must outrank an infeasible one of
    !!     equal promise, and a cheap point must outrank an expensive one of
    !!     equal promise;
    !!   * limits that can be stated: certainty reduces to the unweighted
    !!     acquisition, `alpha = 0` ignores cost entirely;
    !!   * the refusals that keep the ordering meaningful, in particular that a
    !!     negative base acquisition is rejected rather than silently inverted.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_constrained, only: fortbo_feasibility_probability, &
        fortbo_constrained_acquisition, fortbo_cost_aware_acquisition
    implicit none

    integer :: failures

    failures = 0
    call check_feasibility_against_monte_carlo(failures)
    call check_feasibility_limits(failures)
    call check_constrained_ordering(failures)
    call check_cost_aware_ordering(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_constrained: PASS"
    else
        print *, "test_constrained: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_feasibility_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_constraints = 3, n_samples = 400000
        real(dp) :: mean(n_constraints, 1), sd(n_constraints, 1)
        real(dp) :: threshold(n_constraints), probability(1)
        real(dp) :: draw, value, estimate, standard_error
        integer :: k, s, feasible
        logical :: all_satisfied

        mean(:, 1) = [-0.5_dp, 0.2_dp, -1.0_dp]
        sd(:, 1) = [0.4_dp, 0.6_dp, 0.3_dp]
        threshold = 0.0_dp

        call fortbo_feasibility_probability(mean, sd, threshold, probability, status)
        call expect(status%code == FORTNUM_OK, "the feasibility probability computes", &
            failures)

        call rng_seed(generator, int(505050, int64), status)
        feasible = 0
        do s = 1, n_samples
            all_satisfied = .true.
            do k = 1, n_constraints
                call rng_normal(generator, draw)
                value = mean(k, 1) + sd(k, 1)*draw
                if (value > threshold(k)) all_satisfied = .false.
            end do
            if (all_satisfied) feasible = feasible + 1
        end do
        estimate = real(feasible, dp)/real(n_samples, dp)
        standard_error = sqrt(probability(1)*(1.0_dp - probability(1)) &
            /real(n_samples, dp))
        call expect(abs(estimate - probability(1)) < 5.0_dp*standard_error, &
            "feasibility matches simulated constraint beliefs", failures)
    end subroutine check_feasibility_against_monte_carlo

    subroutine check_feasibility_limits(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(2, 3), sd(2, 3), threshold(2), probability(3)

        threshold = 0.0_dp
        ! Point 1 certainly feasible, point 2 certainly infeasible, point 3 on
        ! the boundary in one constraint.
        mean(:, 1) = [-1.0_dp, -2.0_dp]
        mean(:, 2) = [1.0_dp, -2.0_dp]
        mean(:, 3) = [0.0_dp, -2.0_dp]
        sd = 0.0_dp

        call fortbo_feasibility_probability(mean, sd, threshold, probability, status)
        call expect(probability(1) == 1.0_dp, &
            "a certainly feasible point has probability one", failures)
        call expect(probability(2) == 0.0_dp, &
            "one violated constraint drives the product to zero", failures)
        call expect(abs(probability(3) - 0.5_dp) < 1.0e-14_dp, &
            "a boundary constraint contributes one half", failures)

        ! Adding a constraint can only reduce feasibility.
        block
            real(dp) :: one(1, 1), one_sd(1, 1), one_threshold(1), one_probability(1)
            real(dp) :: two(2, 1), two_sd(2, 1), two_threshold(2), two_probability(1)
            one(1, 1) = -0.3_dp
            one_sd(1, 1) = 0.5_dp
            one_threshold = 0.0_dp
            two(:, 1) = [-0.3_dp, -0.1_dp]
            two_sd(:, 1) = [0.5_dp, 0.4_dp]
            two_threshold = 0.0_dp
            call fortbo_feasibility_probability(one, one_sd, one_threshold, &
                one_probability, status)
            call fortbo_feasibility_probability(two, two_sd, two_threshold, &
                two_probability, status)
            call expect(two_probability(1) < one_probability(1), &
                "an extra constraint can only lower feasibility", failures)
        end block
    end subroutine check_feasibility_limits

    !! The ordering is the whole product. Two points of identical promise must
    !! be separated by feasibility, and certainty must leave the base
    !! acquisition untouched.
    subroutine check_constrained_ordering(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: base(3), probability(3), value(3)

        base = [1.0_dp, 1.0_dp, 2.0_dp]
        probability = [0.9_dp, 0.1_dp, 0.05_dp]
        call fortbo_constrained_acquisition(base, probability, value, status)
        call expect(status%code == FORTNUM_OK, "the constrained acquisition computes", &
            failures)
        call expect(value(1) > value(2), &
            "of two equally promising points the feasible one wins", failures)
        call expect(value(1) > value(3), &
            "a modest but likely-feasible point beats a better unlikely one", &
            failures)

        ! Certain feasibility must leave the acquisition exactly alone.
        probability = 1.0_dp
        call fortbo_constrained_acquisition(base, probability, value, status)
        call expect(maxval(abs(value - base)) == 0.0_dp, &
            "certain feasibility reduces to the unweighted acquisition", failures)

        ! Certain infeasibility must remove the point entirely.
        probability = 0.0_dp
        call fortbo_constrained_acquisition(base, probability, value, status)
        call expect(all(value == 0.0_dp), &
            "a certainly infeasible point scores exactly zero", failures)
    end subroutine check_constrained_ordering

    subroutine check_cost_aware_ordering(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: base(3), cost(3), value(3), half(3)

        base = [1.0_dp, 1.0_dp, 1.5_dp]
        cost = [1.0_dp, 10.0_dp, 10.0_dp]

        call fortbo_cost_aware_acquisition(base, cost, 1.0_dp, value, status)
        call expect(status%code == FORTNUM_OK, "the cost-aware acquisition computes", &
            failures)
        call expect(value(1) > value(2), &
            "of two equally promising points the cheap one wins", failures)
        call expect(value(1) > value(3), &
            "a cheap modest point beats an expensive better one per unit cost", &
            failures)

        ! alpha zero ignores cost entirely, which is what makes alpha an
        ! interpolation rather than an arbitrary exponent.
        call fortbo_cost_aware_acquisition(base, cost, 0.0_dp, value, status)
        call expect(maxval(abs(value - base)) == 0.0_dp, &
            "alpha zero recovers the unweighted acquisition", failures)

        ! A partial alpha must sit between the two extremes.
        call fortbo_cost_aware_acquisition(base, cost, 0.5_dp, half, status)
        call fortbo_cost_aware_acquisition(base, cost, 1.0_dp, value, status)
        call expect(half(2) > value(2) .and. half(2) < base(2), &
            "a partial alpha interpolates between ignoring and charging cost", &
            failures)

        ! Equal costs must leave the ordering of the base acquisition alone.
        cost = 7.0_dp
        call fortbo_cost_aware_acquisition(base, cost, 1.0_dp, value, status)
        call expect(value(3) > value(1) .and. value(1) == value(2), &
            "a uniform cost cannot change the ordering", failures)
    end subroutine check_cost_aware_ordering

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: base(2), probability(2), cost(2), value(2)
        real(dp) :: mean(2, 2), sd(2, 2), threshold(2), feasibility(2)

        base = [1.0_dp, 1.0_dp]
        probability = [0.5_dp, 0.5_dp]
        cost = [1.0_dp, 1.0_dp]

        ! The refusal that matters: multiplying a negative acquisition by a
        ! probability moves it up, so an infeasible point would outrank a
        ! feasible one with nothing in the output saying so.
        call fortbo_constrained_acquisition([-1.0_dp, 1.0_dp], probability, value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative base acquisition is refused, not silently inverted", &
            failures)

        call fortbo_constrained_acquisition(base, [1.5_dp, 0.5_dp], value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a weight outside [0,1] is refused", failures)

        call fortbo_cost_aware_acquisition(base, [0.0_dp, 1.0_dp], 1.0_dp, value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero cost is refused rather than clamped", failures)

        call fortbo_cost_aware_acquisition(base, [-1.0_dp, 1.0_dp], 1.0_dp, value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative cost is refused", failures)

        call fortbo_cost_aware_acquisition(base, cost, 1.5_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an alpha outside [0,1] is refused", failures)

        mean = 0.0_dp
        sd = 0.0_dp
        sd(1, 1) = -1.0_dp
        threshold = 0.0_dp
        call fortbo_feasibility_probability(mean, sd, threshold, feasibility, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative constraint standard deviation is refused", failures)
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

end program test_constrained
