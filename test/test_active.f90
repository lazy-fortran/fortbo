program test_active
    !! BO4: active learning, level-set estimation, and design of experiments.
    !!
    !! Oracles:
    !!
    !!   * the level-set probability is checked against Monte Carlo simulation
    !!     of the posterior it summarizes;
    !!   * the straddle rule is checked by the behavior it exists for, on a
    !!     constructed case where the two terms disagree: the point nearest the
    !!     contour must lose to one slightly further away but far less certain,
    !!     and pure variance sampling must pick a different point entirely;
    !!   * the maximin score is checked against an exhaustive pairwise
    !!     computation and against a design whose spacing is known by
    !!     construction;
    !!   * every score follows the lower-is-better convention, which is checked
    !!     rather than assumed, since a sign error here would silently make
    !!     every optimizer seek the *most* certain point.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_active, only: fortbo_active_learning_score, fortbo_straddle_score, &
        fortbo_level_set_probability, fortbo_minimum_pairwise_distance, &
        fortbo_maximin_score
    implicit none

    integer :: failures

    failures = 0
    call check_active_learning_direction(failures)
    call check_level_set_against_monte_carlo(failures)
    call check_straddle_balances_its_two_terms(failures)
    call check_maximin(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_active: PASS"
    else
        print *, "test_active: FAIL", failures
        error stop 1
    end if

contains

    !! Lower is better, so the least certain point must score lowest. A sign
    !! error here would make every optimizer seek the most certain point, which
    !! is the exact opposite of active learning and would still run cleanly.
    subroutine check_active_learning_direction(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: variance(4), score(4)
        integer :: chosen

        variance = [0.1_dp, 2.5_dp, 0.4_dp, 1.0_dp]
        call fortbo_active_learning_score(variance, score, status)
        call expect(status%code == FORTNUM_OK, "the active learning score computes", &
            failures)
        chosen = minloc(score, dim=1)
        call expect(chosen == 2, &
            "the least certain point is selected", failures)
        call expect(all(score <= 0.0_dp), &
            "the score is a negated variance", failures)
    end subroutine check_active_learning_direction

    subroutine check_level_set_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        real(dp) :: mean(1), sd(1), probability(1)
        real(dp) :: draw, estimate, standard_error
        real(dp), parameter :: threshold = 0.75_dp
        integer :: s, below

        mean(1) = 0.4_dp
        sd(1) = 0.6_dp
        call fortbo_level_set_probability(mean, sd, threshold, probability, status)
        call expect(status%code == FORTNUM_OK, "the level-set probability computes", &
            failures)

        call rng_seed(generator, int(838383, int64), status)
        below = 0
        do s = 1, n_samples
            call rng_normal(generator, draw)
            if (mean(1) + sd(1)*draw <= threshold) below = below + 1
        end do
        estimate = real(below, dp)/real(n_samples, dp)
        standard_error = sqrt(probability(1)*(1.0_dp - probability(1)) &
            /real(n_samples, dp))
        call expect(abs(estimate - probability(1)) < 5.0_dp*standard_error, &
            "level-set membership matches the simulated posterior", failures)

        ! At the threshold the answer is one half whatever the spread.
        call fortbo_level_set_probability([threshold], [3.0_dp], threshold, &
            probability, status)
        call expect(abs(probability(1) - 0.5_dp) < 1.0e-14_dp, &
            "a point exactly on the contour is a coin flip", failures)
    end subroutine check_level_set_against_monte_carlo

    !! The straddle rule exists because neither of its terms is enough alone.
    !! The case here makes them disagree: point 1 sits on the contour but is
    !! almost certain, point 2 is slightly off it but far less certain, and
    !! point 3 is wildly uncertain but nowhere near the contour.
    subroutine check_straddle_balances_its_two_terms(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(3), sd(3), score(3), variance_score(3)
        real(dp), parameter :: threshold = 1.0_dp

        mean = [1.0_dp, 1.2_dp, 8.0_dp]
        sd = [0.01_dp, 0.9_dp, 3.0_dp]

        call fortbo_straddle_score(mean, sd, threshold, 1.96_dp, score, status)
        call expect(status%code == FORTNUM_OK, "the straddle score computes", &
            failures)

        call expect(score(2) < score(1), &
            "a slightly off-contour but uncertain point beats a settled one", &
            failures)
        call expect(score(2) < score(3), &
            "a relevant uncertain point beats an irrelevant more uncertain one", &
            failures)

        ! Pure variance sampling would pick point 3, which is the failure the
        ! straddle rule exists to avoid.
        call fortbo_active_learning_score(sd**2, variance_score, status)
        call expect(minloc(variance_score, dim=1) == 3, &
            "pure variance sampling chases the least-known point instead", &
            failures)
        call expect(minloc(score, dim=1) == 2, &
            "the straddle rule picks the informative point near the contour", &
            failures)

        ! kappa zero collapses the rule to nearest-the-contour, which is a
        ! reachable and useless limit worth pinning.
        call fortbo_straddle_score(mean, sd, threshold, 0.0_dp, score, status)
        call expect(minloc(score, dim=1) == 1, &
            "kappa zero reduces the rule to sampling nearest the contour", &
            failures)
    end subroutine check_straddle_balances_its_two_terms

    subroutine check_maximin(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: grid(4, 2), clustered(4, 2)
        real(dp) :: distance, spread_score, clustered_score, brute
        integer :: i, j

        ! Corners of the unit square: every neighbouring pair is one apart, and
        ! the diagonals are further, so the minimum is exactly one.
        grid = reshape([0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, &
            0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], [4, 2])
        call fortbo_minimum_pairwise_distance(grid, distance, status)
        call expect(status%code == FORTNUM_OK, "the minimum distance computes", &
            failures)
        call expect(abs(distance - 1.0_dp) < 1.0e-14_dp, &
            "the corners of a unit square are exactly one apart", failures)

        ! Exhaustive recomputation, independently.
        brute = huge(1.0_dp)
        do i = 1, 3
            do j = i + 1, 4
                brute = min(brute, sqrt(sum((grid(i, :) - grid(j, :))**2)))
            end do
        end do
        call expect(abs(distance - brute) == 0.0_dp, &
            "the minimum distance matches an exhaustive sweep", failures)

        ! A spread design must score better (lower) than a clustered one.
        clustered = reshape([0.0_dp, 0.01_dp, 0.02_dp, 0.03_dp, &
            0.0_dp, 0.01_dp, 0.0_dp, 0.01_dp], [4, 2])
        call fortbo_maximin_score(grid, spread_score, status)
        call fortbo_maximin_score(clustered, clustered_score, status)
        call expect(spread_score < clustered_score, &
            "a spread design scores better than a clustered one", failures)
        call expect(spread_score == -1.0_dp, &
            "the score is the negated minimum distance", failures)
    end subroutine check_maximin

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: score(2), probability(2), distance
        real(dp) :: single(1, 2)

        call fortbo_active_learning_score([-1.0_dp, 1.0_dp], score, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative variance is refused", failures)

        call fortbo_straddle_score([0.0_dp, 0.0_dp], [1.0_dp, 1.0_dp], 0.0_dp, &
            -1.0_dp, score, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative kappa is refused, since it would prefer certainty", &
            failures)

        call fortbo_level_set_probability([0.0_dp, 0.0_dp], [1.0_dp, -1.0_dp], &
            0.0_dp, probability, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative standard deviation is refused", failures)

        single = 0.0_dp
        call fortbo_minimum_pairwise_distance(single, distance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a one-point design has no pairwise distance", failures)
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

end program test_active
