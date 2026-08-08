program test_batch
    !! BO1: batch acquisitions qEI, qNEI, qUCB.
    !!
    !! Oracles:
    !!
    !!   * a batch of one must reproduce the sequential acquisition exactly.
    !!     That is the only anchor tying these estimators to the analytic forms
    !!     already validated elsewhere, and it fails for any error in the
    !!     threshold, the sign, or the averaging;
    !!   * the behavior batch acquisitions exist for: duplicating a point must
    !!     add nothing, and a diverse pair must beat a clustered one. Both are
    !!     measured, and both are wrong under independent marginals — which is
    !!     precisely the mistake a joint draw prevents;
    !!   * monotonicity in the batch: adding a point can never lower `E[max]`;
    !!   * qUCB's beta term must not average away, which is what the absolute
    !!     value is for.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_batch, only: fortbo_batch_samples_t, fortbo_qei, fortbo_qnei, &
        fortbo_qucb
    use fortbo_acquisition, only: fortbo_expected_improvement
    use fortbo_test_posteriors, only: demo_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_single_point_matches_sequential(failures)
    call check_duplicates_add_nothing(failures)
    call check_diversity_is_rewarded(failures)
    call check_monotone_in_the_batch(failures)
    call check_qnei_couples_its_incumbent(failures)
    call check_qucb(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_batch: PASS"
    else
        print *, "test_batch: FAIL", failures
        error stop 1
    end if

contains

    !! A batch of one is ordinary EI. With enough samples the Monte Carlo
    !! estimate must land on the analytic value within sampling error, computed
    !! here from the estimator's own spread rather than guessed.
    subroutine check_single_point_matches_sequential(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: samples
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(1, 1), mean(1), variance(1)
        real(dp) :: estimate, analytic, standard_error, spread
        integer, parameter :: n_samples = 200000
        integer :: s
        real(dp), parameter :: best = 0.4_dp

        posterior%dimension = 1
        points(1, 1) = 0.3_dp
        call rng_seed(generator, int(191919, int64), status)
        call samples%generate(posterior, points, n_samples, generator, status)
        call expect(status%code == FORTNUM_OK, "the joint samples generate", failures)

        call fortbo_qei(samples, best, 0.0_dp, estimate, status)
        call expect(status%code == FORTNUM_OK, "qEI evaluates", failures)

        call posterior%moments(points, mean, variance, status)
        call fortbo_expected_improvement(mean(1), sqrt(variance(1)), best, &
            0.0_dp, analytic)

        spread = 0.0_dp
        do s = 1, n_samples
            spread = spread + (max(best - samples%draws(1, s), 0.0_dp) - estimate)**2
        end do
        standard_error = sqrt(spread/real(n_samples, dp)/real(n_samples, dp))
        call expect(abs(estimate - analytic) < 5.0_dp*standard_error, &
            "a batch of one reproduces analytic expected improvement", failures)
    end subroutine check_single_point_matches_sequential

    !! The property that makes a batch acquisition a batch acquisition. Under
    !! independent marginals two copies of a point would score roughly twice as
    !! well as one; under a joint draw they must score identically, because the
    !! two copies are the same random variable.
    subroutine check_duplicates_add_nothing(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: single, doubled
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: one(1, 1), two(2, 1)
        real(dp) :: single_value, doubled_value
        integer, parameter :: n_samples = 40000

        posterior%dimension = 1
        one(1, 1) = 0.25_dp
        two(1, 1) = 0.25_dp
        two(2, 1) = 0.25_dp

        call rng_seed(generator, int(313131, int64), status)
        call single%generate(posterior, one, n_samples, generator, status)
        call rng_seed(generator, int(313131, int64), status)
        call doubled%generate(posterior, two, n_samples, generator, status)

        call fortbo_qei(single, 0.5_dp, 0.0_dp, single_value, status)
        call fortbo_qei(doubled, 0.5_dp, 0.0_dp, doubled_value, status)

        ! The duplicate is perfectly correlated with the original, so the two
        ! realizations coincide in every sample and the maximum is unchanged.
        !
        ! They coincide only to about sqrt(jitter), not to rounding: a
        ! covariance with a repeated point is singular, and the jitter that
        ! makes it factorizable is exactly what lets the copy wander. That
        ! wandering enters the draw through a Cholesky factor, so it scales as
        ! the square root of the jitter rather than the jitter itself, which is
        ! why the bound here is 1e-4 and not 1e-10.
        call expect(maxval(abs(doubled%draws(1, :) - doubled%draws(2, :))) &
            < 1.0e-4_dp, "a duplicated point draws essentially the same value", &
            failures)
        call expect(abs(doubled_value - single_value) &
            < 0.02_dp*max(single_value, 1.0e-12_dp), &
            "duplicating a point adds essentially nothing to qEI", failures)
    end subroutine check_duplicates_add_nothing

    !! Two well-separated points must beat two clustered ones at the same
    !! budget. This is the diversity that falls out of the joint draw, with no
    !! explicit diversity term anywhere.
    subroutine check_diversity_is_rewarded(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: clustered, spread_out
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: near(2, 1), far(2, 1)
        real(dp) :: near_value, far_value
        integer, parameter :: n_samples = 60000

        posterior%dimension = 1
        near(1, 1) = 0.0_dp
        near(2, 1) = 0.01_dp
        far(1, 1) = -0.8_dp
        far(2, 1) = 0.8_dp

        call rng_seed(generator, int(626262, int64), status)
        call clustered%generate(posterior, near, n_samples, generator, status)
        call rng_seed(generator, int(626262, int64), status)
        call spread_out%generate(posterior, far, n_samples, generator, status)

        call fortbo_qei(clustered, 0.5_dp, 0.0_dp, near_value, status)
        call fortbo_qei(spread_out, 0.5_dp, 0.0_dp, far_value, status)
        call expect(far_value > near_value, &
            "a diverse batch beats a clustered one of the same size", failures)
    end subroutine check_diversity_is_rewarded

    !! `E[max]` over a larger set cannot be smaller. A sign error or a
    !! mis-initialized running maximum breaks this immediately.
    subroutine check_monotone_in_the_batch(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: small, large
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: two(2, 1), three(3, 1)
        real(dp) :: small_value, large_value
        integer, parameter :: n_samples = 40000

        posterior%dimension = 1
        two(:, 1) = [-0.4_dp, 0.4_dp]
        three(:, 1) = [-0.4_dp, 0.4_dp, 0.9_dp]

        call rng_seed(generator, int(858585, int64), status)
        call small%generate(posterior, two, n_samples, generator, status)
        call rng_seed(generator, int(858585, int64), status)
        call large%generate(posterior, three, n_samples, generator, status)

        call fortbo_qei(small, 0.5_dp, 0.0_dp, small_value, status)
        call fortbo_qei(large, 0.5_dp, 0.0_dp, large_value, status)
        call expect(large_value >= small_value - 1.0e-9_dp, &
            "adding a point never lowers qEI", failures)

        ! qEI is non-negative by construction: the improvement is clipped at
        ! zero inside the expectation, not after averaging.
        call expect(small_value >= 0.0_dp .and. large_value >= 0.0_dp, &
            "qEI is never negative", failures)
    end subroutine check_monotone_in_the_batch

    !! qNEI must use the incumbent from the *same* sample index. Shuffling the
    !! observed columns changes the answer, which is the evidence that the
    !! coupling is real rather than incidental.
    subroutine check_qnei_couples_its_incumbent(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: samples
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 20000
        real(dp) :: points(2, 1)
        real(dp) :: observed(2, n_samples), reversed(2, n_samples)
        real(dp) :: coupled, shuffled, constant_value
        real(dp) :: fixed(1, n_samples)
        integer :: s

        posterior%dimension = 1
        points(:, 1) = [-0.3_dp, 0.3_dp]
        call rng_seed(generator, int(474747, int64), status)
        call samples%generate(posterior, points, n_samples, generator, status)

        ! Observed latent values that vary strongly across samples.
        do s = 1, n_samples
            observed(1, s) = 0.5_dp*sin(real(s, dp))
            observed(2, s) = 0.5_dp*cos(real(s, dp))
            reversed(:, n_samples - s + 1) = observed(:, s)
        end do

        call fortbo_qnei(samples, observed, coupled, status)
        call expect(status%code == FORTNUM_OK, "qNEI evaluates", failures)
        call fortbo_qnei(samples, reversed, shuffled, status)
        call expect(abs(coupled - shuffled) > 1.0e-6_dp, &
            "qNEI depends on which sample the incumbent came from", failures)

        ! A constant incumbent must reproduce qEI against that same value,
        ! which ties qNEI back to the estimator already checked above.
        fixed = 0.5_dp
        call fortbo_qnei(samples, fixed, constant_value, status)
        call fortbo_qei(samples, 0.5_dp, 0.0_dp, coupled, status)
        call expect(abs(constant_value - coupled) < 1.0e-12_dp, &
            "a constant incumbent reduces qNEI to qEI", failures)
    end subroutine check_qnei_couples_its_incumbent

    subroutine check_qucb(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortbo_batch_samples_t) :: samples
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), mean(2), variance(2)
        real(dp) :: optimistic, neutral
        integer, parameter :: n_samples = 60000

        posterior%dimension = 1
        points(:, 1) = [-0.5_dp, 0.5_dp]
        call rng_seed(generator, int(969696, int64), status)
        call samples%generate(posterior, points, n_samples, generator, status)
        call posterior%moments(points, mean, variance, status)

        call fortbo_qucb(samples, mean, 0.0_dp, neutral, status)
        call expect(status%code == FORTNUM_OK, "qUCB evaluates", failures)
        call expect(abs(neutral - minval(mean)) < 1.0e-12_dp, &
            "beta zero reduces qUCB to the best posterior mean", failures)

        call fortbo_qucb(samples, mean, 2.0_dp, optimistic, status)
        ! FortBO minimizes, so optimism means a *lower* score. Without the
        ! absolute value the beta term would average to zero and this would be
        ! indistinguishable from the neutral case.
        call expect(optimistic < neutral, &
            "a positive beta makes qUCB optimistic rather than averaging away", &
            failures)
    end subroutine check_qucb

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(moments_only_posterior_t) :: plain
        type(fortbo_batch_samples_t) :: samples, empty
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), mean(2), value
        real(dp) :: mismatched(2, 5)

        posterior%dimension = 1
        plain%dimension = 1
        points(:, 1) = [-0.2_dp, 0.2_dp]
        call rng_seed(generator, int(1, int64), status)

        ! A posterior with marginal moments only cannot express the correlation
        ! a batch acquisition is built on, and must refuse rather than pretend
        ! the points are independent.
        call samples%generate(plain, points, 10, generator, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a posterior without joint sampling refuses by capability", failures)

        call samples%generate(posterior, points, 0, generator, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero sample count is refused", failures)

        call fortbo_qei(empty, 1.0_dp, 0.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "scoring ungenerated samples is refused", failures)

        call samples%generate(posterior, points, 10, generator, status)
        call fortbo_qnei(samples, mismatched, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "observed draws of the wrong sample count are refused", failures)

        mean = 0.0_dp
        call fortbo_qucb(samples, mean, -1.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative beta is refused", failures)

        call fortbo_qucb(samples, [0.0_dp], 1.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mean of the wrong length is refused", failures)
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

end program test_batch
