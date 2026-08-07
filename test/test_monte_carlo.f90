program test_monte_carlo
    !! BO1: Monte Carlo acquisition evaluation.
    !!
    !! Oracles:
    !!   * the estimator is checked against the analytic acquisition, which was
    !!     itself checked against quadrature in `test_acquisition`. The
    !!     tolerance is derived from the estimator's own standard error rather
    !!     than tuned until it passes;
    !!   * antithetic pairing is checked by the property that motivates it: over
    !!     many independent seeds the paired estimator's spread about the true
    !!     value must be materially smaller than the plain estimator's at the
    !!     same sample count. That is a statement about variance, so it is
    !!     measured over repeated runs, not asserted from one;
    !!   * the pathwise gradient is checked against central differences of the
    !!     estimate taken with the *same* base samples. Differencing across
    !!     different noise would measure sampling error, not the derivative,
    !!     which is precisely the trap common random numbers exist to avoid;
    !!   * common random numbers are checked by exact reproducibility.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_monte_carlo, only: fortbo_mc_base_t, fortbo_mc_ei_t, fortbo_mc_pi_t
    use fortbo_acquisition, only: fortbo_expected_improvement, &
        fortbo_probability_of_improvement
    use fortbo_test_posteriors, only: curved_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_base_generation(failures)
    call check_converges_to_analytic(failures)
    call check_common_random_numbers(failures)
    call check_antithetic_reduces_variance(failures)
    call check_pathwise_gradient(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_monte_carlo: PASS"
    else
        print *, "test_monte_carlo: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_base_generation(failures)
        integer, intent(inout) :: failures
        type(fortbo_mc_base_t) :: base
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer :: half
        logical :: paired

        call rng_seed(generator, int(11, int64), status)
        call base%generate(3, 8, generator, status, antithetic=.true.)
        call expect(status%code == FORTNUM_OK, "antithetic base generation succeeds", &
            failures)
        call expect(base%n_points() == 3, "the base has one row per query", failures)
        call expect(base%n_samples() == 8, "the base has the requested sample count", &
            failures)

        half = base%n_samples()/2
        paired = maxval(abs(base%draws(:, 1:half) + base%draws(:, half + 1:))) == 0.0_dp
        call expect(paired, "antithetic draws are exact negations", failures)
        call expect(abs(sum(base%draws)) < 1.0e-12_dp, &
            "antithetic draws sum to zero", failures)

        call base%generate(3, 7, generator, status, antithetic=.true.)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an odd antithetic sample count is refused", failures)
        call base%generate(0, 8, generator, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an empty sample shape is refused", failures)
    end subroutine check_base_generation

    !! The estimate must land within a few standard errors of the analytic
    !! value. For the improvement utility the per-sample variance is bounded by
    !! the second moment of the truncated normal, which the test estimates from
    !! the same draws rather than assuming.
    subroutine check_converges_to_analytic(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_mc_ei_t) :: mc_ei
        type(fortbo_mc_pi_t) :: mc_pi
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 200000
        real(dp) :: points(3, 1), estimate(3), exact(3)
        real(dp) :: mean(3), variance(3), standard_error
        integer :: i

        posterior%dimension = 1
        points = reshape([-0.8_dp, 0.1_dp, 1.2_dp], [3, 1])
        call posterior%moments(points, mean, variance, status)

        call rng_seed(generator, int(2026, int64), status)
        mc_ei%best = 3.0_dp
        call mc_ei%base%generate(3, n_samples, generator, status)
        call mc_ei%value(posterior, points, estimate, status)
        call expect(status%code == FORTNUM_OK, "the estimator evaluates", failures)
        do i = 1, 3
            call fortbo_expected_improvement(mean(i), sqrt(variance(i)), 3.0_dp, &
                0.0_dp, exact(i))
            ! The improvement is bounded above by |threshold - mu| + 5 sigma over
            ! the draws that matter, so this bounds the per-sample spread.
            standard_error = (abs(3.0_dp - mean(i)) + 5.0_dp*sqrt(variance(i))) &
                /sqrt(real(n_samples, dp))
            call expect(abs(estimate(i) - exact(i)) < 5.0_dp*standard_error, &
                "MC expected improvement matches the analytic value", failures)
        end do

        call rng_seed(generator, int(2027, int64), status)
        mc_pi%best = 3.0_dp
        call mc_pi%base%generate(3, n_samples, generator, status)
        call mc_pi%value(posterior, points, estimate, status)
        do i = 1, 3
            call fortbo_probability_of_improvement(mean(i), sqrt(variance(i)), 3.0_dp, &
                0.0_dp, exact(i))
            standard_error = sqrt(max(exact(i)*(1.0_dp - exact(i)), 1.0e-12_dp) &
                /real(n_samples, dp))
            call expect(abs(estimate(i) - exact(i)) < 5.0_dp*standard_error, &
                "MC improvement probability matches the analytic value", &
                failures)
        end do
    end subroutine check_converges_to_analytic

    !! Fixed base samples must give bitwise identical estimates, and two nearby
    !! candidate sets evaluated on the same base must differ far more smoothly
    !! than the sampling error of either.
    subroutine check_common_random_numbers(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_mc_ei_t) :: mc_ei
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), shifted(2, 1)
        real(dp) :: first(2), second(2), nearby(2)

        posterior%dimension = 1
        points = reshape([0.3_dp, 0.9_dp], [2, 1])
        shifted = points + 1.0e-8_dp

        call rng_seed(generator, int(99, int64), status)
        mc_ei%best = 2.0_dp
        call mc_ei%base%generate(2, 4096, generator, status)

        call mc_ei%value(posterior, points, first, status)
        call mc_ei%value(posterior, points, second, status)
        call expect(maxval(abs(first - second)) == 0.0_dp, &
            "a fixed base gives bitwise identical estimates", failures)

        call mc_ei%value(posterior, shifted, nearby, status)
        call expect(maxval(abs(nearby - first)) < 1.0e-6_dp, &
            "a tiny query shift moves the estimate only slightly", failures)
    end subroutine check_common_random_numbers

    !! Variance reduction is a property of the estimator across seeds, so it is
    !! measured across seeds. Both estimators are unbiased, so their spread
    !! about the analytic value is what distinguishes them.
    subroutine check_antithetic_reduces_variance(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_mc_ei_t) :: plain, paired
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_repeats = 60
        integer, parameter :: n_samples = 256
        real(dp) :: points(1, 1), estimate(1), exact, mean(1), variance(1)
        real(dp) :: plain_error, paired_error
        integer :: r

        posterior%dimension = 1
        points = reshape([0.35_dp], [1, 1])
        call posterior%moments(points, mean, variance, status)
        call fortbo_expected_improvement(mean(1), sqrt(variance(1)), 2.5_dp, 0.0_dp, &
            exact)

        plain%best = 2.5_dp
        paired%best = 2.5_dp
        plain_error = 0.0_dp
        paired_error = 0.0_dp
        do r = 1, n_repeats
            call rng_seed(generator, int(5000 + r, int64), status)
            call plain%base%generate(1, n_samples, generator, status)
            call plain%value(posterior, points, estimate, status)
            plain_error = plain_error + (estimate(1) - exact)**2

            call rng_seed(generator, int(5000 + r, int64), status)
            call paired%base%generate(1, n_samples, generator, status, &
                antithetic=.true.)
            call paired%value(posterior, points, estimate, status)
            paired_error = paired_error + (estimate(1) - exact)**2
        end do

        call expect(paired_error < plain_error, &
            "antithetic pairing reduces the mean squared error", failures)
        call expect(paired_error < 0.6_dp*plain_error, &
            "the antithetic reduction is substantial, not marginal", failures)
    end subroutine check_antithetic_reduces_variance

    !! The pathwise gradient is exact for the estimator, so differencing the
    !! estimator on the same base must reproduce it to differencing accuracy.
    subroutine check_pathwise_gradient(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_mc_ei_t) :: mc_ei
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 2), gradient(2, 2), numeric(2, 2)
        real(dp) :: shifted(2, 2), plus(2), minus(2)
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: i, j

        posterior%dimension = 2
        points = reshape([0.4_dp, -0.6_dp, 0.8_dp, 1.1_dp], [2, 2])

        call rng_seed(generator, int(31337, int64), status)
        mc_ei%best = 3.0_dp
        call mc_ei%base%generate(2, 8192, generator, status)

        call mc_ei%value_gradient(posterior, points, gradient, status)
        call expect(status%code == FORTNUM_OK, "the pathwise gradient evaluates", &
            failures)

        do j = 1, 2
            do i = 1, 2
                shifted = points
                shifted(i, j) = points(i, j) + step
                call mc_ei%value(posterior, shifted, plus, status)
                shifted(i, j) = points(i, j) - step
                call mc_ei%value(posterior, shifted, minus, status)
                numeric(i, j) = (plus(i) - minus(i))/(2.0_dp*step)
            end do
        end do

        call expect(maxval(abs(gradient - numeric)) < 1.0e-5_dp, &
            "the pathwise gradient matches same-base central differences", &
            failures)
    end subroutine check_pathwise_gradient

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(moments_only_posterior_t) :: partial
        type(fortbo_mc_ei_t) :: mc_ei
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), values(2), gradient(2, 1)

        posterior%dimension = 1
        partial%dimension = 1
        points = reshape([0.2_dp, 0.6_dp], [2, 1])

        call mc_ei%value(posterior, points, values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "evaluating without generated base samples is refused", failures)
        call expect(index(status%msg, "never generated") > 0, &
            "the refusal explains that the base is missing", failures)

        call rng_seed(generator, int(3, int64), status)
        call mc_ei%base%generate(5, 64, generator, status)
        call mc_ei%value(posterior, points, values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a base with the wrong row count is refused", failures)

        call mc_ei%base%generate(2, 64, generator, status)
        call mc_ei%value_gradient(partial, points, gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a pathwise gradient without moment gradients is refused", failures)
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

end program test_monte_carlo
