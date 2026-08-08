program test_knowledge_gradient
    !! BO1: knowledge gradient.
    !!
    !! Oracles:
    !!
    !!   * the exact envelope integral is checked against Monte Carlo over the
    !!     same standard normal, with the tolerance taken from the sampling
    !!     standard error rather than tuned. That is the whole algorithm, and
    !!     the reason it is written exactly rather than sampled is precisely
    !!     that sampling noise would break replay;
    !!   * degenerate cases have answers that can be stated: one line, parallel
    !!     lines, a line that never reaches the envelope;
    !!   * KG itself is checked against a direct simulation of its definition —
    !!     draw a fantasized observation, condition, re-minimize — which shares
    !!     no code with the closed form;
    !!   * the properties that make KG a sound acquisition are checked as
    !!     behavior: it is never negative, it is exactly zero when the
    !!     observation cannot teach anything, and it prefers sampling a point
    !!     that informs the decision over one that does not.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_knowledge_gradient, only: fortbo_knowledge_gradient_value, &
        fortbo_expected_minimum_of_lines, fortbo_batch_knowledge_gradient
    use fortbo_test_posteriors, only: demo_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_envelope_against_monte_carlo(failures)
    call check_envelope_degenerate_cases(failures)
    call check_kg_against_simulation(failures)
    call check_kg_is_non_negative(failures)
    call check_kg_rewards_informative_samples(failures)
    call check_batch_reduces_to_sequential(failures)
    call check_batch_is_monotone(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_knowledge_gradient: PASS"
    else
        print *, "test_knowledge_gradient: FAIL", failures
        error stop 1
    end if

contains

    !! The closed form against simulation of the very quantity it computes.
    subroutine check_envelope_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        real(dp) :: intercepts(6), slopes(6)
        real(dp) :: exact, estimate, draw, sample_min, total, total_squares
        real(dp) :: standard_error
        integer :: k, case_index

        do case_index = 1, 3
            select case (case_index)
            case (1)
                intercepts = [1.0_dp, 0.5_dp, 2.0_dp, -0.5_dp, 0.0_dp, 1.5_dp]
                slopes = [-2.0_dp, -0.5_dp, 1.0_dp, 0.25_dp, 2.0_dp, -1.0_dp]
            case (2)
                ! Several lines that never reach the envelope.
                intercepts = [0.0_dp, 5.0_dp, 6.0_dp, 7.0_dp, 8.0_dp, 0.1_dp]
                slopes = [0.0_dp, 0.1_dp, 0.2_dp, 0.3_dp, 0.4_dp, 0.05_dp]
            case (3)
                ! Steep slopes, so the breakpoints sit far into the tails.
                intercepts = [0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, -10.0_dp, 3.0_dp]
                slopes = [-30.0_dp, 0.0_dp, 30.0_dp, -1.0_dp, 1.0_dp, 0.5_dp]
            end select

            call fortbo_expected_minimum_of_lines(intercepts, slopes, exact, status)
            call expect(status%code == FORTNUM_OK, "the envelope integral computes", &
                failures)

            call rng_seed(generator, int(13579, int64), status)
            total = 0.0_dp
            total_squares = 0.0_dp
            do k = 1, n_samples
                call rng_normal(generator, draw)
                sample_min = minval(intercepts + slopes*draw)
                total = total + sample_min
                total_squares = total_squares + sample_min**2
            end do
            estimate = total/real(n_samples, dp)
            standard_error = sqrt(max(total_squares/real(n_samples, dp) &
                - estimate**2, 0.0_dp)/real(n_samples, dp))
            call expect(abs(exact - estimate) < 5.0_dp*standard_error, &
                "the exact envelope matches simulation of the same minimum", &
                failures)
        end do
    end subroutine check_envelope_against_monte_carlo

    !! Cases whose answers can be written down.
    subroutine check_envelope_degenerate_cases(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: expectation

        ! One line: the expectation is its intercept, since E[Z] is zero.
        call fortbo_expected_minimum_of_lines([2.5_dp], [7.0_dp], expectation, status)
        call expect(abs(expectation - 2.5_dp) < 1.0e-14_dp, &
            "a single line contributes only its intercept", failures)

        ! Parallel lines: only the lowest survives, and the answer is again an
        ! intercept. A sweep that mishandled the slope tie would divide by zero.
        call fortbo_expected_minimum_of_lines([3.0_dp, 1.0_dp, 2.0_dp], &
            [0.5_dp, 0.5_dp, 0.5_dp], expectation, status)
        call expect(status%code == FORTNUM_OK .and. &
            abs(expectation - 1.0_dp) < 1.0e-14_dp, &
            "parallel lines keep only the lowest", failures)

        ! min(Z, -Z) = -|Z|, whose expectation is -sqrt(2/pi).
        call fortbo_expected_minimum_of_lines([0.0_dp, 0.0_dp], [1.0_dp, -1.0_dp], &
            expectation, status)
        call expect(abs(expectation + sqrt(2.0_dp/(4.0_dp*atan(1.0_dp)))) &
            < 1.0e-12_dp, "min(Z, -Z) integrates to -sqrt(2/pi)", failures)

        ! A line far above the others never appears and cannot change the value.
        call fortbo_expected_minimum_of_lines([0.0_dp, 0.0_dp], [1.0_dp, -1.0_dp], &
            expectation, status)
        block
            real(dp) :: with_extra
            call fortbo_expected_minimum_of_lines([0.0_dp, 0.0_dp, 50.0_dp], &
                [1.0_dp, -1.0_dp, 0.0_dp], with_extra, status)
            call expect(abs(with_extra - expectation) < 1.0e-12_dp, &
                "a dominated line changes nothing", failures)
        end block
    end subroutine check_envelope_degenerate_cases

    !! Simulate the definition: fantasize an observation, condition the
    !! reference means on it, re-minimize, average. This shares no code with the
    !! closed form beyond the posterior itself.
    subroutine check_kg_against_simulation(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 200000
        integer, parameter :: m = 5
        real(dp) :: reference(m, 1), candidate(1)
        real(dp) :: joint(m + 1, 1), mean(m + 1), variance(m + 1)
        real(dp) :: covariance(m + 1, m + 1)
        real(dp) :: tilde(m), draw, conditioned(m)
        real(dp) :: exact, estimate, total, total_squares, standard_error
        real(dp) :: denominator, sample_value
        real(dp), parameter :: noise = 0.05_dp
        integer :: k, i

        posterior%dimension = 1
        do i = 1, m
            reference(i, 1) = -1.0_dp + 0.5_dp*real(i - 1, dp)
        end do
        candidate = [0.35_dp]

        call fortbo_knowledge_gradient_value(posterior, candidate, reference, noise, &
            exact, status)
        call expect(status%code == FORTNUM_OK, "knowledge gradient computes", &
            failures)

        joint(:m, :) = reference
        joint(m + 1, :) = candidate
        call posterior%moments(joint, mean, variance, status)
        call posterior%covariance(joint, covariance, status)
        denominator = sqrt(variance(m + 1) + noise)
        do i = 1, m
            tilde(i) = covariance(i, m + 1)/denominator
        end do

        call rng_seed(generator, int(24680, int64), status)
        total = 0.0_dp
        total_squares = 0.0_dp
        do k = 1, n_samples
            call rng_normal(generator, draw)
            conditioned = mean(:m) + tilde*draw
            sample_value = minval(mean(:m)) - minval(conditioned)
            total = total + sample_value
            total_squares = total_squares + sample_value**2
        end do
        estimate = total/real(n_samples, dp)
        standard_error = sqrt(max(total_squares/real(n_samples, dp) - estimate**2, &
            0.0_dp)/real(n_samples, dp))
        call expect(abs(exact - estimate) < 5.0_dp*standard_error, &
            "knowledge gradient matches a direct simulation of its definition", &
            failures)
        call expect(exact > 0.0_dp, &
            "sampling somewhere informative has positive value", failures)
    end subroutine check_kg_against_simulation

    !! Information cannot make the best decision worse in expectation. This is a
    !! theorem, so it must hold at every candidate, not just convenient ones.
    subroutine check_kg_is_non_negative(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: reference(4, 1), candidate(1), value
        integer :: i, k
        logical :: non_negative

        posterior%dimension = 1
        do i = 1, 4
            reference(i, 1) = -0.9_dp + 0.6_dp*real(i - 1, dp)
        end do

        non_negative = .true.
        do k = -30, 30
            candidate = [0.1_dp*real(k, dp)]
            call fortbo_knowledge_gradient_value(posterior, candidate, reference, &
                0.01_dp, value, status)
            if (status%code /= FORTNUM_OK) non_negative = .false.
            if (value < 0.0_dp) non_negative = .false.
        end do
        call expect(non_negative, &
            "knowledge gradient is never negative anywhere", failures)

        ! Enormous observation noise teaches almost nothing, so the value of the
        ! sample must collapse toward zero.
        candidate = [0.0_dp]
        call fortbo_knowledge_gradient_value(posterior, candidate, reference, &
            1.0e12_dp, value, status)
        call expect(value < 1.0e-5_dp, &
            "an uninformative measurement is worth almost nothing", failures)
    end subroutine check_kg_is_non_negative

    !! The behavioral claim that separates KG from expected improvement: a
    !! sample is valuable when it *informs the decision*, and a point the
    !! reference set is blind to is worth nothing however uncertain it is.
    subroutine check_kg_rewards_informative_samples(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: reference(3, 1), near(1), far(1)
        real(dp) :: near_value, far_value

        posterior%dimension = 1
        reference(:, 1) = [-0.5_dp, 0.0_dp, 0.5_dp]

        ! A candidate sitting among the reference points is correlated with them
        ! and so teaches the decision something.
        near = [0.25_dp]
        call fortbo_knowledge_gradient_value(posterior, near, reference, 0.01_dp, &
            near_value, status)

        ! One far outside the kernel's reach is uncertain but uninformative: the
        ! reference means barely move whatever it turns out to be. Expected
        ! improvement would happily chase this point's variance.
        far = [40.0_dp]
        call fortbo_knowledge_gradient_value(posterior, far, reference, 0.01_dp, &
            far_value, status)

        call expect(near_value > far_value, &
            "a sample that informs the decision beats one that does not", failures)
        call expect(far_value < 1.0e-8_dp, &
            "a sample uncorrelated with every reference point is worthless", &
            failures)
    end subroutine check_kg_rewards_informative_samples

    !! The anchor tying the batch estimator to the exact one: with a single
    !! fantasy slot, d-KG *is* the sequential knowledge gradient, so the Monte
    !! Carlo estimate must land on the closed-form envelope value.
    subroutine check_batch_reduces_to_sequential(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: m = 6, n_samples = 400000
        real(dp) :: reference_mean(m), loading(m, 1)
        real(dp) :: batch_value, exact, expected_min, standard_error
        integer :: k

        reference_mean = [0.5_dp, -0.2_dp, 1.0_dp, 0.1_dp, -0.6_dp, 0.3_dp]
        loading(:, 1) = [0.4_dp, -0.7_dp, 0.2_dp, 0.9_dp, -0.1_dp, 0.5_dp]

        ! The closed form, through the envelope.
        call fortbo_expected_minimum_of_lines(reference_mean, loading(:, 1), &
            expected_min, status)
        exact = minval(reference_mean) - expected_min

        call rng_seed(generator, int(606060, int64), status)
        call fortbo_batch_knowledge_gradient(reference_mean, loading, n_samples, &
            generator, batch_value, status)
        call expect(status%code == FORTNUM_OK, "the batch estimator evaluates", &
            failures)

        ! The estimator averages a bounded piecewise-linear function; its spread
        ! is at most the range of the reference means plus a few loadings.
        standard_error = (maxval(reference_mean) - minval(reference_mean) &
            + maxval(abs(loading)))/sqrt(real(n_samples, dp))
        call expect(abs(batch_value - exact) < 6.0_dp*standard_error, &
            "a batch of one reproduces the exact sequential value", failures)
        call expect(batch_value > 0.0_dp, &
            "a batch that can teach something has positive value", failures)
    end subroutine check_batch_reduces_to_sequential

    !! Adding a fantasy slot cannot reduce the value: the extra observation can
    !! always be ignored. A sign error or a mis-shaped loading loop breaks this
    !! without needing a reference value.
    subroutine check_batch_is_monotone(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: m = 5, n_samples = 200000
        real(dp) :: reference_mean(m), small(m, 1), large(m, 2)
        real(dp) :: small_value, large_value, zero_value
        real(dp) :: empty_loading(m, 1)

        reference_mean = [0.4_dp, -0.1_dp, 0.9_dp, 0.2_dp, -0.5_dp]
        small(:, 1) = [0.3_dp, -0.5_dp, 0.2_dp, 0.6_dp, -0.2_dp]
        large(:, 1) = small(:, 1)
        large(:, 2) = [-0.4_dp, 0.3_dp, -0.6_dp, 0.1_dp, 0.5_dp]

        call rng_seed(generator, int(717272, int64), status)
        call fortbo_batch_knowledge_gradient(reference_mean, small, n_samples, &
            generator, small_value, status)
        call rng_seed(generator, int(717272, int64), status)
        call fortbo_batch_knowledge_gradient(reference_mean, large, n_samples, &
            generator, large_value, status)
        call expect(large_value > small_value, &
            "a larger batch is worth at least as much", failures)

        ! A batch that moves nothing teaches nothing.
        empty_loading = 0.0_dp
        call rng_seed(generator, int(717272, int64), status)
        call fortbo_batch_knowledge_gradient(reference_mean, empty_loading, 1000, &
            generator, zero_value, status)
        call expect(zero_value == 0.0_dp, &
            "a batch that shifts no reference mean is worthless", failures)
    end subroutine check_batch_is_monotone

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(moments_only_posterior_t) :: plain
        type(fortnum_status_t) :: status
        real(dp) :: reference(2, 1), candidate(1), value
        real(dp) :: wide(2, 3), expectation

        posterior%dimension = 1
        plain%dimension = 1
        reference(:, 1) = [0.0_dp, 0.5_dp]
        candidate = [0.25_dp]

        call fortbo_knowledge_gradient_value(posterior, candidate, reference, &
            -1.0_dp, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative noise variance is refused", failures)

        call fortbo_knowledge_gradient_value(posterior, candidate, wide, 0.1_dp, &
            value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a reference set of the wrong width is refused", failures)

        ! A posterior with marginal moments but no joint covariance cannot
        ! express how a sample here moves the mean there, which is the whole
        ! quantity. It must refuse rather than assume independence.
        call fortbo_knowledge_gradient_value(plain, candidate, reference, 0.1_dp, &
            value, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a posterior without joint covariance refuses by capability", failures)

        call fortbo_expected_minimum_of_lines([1.0_dp, 2.0_dp], [1.0_dp], &
            expectation, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched line arrays are refused", failures)

        block
            type(rng_t) :: generator
            real(dp) :: loading(2, 1), batch_value
            loading = 0.0_dp
            call rng_seed(generator, int(1, int64), status)
            call fortbo_batch_knowledge_gradient([1.0_dp, 2.0_dp, 3.0_dp], &
                loading, 10, generator, batch_value, status)
            call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                "a loading that does not match the reference set is refused", &
                failures)
            call fortbo_batch_knowledge_gradient([1.0_dp, 2.0_dp], loading, 0, &
                generator, batch_value, status)
            call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                "a zero sample count is refused", failures)
        end block
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

end program test_knowledge_gradient
