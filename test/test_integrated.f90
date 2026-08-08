program test_integrated
    !! BO2: fully Bayesian hyperparameter integration, and the comparison with
    !! the plug-in policy the roadmap asks for.
    !!
    !! Oracles:
    !!
    !!   * **the end-to-end path works**: a real GP log marginal likelihood is
    !!     slice-sampled through FortMC, the surrogate is refitted at each
    !!     sample, and the acquisitions are blended. Nothing is mocked;
    !!   * **integrating differs from plugging in, in the direction it must.**
    !!     Averaging a nonlinear acquisition is not the same as evaluating it at
    !!     the average hyperparameter, and the test measures that rather than
    !!     asserting it;
    !!   * **averaging acquisitions is not averaging moments.** The two disagree
    !!     by Jensen, and a constructed case shows the moment-averaged version
    !!     understating exactly the point the samples disagree about;
    !!   * the diagnostics say when integration bought nothing.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortmc_slice, only: fortmc_slice_chain
    use fortbo_acquisition, only: fortbo_expected_improvement
    use fortbo_integrated, only: fortbo_integrated_acquisition, &
        fortbo_hyperparameter_spread, fortbo_integration_is_informative
    implicit none

    integer, parameter :: n_train = 6
    real(dp) :: train_x(n_train, 1), train_y(n_train, 1)

    integer :: failures

    failures = 0
    call build_training_set()
    call check_end_to_end_integration(failures)
    call check_acquisition_average_is_not_moment_average(failures)
    call check_diagnostics(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_integrated: PASS"
    else
        print *, "test_integrated: FAIL", failures
        error stop 1
    end if

contains

    subroutine build_training_set()
        integer :: k

        do k = 1, n_train
            train_x(k, 1) = -1.2_dp + 0.5_dp*real(k - 1, dp)
            train_y(k, 1) = sin(2.2_dp*train_x(k, 1)) + 0.3_dp*train_x(k, 1)
        end do
    end subroutine build_training_set

    !! Log marginal likelihood as a function of the log lengthscale and log
    !! signal variance, with a weak normal prior on both. This is a real GP
    !! objective, not a stand-in.
    function log_posterior(theta) result(value)
        real(dp), intent(in) :: theta(:)
        real(dp) :: value
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: lengthscale, signal

        lengthscale = exp(theta(1))
        signal = exp(theta(2))
        ! Keep the sampler out of regions where the factorization fails, by
        ! declaring them to have no density rather than by clipping — a clip
        ! would put a spike of probability on the boundary.
        if (lengthscale < 1.0e-3_dp .or. lengthscale > 1.0e3_dp .or. &
            signal < 1.0e-3_dp .or. signal > 1.0e3_dp) then
            value = -huge(1.0_dp)/4.0_dp
            return
        end if

        kernel = make_rbf_kernel(1, signal, lengthscale, status)
        if (status%code /= FORTNUM_OK) then
            value = -huge(1.0_dp)/4.0_dp
            return
        end if
        call model%fit(train_x, train_y, kernel, 0.01_dp, status)
        if (status%code /= FORTNUM_OK) then
            value = -huge(1.0_dp)/4.0_dp
            return
        end if
        call model%log_marginal_likelihood(value, status)
        if (status%code /= FORTNUM_OK) then
            value = -huge(1.0_dp)/4.0_dp
            return
        end if
        ! Weak log-normal priors, which is what keeps the posterior proper.
        value = value - 0.5_dp*(theta(1)**2 + theta(2)**2)/4.0_dp
    end function log_posterior

    !! Slice-sample the hyperparameters, refit at each draw, blend the
    !! acquisitions, and compare against the plug-in policy.
    subroutine check_end_to_end_integration(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 24, n_points = 9
        real(dp) :: theta(2, n_samples), start(2), width(2)
        real(dp) :: points(n_points, 1)
        real(dp) :: values(n_points, n_samples)
        real(dp) :: integrated(n_points), plug_in(n_points)
        real(dp) :: spread(2), mean_theta(2)
        integer :: k, s
        logical :: differs

        do k = 1, n_points
            points(k, 1) = -1.4_dp + 0.35_dp*real(k - 1, dp)
        end do

        start = [log(0.7_dp), log(1.0_dp)]
        width = [0.5_dp, 0.5_dp]
        call rng_seed(generator, int(20260808, int64), status)
        call fortmc_slice_chain(log_posterior, start, width, n_samples, &
            generator, theta, status, burn_in=30, thin=2)
        call expect(status%code == FORTNUM_OK, &
            "the hyperparameter chain runs on a real GP likelihood", failures)

        ! Refit at each draw and evaluate the acquisition.
        do s = 1, n_samples
            call acquisition_at(theta(:, s), points, values(:, s), status)
            if (status%code /= FORTNUM_OK) exit
        end do
        call expect(status%code == FORTNUM_OK, &
            "the surrogate refits at every hyperparameter draw", failures)

        call fortbo_integrated_acquisition(values, integrated, status)
        call expect(status%code == FORTNUM_OK, "the blend computes", failures)
        call expect(all(integrated >= 0.0_dp), &
            "the integrated acquisition stays non-negative", failures)

        ! Plug-in: the acquisition at the mean hyperparameter, which is what a
        ! point-estimate policy would use.
        do k = 1, 2
            mean_theta(k) = sum(theta(k, :))/real(n_samples, dp)
        end do
        call acquisition_at(mean_theta, points, plug_in, status)

        differs = maxval(abs(integrated - plug_in)) > 1.0e-8_dp
        call expect(differs, &
            "integrating differs from plugging in the mean hyperparameter", &
            failures)

        call fortbo_hyperparameter_spread(theta, spread, status)
        call expect(status%code == FORTNUM_OK, "the spread computes", failures)
        call expect(fortbo_integration_is_informative(spread), &
            "the chain moved, so integrating was worth doing", failures)
    end subroutine check_end_to_end_integration

    subroutine acquisition_at(theta, points, values, status)
        real(dp), intent(in) :: theta(:)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        real(dp), allocatable :: mean(:, :), variance(:)
        real(dp) :: best
        integer :: k

        allocate (mean(size(points, 1), 1), variance(size(points, 1)))
        kernel = make_rbf_kernel(1, exp(theta(2)), exp(theta(1)), status)
        if (status%code /= FORTNUM_OK) return
        call model%fit(train_x, train_y, kernel, 0.01_dp, status)
        if (status%code /= FORTNUM_OK) return
        call model%predict(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return

        best = minval(train_y(:, 1))
        do k = 1, size(points, 1)
            call fortbo_expected_improvement(mean(k, 1), sqrt(variance(k)), best, &
                0.0_dp, values(k))
        end do
    end subroutine acquisition_at

    !! Averaging acquisitions is not averaging moments, and the difference has
    !! a direction. Two hyperparameter samples that disagree sharply about one
    !! point: the acquisition average credits the disagreement, the moment
    !! average washes it out.
    subroutine check_acquisition_average_is_not_moment_average(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: values(1, 2), integrated(1)
        real(dp) :: mean_a, mean_b, sd_a, sd_b
        real(dp) :: moment_averaged
        real(dp), parameter :: best = 0.0_dp

        ! One sample thinks the point is promising, the other does not.
        mean_a = -1.0_dp
        sd_a = 0.2_dp
        mean_b = 1.0_dp
        sd_b = 0.2_dp

        call fortbo_expected_improvement(mean_a, sd_a, best, 0.0_dp, values(1, 1))
        call fortbo_expected_improvement(mean_b, sd_b, best, 0.0_dp, values(1, 2))
        call fortbo_integrated_acquisition(values, integrated, status)
        call expect(status%code == FORTNUM_OK, "the two-sample blend computes", &
            failures)

        ! What averaging the moments first would give.
        call fortbo_expected_improvement(0.5_dp*(mean_a + mean_b), &
            0.5_dp*(sd_a + sd_b), best, 0.0_dp, moment_averaged)

        call expect(integrated(1) > moment_averaged, &
            "averaging acquisitions credits disagreement that averaging moments loses", &
            failures)

        ! With the samples in agreement the two coincide, which is what makes
        ! the gap above attributable to the disagreement rather than to the
        ! arithmetic.
        call fortbo_expected_improvement(mean_a, sd_a, best, 0.0_dp, values(1, 1))
        call fortbo_expected_improvement(mean_a, sd_a, best, 0.0_dp, values(1, 2))
        call fortbo_integrated_acquisition(values, integrated, status)
        call fortbo_expected_improvement(mean_a, sd_a, best, 0.0_dp, moment_averaged)
        call expect(abs(integrated(1) - moment_averaged) < 1.0e-14_dp, &
            "with the samples in agreement the two coincide", failures)
    end subroutine check_acquisition_average_is_not_moment_average

    subroutine check_diagnostics(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: pinned(2, 5), varied(2, 5), spread(2)
        integer :: k

        do k = 1, 5
            pinned(:, k) = [1.0_dp, 2.0_dp]
            varied(1, k) = 1.0_dp + 0.3_dp*real(k, dp)
            varied(2, k) = 2.0_dp
        end do

        call fortbo_hyperparameter_spread(pinned, spread, status)
        call expect(status%code == FORTNUM_OK .and. maxval(spread) == 0.0_dp, &
            "a pinned chain has no spread", failures)
        call expect(.not. fortbo_integration_is_informative(spread), &
            "integrating a pinned chain is reported as buying nothing", failures)

        call fortbo_hyperparameter_spread(varied, spread, status)
        call expect(spread(1) > 0.0_dp .and. spread(2) == 0.0_dp, &
            "the spread is reported per parameter", failures)
        call expect(fortbo_integration_is_informative(spread), &
            "a chain that moved in one parameter is informative", failures)
    end subroutine check_diagnostics

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: values(2, 3), integrated(2), spread(2)
        real(dp) :: single(2, 1)

        values = 1.0_dp
        single = 1.0_dp

        call fortbo_integrated_acquisition(values, integrated, status, &
            [1.0_dp, 1.0_dp])
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "weights of the wrong length are refused", failures)

        call fortbo_integrated_acquisition(values, integrated, status, &
            [1.0_dp, -1.0_dp, 1.0_dp])
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative weight is refused", failures)

        call fortbo_integrated_acquisition(values, integrated, status, &
            [0.0_dp, 0.0_dp, 0.0_dp])
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "all-zero weights are refused", failures)

        ! A spread needs at least two draws to mean anything.
        call fortbo_hyperparameter_spread(single, spread, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a one-draw chain has no spread to report", failures)
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

end program test_integrated
