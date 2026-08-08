program test_structured
    !! BO2: multi-task and deep-kernel surrogates behind the posterior contract.
    !!
    !! The point of these adapters is that a policy cannot tell what is behind
    !! them, so the checks are about the contract holding rather than about the
    !! models, which FortML's own suite covers.
    !!
    !! The oracles that matter:
    !!
    !!   * **The target output is respected.** A multi-output model asked for
    !!     output two must not report output one. The fixture gives the two
    !!     outputs genuinely different data, so an adapter reading the wrong
    !!     one is caught. A fixture where both outputs looked alike would pass
    !!     while discriminating nothing, which is a mistake worth naming
    !!     because it is easy to write by accident.
    !!   * **The variance comes from the right place in the joint covariance.**
    !!     The joint matrix is ordered outputs-fastest, and the marginal for a
    !!     given query and output sits on its diagonal. A transposed stride
    !!     yields a plausible number belonging to another task, so the reported
    !!     variance is checked against the model's own covariance diagonal
    !!     rather than against a tolerance on plausibility.
    !!   * **Capabilities are declared honestly.** Neither adapter offers
    !!     moment gradients, and a policy asking for one must get a refusal by
    !!     name rather than a differenced approximation.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_deep_kernel_gp, only: deep_kernel_gp_t
    use fortbo_posterior, only: FORTBO_CAP_MOMENTS, FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_structured, only: fortbo_multi_task_posterior_t, &
        fortbo_deep_kernel_posterior_t
    use fortbo_acquisition, only: fortbo_ei_t
    implicit none

    integer :: failures

    failures = 0
    call check_multi_task_reports_the_named_output(failures)
    call check_multi_task_variance_is_the_right_marginal(failures)
    call check_deep_kernel_posterior(failures)
    call check_an_acquisition_can_use_them(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_structured: PASS"
    else
        print *, "test_structured: FAIL", failures
        error stop 1
    end if

contains

    !! Two outputs with genuinely different behaviour, so reading the wrong one
    !! is detectable.
    subroutine fixture(inputs, targets)
        real(dp), intent(out) :: inputs(:, :), targets(:, :)
        integer :: k

        do k = 1, size(inputs, 1)
            inputs(k, 1) = -1.0_dp + 0.25_dp*real(k, dp)
            inputs(k, 2) = 0.3_dp*cos(0.9_dp*real(k, dp))
            ! Output one rises with the first coordinate; output two falls and
            ! is offset. Nothing about them coincides.
            targets(k, 1) = 2.0_dp*inputs(k, 1) + 0.5_dp
            targets(k, 2) = -3.0_dp*inputs(k, 1) - 1.5_dp
        end do
    end subroutine fixture

    subroutine build_multi_output(model, status)
        type(multi_output_gp_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel
        real(dp) :: inputs(10, 2), targets(10, 2)
        real(dp) :: weights(2, 1), independent(2)

        kernel = make_rbf_kernel(2, 1.0_dp, 0.7_dp, status)
        if (status%code /= FORTNUM_OK) return
        ! One shared latent process plus per-output independent variance: the
        ! coregionalization that makes the two tasks share information.
        weights(1, 1) = 1.0_dp
        weights(2, 1) = -0.6_dp
        independent = [0.3_dp, 0.4_dp]
        call model%initialize(kernel, weights, independent, 1.0e-6_dp, status)
        if (status%code /= FORTNUM_OK) return

        call fixture(inputs, targets)
        call model%fit(inputs, targets, status)
    end subroutine build_multi_output

    subroutine check_multi_task_reports_the_named_output(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(fortbo_multi_task_posterior_t) :: first, second
        type(fortnum_status_t) :: status
        real(dp) :: query(4, 2)
        real(dp) :: mean_one(4), variance_one(4)
        real(dp) :: mean_two(4), variance_two(4)
        real(dp) :: direct(4, 2)
        integer :: k

        call build_multi_output(model, status)
        call expect(status%code == FORTNUM_OK, "the multi-output model fits", &
            failures)
        if (status%code /= FORTNUM_OK) return

        do k = 1, 4
            query(k, 1) = -0.5_dp + 0.3_dp*real(k, dp)
            query(k, 2) = 0.1_dp*real(k, dp)
        end do

        call first%adopt(model, 2, 1, status)
        call expect(status%code == FORTNUM_OK, "the first task is adopted", &
            failures)
        call second%adopt(model, 2, 2, status)
        call expect(status%code == FORTNUM_OK, "the second task is adopted", &
            failures)

        call first%moments(query, mean_one, variance_one, status)
        call expect(status%code == FORTNUM_OK, "the first task reports moments", &
            failures)
        call second%moments(query, mean_two, variance_two, status)
        call expect(status%code == FORTNUM_OK, "the second task reports moments", &
            failures)

        ! Against the model's own prediction, so the adapter is checked rather
        ! than assumed to have picked the right column.
        call model%predict(query, direct, status)
        call expect(maxval(abs(mean_one - direct(:, 1))) < 1.0e-12_dp, &
            "the target output one is the one reported", failures)
        call expect(maxval(abs(mean_two - direct(:, 2))) < 1.0e-12_dp, &
            "the target output two is the one reported", failures)

        ! And the two really differ, so the check above discriminates.
        call expect(maxval(abs(mean_one - mean_two)) > 0.1_dp, &
            "the two tasks are actually distinguishable in this fixture", &
            failures)
    end subroutine check_multi_task_reports_the_named_output

    !! The variance must be the diagonal entry of the joint covariance for the
    !! named output, not for some other task at the same query.
    subroutine check_multi_task_variance_is_the_right_marginal(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(fortbo_multi_task_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: query(3, 2), mean(3), variance(3)
        real(dp), allocatable :: covariance(:, :)
        integer :: k, index
        logical :: matches

        call build_multi_output(model, status)
        if (status%code /= FORTNUM_OK) return
        do k = 1, 3
            query(k, 1) = -0.3_dp + 0.4_dp*real(k, dp)
            query(k, 2) = -0.2_dp*real(k, dp)
        end do

        call posterior%adopt(model, 2, 2, status)
        call posterior%moments(query, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "moments are reported", failures)

        allocate (covariance(3*2, 3*2))
        call model%predict_covariance(query, covariance, status)
        call expect(status%code == FORTNUM_OK, "the joint covariance is available", &
            failures)

        matches = .true.
        do k = 1, 3
            index = (k - 1)*2 + 2
            if (abs(variance(k) - covariance(index, index)) > 1.0e-12_dp) &
                matches = .false.
        end do
        call expect(matches, &
            "the reported variance is the target output's own marginal", failures)

        ! And it is not the *other* output's, which a transposed stride would
        ! produce and which the check above would otherwise not distinguish
        ! from correctness if the two happened to be close.
        matches = .false.
        do k = 1, 3
            index = (k - 1)*2 + 1
            if (abs(variance(k) - covariance(index, index)) > 1.0e-10_dp) &
                matches = .true.
        end do
        call expect(matches, &
            "the two outputs' marginals differ, so the stride is pinned", &
            failures)
    end subroutine check_multi_task_variance_is_the_right_marginal

    subroutine check_deep_kernel_posterior(failures)
        integer, intent(inout) :: failures
        type(deep_kernel_gp_t) :: model
        type(fortbo_deep_kernel_posterior_t) :: posterior
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), y(12, 1), query(4, 2)
        real(dp) :: mean(4), variance(4)
        real(dp) :: direct_mean(4, 1), direct_variance(4)
        integer :: k

        do k = 1, 12
            x(k, 1) = -1.4_dp + 0.26_dp*real(k, dp)
            x(k, 2) = 0.35_dp*sin(0.8_dp*real(k, dp))
            y(k, 1) = sin(1.1_dp*x(k, 1)) - 0.4_dp*x(k, 2)
        end do
        do k = 1, 4
            query(k, 1) = -0.7_dp + 0.4_dp*real(k, dp)
            query(k, 2) = 0.15_dp*real(k, dp)
        end do

        kernel = make_rbf_kernel(3, 1.0_dp, 0.9_dp, status)
        call model%initialize([2, 5, 3], kernel, status, initialization_seed=3)
        call expect(status%code == FORTNUM_OK, "the deep kernel model builds", &
            failures)
        call model%fit(x, y, 1.0e-5_dp, status)
        call expect(status%code == FORTNUM_OK, "the deep kernel model fits", &
            failures)
        if (status%code /= FORTNUM_OK) return

        call posterior%adopt(model, status)
        call expect(status%code == FORTNUM_OK, "the deep kernel model is adopted", &
            failures)
        call expect(posterior%n_inputs() == 2, &
            "the posterior reports the input width, not the feature width", &
            failures)

        call posterior%moments(query, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "the posterior reports moments", &
            failures)

        call model%predict(query, direct_mean, direct_variance, status)
        call expect(maxval(abs(mean - direct_mean(:, 1))) < 1.0e-14_dp, &
            "the adapter reports the model's own mean", failures)
        call expect(maxval(abs(variance - direct_variance)) < 1.0e-14_dp, &
            "the adapter reports the model's own variance", failures)
        call expect(all(variance >= 0.0_dp), &
            "the reported variance is never negative", failures)
    end subroutine check_deep_kernel_posterior

    !! The whole purpose: an acquisition works against these without knowing
    !! what is behind them.
    subroutine check_an_acquisition_can_use_them(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model
        type(fortbo_multi_task_posterior_t) :: posterior
        type(fortbo_ei_t) :: acquisition
        type(fortnum_status_t) :: status
        real(dp) :: query(4, 2), values(4)
        integer :: k

        call build_multi_output(model, status)
        if (status%code /= FORTNUM_OK) return
        do k = 1, 4
            query(k, 1) = -0.4_dp + 0.3_dp*real(k, dp)
            query(k, 2) = 0.05_dp*real(k, dp)
        end do

        call posterior%adopt(model, 2, 1, status)
        acquisition%best = 0.0_dp
        acquisition%xi = 0.0_dp
        call acquisition%value(posterior, query, values, status)
        call expect(status%code == FORTNUM_OK, &
            "expected improvement evaluates against a multi-task surrogate", &
            failures)
        call expect(all(values >= 0.0_dp), &
            "expected improvement is never negative", failures)
    end subroutine check_an_acquisition_can_use_them

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(multi_output_gp_t) :: model, unfitted_model
        type(deep_kernel_gp_t) :: deep
        type(fortbo_multi_task_posterior_t) :: posterior, fresh
        type(fortbo_deep_kernel_posterior_t) :: deep_posterior
        type(fortnum_status_t) :: status
        real(dp) :: query(3, 2), mean(3), variance(3)
        real(dp) :: narrow(3, 1), short_mean(2), short_variance(2)

        call build_multi_output(model, status)
        query = 0.1_dp
        narrow = 0.1_dp

        ! Capabilities are declared, and the ones not offered are not claimed.
        call posterior%adopt(model, 2, 1, status)
        call expect(posterior%supports(FORTBO_CAP_MOMENTS), &
            "the multi-task adapter declares moments", failures)
        call expect(.not. posterior%supports(FORTBO_CAP_MOMENT_GRADIENT), &
            "and does not claim moment gradients it cannot supply", failures)

        ! A model that was never fitted cannot be adopted: the alternative is a
        ! posterior that reports its prior as though it had seen data.
        call fresh%adopt(unfitted_model, 2, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "adopting an unfitted multi-output model is refused", failures)

        call fresh%adopt(model, 2, 3, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a target output beyond the model's width is refused", failures)
        call fresh%adopt(model, 2, 0, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a target output of zero is refused rather than defaulted", failures)
        call fresh%adopt(model, 0, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an input width of zero is refused", failures)

        call fresh%moments(query, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "moments before adoption are refused", failures)

        call posterior%moments(narrow, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "query points of the wrong width are refused", failures)
        call posterior%moments(query, short_mean, short_variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched moment buffers are refused", failures)

        call deep_posterior%adopt(deep, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "adopting an unfitted deep kernel model is refused", failures)
        call deep_posterior%moments(query, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "deep kernel moments before adoption are refused", failures)
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

end program test_structured
