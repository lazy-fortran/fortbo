program test_fortml_adapter
    !! BO2: FortML surrogates behind the posterior contract.
    !!
    !! The claim under test is the roadmap's central invariant: derivative
    !! observations must reach *any* method, with no change above the posterior
    !! boundary. The oracles are behavioral:
    !!
    !!   * interpolation. A fitted GP must reproduce its training values at the
    !!     training inputs, with near-zero posterior variance there. That is a
    !!     property of conditioning, checked against the values the test itself
    !!     supplied;
    !!   * the derivative payoff. On a function whose gradient carries real
    !!     information, a model conditioned on values *and* gradients must
    !!     predict held-out points better than the same model conditioned on the
    !!     same values alone. Measured on a held-out grid, not asserted;
    !!   * indistinguishability. The same acquisition code, unchanged, must run
    !!     against both adapters and produce sensible values from each. This is
    !!     what "usable with any method" means operationally;
    !!   * the branch is chosen from the data, and forcing gradients that do not
    !!     exist is refused rather than silently downgraded.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_history, only: fortbo_history_t
    use fortbo_fortml, only: fortbo_fit_from_history, fortbo_gp_posterior_t, &
        fortbo_derivative_gp_posterior_t
    use fortbo_acquisition, only: fortbo_ei_t, fortbo_ucb_t
    implicit none

    integer :: failures

    failures = 0
    call check_value_only_interpolates(failures)
    call check_branch_is_chosen_from_data(failures)
    call check_gradients_improve_prediction(failures)
    call check_acquisitions_are_indistinguishable(failures)
    call check_moment_gradients(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_fortml_adapter: PASS"
    else
        print *, "test_fortml_adapter: FAIL", failures
        error stop 1
    end if

contains

    !! The test objective and its exact gradient. A smooth, non-separable
    !! function whose gradient genuinely constrains the surface.
    pure real(dp) function objective(x) result(value)
        real(dp), intent(in) :: x(:)

        value = sin(3.0_dp*x(1))*cos(2.0_dp*x(2)) + 0.5_dp*x(1)*x(2)
    end function objective

    pure function objective_gradient(x) result(gradient)
        real(dp), intent(in) :: x(:)
        real(dp) :: gradient(2)

        gradient(1) = 3.0_dp*cos(3.0_dp*x(1))*cos(2.0_dp*x(2)) + 0.5_dp*x(2)
        gradient(2) = -2.0_dp*sin(3.0_dp*x(1))*sin(2.0_dp*x(2)) + 0.5_dp*x(1)
    end function objective_gradient

    !! Deterministic training sites on a coarse lattice inside the unit square.
    subroutine training_site(k, point)
        integer, intent(in) :: k
        real(dp), intent(out) :: point(2)
        integer :: row, column

        row = (k - 1)/4
        column = mod(k - 1, 4)
        point(1) = 0.1_dp + 0.26_dp*real(column, dp)
        point(2) = 0.1_dp + 0.26_dp*real(row, dp)
    end subroutine training_site

    subroutine check_value_only_interpolates(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: point(2), points(16, 2), mean(16), variance(16)
        integer :: k
        logical :: interpolates, confident

        call history%initialize(2, 0, status)
        do k = 1, 16
            call training_site(k, point)
            call history%add(point, status, objective=objective(point))
            points(k, :) = point
        end do

        call fortbo_fit_from_history(history, posterior, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-8_dp)
        call expect(status%code == FORTNUM_OK, "fitting a value-only GP succeeds", &
            failures)
        call expect(posterior%supports(FORTBO_CAP_MOMENTS), &
            "the fitted surrogate declares moments", failures)
        call expect(posterior%n_inputs() == 2, "the surrogate reports its width", &
            failures)

        select type (posterior)
            type is (fortbo_gp_posterior_t)
            call expect(.true., "a history without gradients yields the value-only GP", &
                failures)
        class default
            call expect(.false., "a history without gradients yields the value-only GP", &
                failures)
        end select

        call posterior%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "prediction succeeds", failures)

        interpolates = .true.
        confident = .true.
        do k = 1, 16
            if (abs(mean(k) - objective(points(k, :))) > 1.0e-4_dp) interpolates = .false.
            if (variance(k) > 1.0e-4_dp) confident = .false.
        end do
        call expect(interpolates, "the GP reproduces its training values", failures)
        call expect(confident, "the posterior variance collapses at training sites", &
            failures)
    end subroutine check_value_only_interpolates

    subroutine check_branch_is_chosen_from_data(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: point(2)
        integer :: k
        logical :: is_derivative_model

        call history%initialize(2, 0, status)
        do k = 1, 9
            call training_site(k, point)
            call history%add(point, status, objective=objective(point), &
                gradient=objective_gradient(point))
        end do

        call fortbo_fit_from_history(history, posterior, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-8_dp)
        call expect(status%code == FORTNUM_OK, &
            "fitting a derivative-observation GP succeeds", failures)

        is_derivative_model = .false.
        select type (posterior)
            type is (fortbo_derivative_gp_posterior_t)
            is_derivative_model = .true.
        end select
        call expect(is_derivative_model, &
            "a history with gradients yields the derivative GP without asking", &
            failures)
        call expect(posterior%supports(FORTBO_CAP_MOMENTS), &
            "the derivative surrogate presents the same capability", failures)

        ! Forcing the value-only branch must still work: a caller may have a
        ! reason to ignore gradients, but it has to say so.
        call fortbo_fit_from_history(history, posterior, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-8_dp, use_gradients=.false.)
        call expect(status%code == FORTNUM_OK, "the value-only branch can be forced", &
            failures)
        is_derivative_model = .false.
        select type (posterior)
            type is (fortbo_derivative_gp_posterior_t)
            is_derivative_model = .true.
        end select
        call expect(.not. is_derivative_model, &
            "forcing the value-only branch is honored", failures)
    end subroutine check_branch_is_chosen_from_data

    !! The payoff test. Same inputs, same values, same kernel, same noise; the
    !! only difference is whether the gradients were used. On a held-out grid
    !! the derivative-informed model must be measurably better, or conditioning
    !! on gradients is not doing what it claims.
    subroutine check_gradients_improve_prediction(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: plain, informed
        type(fortnum_status_t) :: status
        integer, parameter :: n_test = 49
        real(dp) :: point(2), test_points(n_test, 2)
        real(dp) :: mean_plain(n_test), var_plain(n_test)
        real(dp) :: mean_informed(n_test), var_informed(n_test)
        real(dp) :: error_plain, error_informed, truth
        integer :: k, i, j

        call history%initialize(2, 0, status)
        do k = 1, 9
            call training_site(k, point)
            call history%add(point, status, objective=objective(point), &
                gradient=objective_gradient(point))
        end do

        k = 0
        do i = 1, 7
            do j = 1, 7
                k = k + 1
                test_points(k, 1) = 0.15_dp + 0.1_dp*real(i, dp)
                test_points(k, 2) = 0.15_dp + 0.1_dp*real(j, dp)
            end do
        end do

        call fortbo_fit_from_history(history, plain, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-8_dp, use_gradients=.false.)
        call expect(status%code == FORTNUM_OK, "the value-only fit succeeds", failures)
        call plain%moments(test_points, mean_plain, var_plain, status)
        call expect(status%code == FORTNUM_OK, "the value-only prediction succeeds", &
            failures)

        call fortbo_fit_from_history(history, informed, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-8_dp, use_gradients=.true.)
        call expect(status%code == FORTNUM_OK, "the derivative-informed fit succeeds", &
            failures)
        call informed%moments(test_points, mean_informed, var_informed, status)
        call expect(status%code == FORTNUM_OK, &
            "the derivative-informed prediction succeeds", failures)

        error_plain = 0.0_dp
        error_informed = 0.0_dp
        do k = 1, n_test
            truth = objective(test_points(k, :))
            error_plain = error_plain + (mean_plain(k) - truth)**2
            error_informed = error_informed + (mean_informed(k) - truth)**2
        end do

        call expect(error_informed < error_plain, &
            "gradient observations reduce held-out prediction error", failures)
        call expect(sum(var_informed) < sum(var_plain), &
            "gradient observations reduce posterior uncertainty", failures)
        if (error_informed >= error_plain) then
            print *, "    plain squared error:", error_plain
            print *, "    informed squared error:", error_informed
        end if
    end subroutine check_gradients_improve_prediction

    !! The same acquisition objects, unchanged, driven by each adapter. Nothing
    !! in the acquisition code branches on how the posterior was conditioned.
    subroutine check_acquisitions_are_indistinguishable(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: plain, informed
        type(fortnum_status_t) :: status
        type(fortbo_ei_t) :: ei
        type(fortbo_ucb_t) :: ucb
        real(dp) :: point(2), query(5, 2)
        real(dp) :: ei_plain(5), ei_informed(5), ucb_plain(5), ucb_informed(5)
        integer :: k

        call history%initialize(2, 0, status)
        do k = 1, 9
            call training_site(k, point)
            call history%add(point, status, objective=objective(point), &
                gradient=objective_gradient(point))
        end do
        do k = 1, 5
            query(k, 1) = 0.2_dp + 0.15_dp*real(k, dp)
            query(k, 2) = 0.8_dp - 0.12_dp*real(k, dp)
        end do

        call fortbo_fit_from_history(history, plain, status, lengthscale=0.4_dp, &
            use_gradients=.false.)
        call fortbo_fit_from_history(history, informed, status, lengthscale=0.4_dp, &
            use_gradients=.true.)

        ei%best = 0.0_dp
        ucb%beta = 2.0_dp

        call ei%value(plain, query, ei_plain, status)
        call expect(status%code == FORTNUM_OK, &
            "EI runs unchanged against the value-only surrogate", failures)
        call ei%value(informed, query, ei_informed, status)
        call expect(status%code == FORTNUM_OK, &
            "EI runs unchanged against the derivative surrogate", failures)
        call expect(all(ei_plain >= 0.0_dp) .and. all(ei_informed >= 0.0_dp), &
            "both surrogates give nonnegative expected improvement", failures)

        call ucb%value(plain, query, ucb_plain, status)
        call expect(status%code == FORTNUM_OK, &
            "the confidence bound runs against the value-only surrogate", &
            failures)
        call ucb%value(informed, query, ucb_informed, status)
        call expect(status%code == FORTNUM_OK, &
            "the confidence bound runs against the derivative surrogate", &
            failures)

        ! The two must genuinely differ, or the derivative rows were ignored.
        call expect(maxval(abs(ucb_plain - ucb_informed)) > 1.0e-8_dp, &
            "the derivative surrogate actually changes the acquisition", &
            failures)
    end subroutine check_acquisitions_are_indistinguishable

    !! The derivative-observation surrogate exposes query-input moment
    !! gradients, checked against central differences of its own moments. This
    !! is what lets gradient-based candidate search run against a real
    !! surrogate rather than only against a synthetic posterior.
    subroutine check_moment_gradients(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: informed, plain
        type(fortnum_status_t) :: status
        real(dp) :: point(2), query(3, 2), shifted(3, 2)
        real(dp) :: mean_gradient(3, 2), sd_gradient(3, 2)
        real(dp) :: mean(3), variance(3), plus_mean(3), minus_mean(3)
        real(dp) :: plus_var(3), minus_var(3)
        real(dp) :: numeric_mean, numeric_sd
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: k, i, j
        logical :: mean_ok, sd_ok

        call history%initialize(2, 0, status)
        do k = 1, 9
            call training_site(k, point)
            call history%add(point, status, objective=objective(point), &
                gradient=objective_gradient(point))
        end do
        call fortbo_fit_from_history(history, informed, status, lengthscale=0.4_dp, &
            noise_variance=1.0e-6_dp, use_gradients=.true.)

        call expect(informed%supports(FORTBO_CAP_MOMENT_GRADIENT), &
            "the derivative surrogate declares moment gradients", failures)

        ! Query points deliberately away from the training lattice, where the
        ! posterior variance is nonzero and the square-root derivative exists.
        query(1, :) = [0.22_dp, 0.62_dp]
        query(2, :) = [0.71_dp, 0.19_dp]
        query(3, :) = [0.48_dp, 0.44_dp]

        call informed%moment_gradient(query, mean_gradient, sd_gradient, status)
        call expect(status%code == FORTNUM_OK, "moment gradients evaluate", failures)
        call informed%moments(query, mean, variance, status)

        mean_ok = .true.
        sd_ok = .true.
        do j = 1, 2
            shifted = query
            shifted(:, j) = query(:, j) + step
            call informed%moments(shifted, plus_mean, plus_var, status)
            shifted(:, j) = query(:, j) - step
            call informed%moments(shifted, minus_mean, minus_var, status)
            do i = 1, 3
                numeric_mean = (plus_mean(i) - minus_mean(i))/(2.0_dp*step)
                if (abs(mean_gradient(i, j) - numeric_mean) > &
                    1.0e-4_dp*max(1.0_dp, abs(numeric_mean))) mean_ok = .false.
                numeric_sd = (sqrt(max(plus_var(i), 0.0_dp)) &
                    - sqrt(max(minus_var(i), 0.0_dp)))/(2.0_dp*step)
                if (abs(sd_gradient(i, j) - numeric_sd) > &
                    1.0e-4_dp*max(1.0_dp, abs(numeric_sd))) sd_ok = .false.
            end do
        end do
        call expect(mean_ok, "the mean gradient matches central differences", failures)
        call expect(sd_ok, &
            "the standard-deviation gradient matches central differences", failures)

        ! The value-only GP must not claim a capability FortML cannot provide.
        call fortbo_fit_from_history(history, plain, status, lengthscale=0.4_dp, &
            use_gradients=.false.)
        call expect(.not. plain%supports(FORTBO_CAP_MOMENT_GRADIENT), &
            "the value-only GP does not claim input gradients it lacks", failures)
        call plain%moment_gradient(query, mean_gradient, sd_gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "the value-only GP refuses an input gradient by name", failures)
    end subroutine check_moment_gradients

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortbo_gp_posterior_t) :: unfitted
        type(fortnum_status_t) :: status
        real(dp) :: point(2), points(1, 2), mean(1), variance(1)

        unfitted%dimension = 2
        points = 0.5_dp
        call unfitted%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unfitted surrogate refuses to predict", failures)
        call expect(.not. unfitted%supports(FORTBO_CAP_MOMENTS), &
            "an unfitted surrogate declares no capability", failures)

        call history%initialize(2, 0, status)
        call training_site(1, point)
        call history%add(point, status, objective=objective(point))
        call fortbo_fit_from_history(history, posterior, status, use_gradients=.true.)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "requesting gradients the history lacks is refused", failures)
        call expect(index(status%msg, "history has none") > 0, &
            "the refusal explains that no gradients were recorded", failures)

        call history%initialize(2, 0, status)
        call fortbo_fit_from_history(history, posterior, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "fitting an empty history is refused", failures)
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

end program test_fortml_adapter
