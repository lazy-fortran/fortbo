program test_variational_derivative
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_COVARIANCE, &
        FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_variational_derivative, only: &
        fortbo_variational_derivative_posterior_t, &
        fortbo_fit_variational_derivative
    implicit none

    integer :: failures

    failures = 0
    call check_paired_observations(failures)
    call check_value_only_override(failures)
    if (failures == 0) then
        print *, "test_variational_derivative: PASS"
    else
        write (error_unit, *) "test_variational_derivative: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_paired_observations(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        real(dp) :: inducing(1, 1), query(1, 1), mean(1), variance(1)
        real(dp) :: mean_gradient(1, 1), sd_gradient(1, 1)
        real(dp) :: expected_mean, expected_variance, k_zx, k_zdx, k_qz
        real(dp) :: k_zz, b, beta, signal, noise, ls, y, dy
        real(dp) :: h, plus(1, 1), minus(1, 1), mean_plus(1), mean_minus(1)
        real(dp) :: variance_plus(1), variance_minus(1)
        real(dp) :: covariance(2, 2)
        type(fortnum_status_t) :: status

        call history%initialize(1, 0, status)
        call expect(status%code == FORTNUM_OK, "history initialized", failures)
        call history%add([0.25_dp], status, objective=0.5_dp, gradient=[1.25_dp])
        call expect(status%code == FORTNUM_OK, "paired row added", failures)
        inducing = reshape([0.0_dp], shape(inducing))
        query = reshape([0.75_dp], shape(query))
        signal = 1.3_dp
        noise = 0.2_dp
        ls = 0.6_dp
        y = 0.5_dp
        dy = 1.25_dp
        call fortbo_fit_variational_derivative(history, inducing, posterior, &
            [ls], status, signal_variance=signal, noise_variance=noise)
        call expect(status%code == FORTNUM_OK, "paired model fitted", failures)
        select type (model => posterior)
        type is (fortbo_variational_derivative_posterior_t)
            call expect(model%n_observations == 2, "value and derivative rows retained", failures)
            call expect(model%n_gradient_observations == 1, &
                "derivative observation counted", failures)
            call expect(iand(model%capabilities(), FORTBO_CAP_COVARIANCE) /= 0, &
                "covariance capability advertised", failures)
            call expect(iand(model%capabilities(), FORTBO_CAP_JOINT_SAMPLE) /= 0, &
                "joint sample capability advertised", failures)
            call expect(iand(model%capabilities(), FORTBO_CAP_MOMENT_GRADIENT) /= 0, &
                "moment gradient capability advertised", failures)
            call model%moments(query, mean, variance, status)
            call expect(status%code == FORTNUM_OK, "paired moments returned", failures)
            call model%moment_gradient(query, mean_gradient, sd_gradient, status)
            call expect(status%code == FORTNUM_OK, &
                "paired moment gradient returned", failures)
            call model%covariance(reshape([0.5_dp, 0.75_dp], [2, 1]), covariance, status)
            call expect(status%code == FORTNUM_OK, "paired covariance returned", failures)
            call expect(maxval(abs(covariance - transpose(covariance))) < 1.0e-12_dp, &
                "paired covariance symmetric", failures)
            h = 1.0e-5_dp
            plus = query + h
            minus = query - h
            call model%moments(plus, mean_plus, variance_plus, status)
            call model%moments(minus, mean_minus, variance_minus, status)
            call expect(abs(mean_gradient(1, 1) - (mean_plus(1) - mean_minus(1))/(2.0_dp*h)) &
                < 1.0e-6_dp, "paired mean gradient matches finite difference", failures)
        class default
            call expect(.false., "paired model has expected concrete type", failures)
            return
        end select

        k_zz = matern52(0.0_dp, 0.0_dp, ls, signal)
        k_zx = matern52(0.0_dp, 0.25_dp, ls, signal)
        k_zdx = -matern52_gradient(0.0_dp, 0.25_dp, ls, signal)
        k_qz = matern52(0.75_dp, 0.0_dp, ls, signal)
        k_zz = k_zz + 1.0e-12_dp*max(signal, 1.0_dp)
        b = k_zz + (k_zx*k_zx + k_zdx*k_zdx)/noise
        beta = (k_zx*y + k_zdx*dy)/(noise*b)
        expected_mean = k_qz*beta
        expected_variance = signal - k_qz*k_qz/k_zz + k_qz*k_qz/b
        call expect(abs(mean(1) - expected_mean) < 1.0e-11_dp, &
            "paired mean matches independent variational oracle", failures)
        call expect(abs(variance(1) - expected_variance) < 1.0e-11_dp, &
            "paired variance matches independent variational oracle", failures)
    end subroutine check_paired_observations

    subroutine check_value_only_override(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: paired, values
        real(dp) :: inducing(1, 1), query(1, 1), paired_mean(1), value_mean(1)
        real(dp) :: paired_variance(1), value_variance(1)
        type(fortnum_status_t) :: status

        call history%initialize(1, 0, status)
        call history%add([0.25_dp], status, objective=0.5_dp, gradient=[1.25_dp])
        inducing = reshape([0.0_dp], shape(inducing))
        query = reshape([0.75_dp], shape(query))
        call fortbo_fit_variational_derivative(history, inducing, paired, [0.6_dp], &
            status, signal_variance=1.3_dp, noise_variance=0.2_dp)
        call expect(status%code == FORTNUM_OK, "automatic derivative fit succeeds", failures)
        call fortbo_fit_variational_derivative(history, inducing, values, [0.6_dp], &
            status, signal_variance=1.3_dp, noise_variance=0.2_dp, use_gradients=.false.)
        call expect(status%code == FORTNUM_OK, "explicit value-only fit succeeds", failures)
        select type (paired_model => paired)
        type is (fortbo_variational_derivative_posterior_t)
            select type (value_model => values)
            type is (fortbo_variational_derivative_posterior_t)
                call paired_model%moments(query, paired_mean, paired_variance, status)
                call value_model%moments(query, value_mean, value_variance, status)
                call expect(paired_model%n_observations == 2 .and. &
                    value_model%n_observations == 1, &
                    "explicit value-only override drops derivative rows", failures)
                call expect(abs(paired_mean(1) - value_mean(1)) > 1.0e-5_dp, &
                    "derivative observation changes the posterior", failures)
            class default
                call expect(.false., "value-only concrete type", failures)
            end select
        class default
            call expect(.false., "automatic derivative concrete type", failures)
        end select
    end subroutine check_value_only_override

    pure real(dp) function matern52(x1, x2, lengthscale, signal) result(value)
        real(dp), intent(in) :: x1, x2, lengthscale, signal
        real(dp) :: scaled

        scaled = sqrt(5.0_dp)*abs(x1 - x2)/lengthscale
        value = signal*(1.0_dp + scaled + scaled*scaled/3.0_dp)*exp(-scaled)
    end function matern52

    pure real(dp) function matern52_gradient(x1, x2, lengthscale, signal) result(value)
        real(dp), intent(in) :: x1, x2, lengthscale, signal
        real(dp) :: delta, scaled, coefficient

        delta = x1 - x2
        scaled = sqrt(5.0_dp)*abs(delta)/lengthscale
        coefficient = -5.0_dp*signal*(1.0_dp + scaled)*exp(-scaled)/3.0_dp
        value = coefficient*delta/(lengthscale*lengthscale)
    end function matern52_gradient

    subroutine expect(condition, message, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, *) "FAIL:", trim(message)
            failures = failures + 1
        end if
    end subroutine expect

end program test_variational_derivative
