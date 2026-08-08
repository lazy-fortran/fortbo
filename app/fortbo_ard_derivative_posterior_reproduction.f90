program fortbo_ard_derivative_posterior_reproduction
    !! Frozen ARD Matern-5/2 value-plus-derivative posterior fixture.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_fortml, only: fortbo_fit_from_history
    implicit none

    type(fortbo_history_t) :: history
    class(fortbo_posterior_t), allocatable :: posterior
    type(fortnum_status_t) :: status
    real(dp), parameter :: train_x(4, 2) = reshape([ &
        0.10_dp, 0.40_dp, 0.70_dp, 0.90_dp, &
        0.20_dp, 0.80_dp, 0.30_dp, 0.95_dp], [4, 2])
    real(dp), parameter :: train_y(4) = [0.7_dp, -0.2_dp, 0.4_dp, 1.1_dp]
    real(dp), parameter :: train_grad(4, 2) = reshape([ &
        0.3_dp, 0.2_dp, -0.2_dp, 0.1_dp, &
        -0.1_dp, 0.4_dp, 0.5_dp, -0.3_dp], [4, 2])
    real(dp), parameter :: query_x(3, 2) = reshape([ &
        0.15_dp, 0.55_dp, 0.85_dp, &
        0.25_dp, 0.65_dp, 0.60_dp], [3, 2])
    real(dp), parameter :: lengthscales(2) = [0.30_dp, 0.65_dp]
    real(dp), allocatable :: mean(:), variance(:)
    real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
    integer :: i

    call history%initialize(2, 0, status)
    if (status%code /= FORTNUM_OK) error stop 1
    do i = 1, size(train_y)
        call history%add(train_x(i, :), status, objective=train_y(i), &
            gradient=train_grad(i, :))
        if (status%code /= FORTNUM_OK) error stop 1
    end do
    call fortbo_fit_from_history(history, posterior, status, &
        lengthscales=lengthscales, signal_variance=1.2_dp, &
        noise_variance=0.04_dp, use_gradients=.true.)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR fit ", trim(status%msg)
        error stop 1
    end if

    allocate (mean(size(query_x, 1)), variance(size(query_x, 1)))
    call posterior%moments(query_x, mean, variance, status)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR moments ", trim(status%msg)
        error stop 1
    end if
    do i = 1, size(mean)
        write (*, '("MOMENT ", I0, 1X, ES24.16E3, 1X, ES24.16E3)') &
            i, mean(i), variance(i)
    end do

    allocate (mean_gradient(size(query_x, 1), 2), sd_gradient(size(query_x, 1), 2))
    call posterior%moment_gradient(query_x, mean_gradient, sd_gradient, status)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR gradients ", trim(status%msg)
        error stop 1
    end if
    do i = 1, size(query_x, 1)
        write (*, '("GRADIENT ", I0, 1X, I0, 1X, ES24.16E3, 1X, ES24.16E3)') &
            i, 1, mean_gradient(i, 1), sd_gradient(i, 1)
        write (*, '("GRADIENT ", I0, 1X, I0, 1X, ES24.16E3, 1X, ES24.16E3)') &
            i, 2, mean_gradient(i, 2), sd_gradient(i, 2)
    end do
end program fortbo_ard_derivative_posterior_reproduction
