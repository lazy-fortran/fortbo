program fortbo_posterior_reproduction
    !! Emit a frozen FortML value-only posterior for the independent Python
    !! Landreman parity checker. No generated kernel or FortBO output is used
    !! as the expected answer on the Python side.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_history, only: fortbo_history_t
    use fortbo_fortml, only: fortbo_fit_from_history
    use fortbo_posterior, only: fortbo_posterior_t
    implicit none

    integer, parameter :: dimension = 2, n_train = 4, n_query = 3
    real(dp) :: train_x(n_train, dimension), train_y(n_train)
    real(dp) :: query_x(n_query, dimension), mean(n_query), variance(n_query)
    integer :: i
    type(fortbo_history_t) :: history
    class(fortbo_posterior_t), allocatable :: posterior
    type(fortnum_status_t) :: status

    train_x(1, :) = [0.0_dp, 0.0_dp]
    train_x(2, :) = [0.4_dp, 0.2_dp]
    train_x(3, :) = [0.8_dp, 0.9_dp]
    train_x(4, :) = [0.2_dp, 0.7_dp]
    train_y = [0.5_dp, -0.2_dp, 0.7_dp, 0.1_dp]
    query_x(1, :) = [0.1_dp, 0.3_dp]
    query_x(2, :) = [0.7_dp, 0.4_dp]
    query_x(3, :) = [0.25_dp, 0.75_dp]

    call history%initialize(dimension, 0, status)
    if (status%code /= FORTNUM_OK) error stop 1
    do i = 1, n_train
        call history%add(train_x(i, :), status, objective=train_y(i))
        if (status%code /= FORTNUM_OK) error stop 1
    end do
    call fortbo_fit_from_history(history, posterior, status, lengthscale=0.3_dp, &
        signal_variance=1.2_dp, noise_variance=0.04_dp, use_gradients=.false.)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR fit ", trim(status%msg)
        error stop 1
    end if
    call posterior%moments(query_x, mean, variance, status)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR moments ", trim(status%msg)
        error stop 1
    end if
    do i = 1, n_query
        write (*, '("MOMENT ", I0, 1X, ES24.16E3, 1X, ES24.16E3)') &
            i - 1, mean(i), variance(i)
    end do
end program fortbo_posterior_reproduction
