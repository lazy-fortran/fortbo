program fortbo_reproduction
    !! Small machine-readable FortBO adapter used by the Python reproduction
    !! runner's synthetic ABI gate. The evaluator is deliberately simple and
    !! remains in the caller: this program only owns ask/tell and policy state.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_turbo_driver, only: fortbo_turbo_config_t, fortbo_turbo_driver_t, &
        FORTBO_TURBO_ACQUISITION_EI, FORTBO_TURBO_ACQUISITION_TS
    implicit none

    integer :: dimension, budget, n_initial, seed
    integer :: used, length
    character(len=64) :: argument
    real(dp), allocatable :: points(:, :), values(:)
    integer, allocatable :: regions(:)
    type(fortbo_turbo_config_t) :: config
    type(fortbo_turbo_driver_t) :: driver
    type(fortnum_status_t) :: status

    if (command_argument_count() < 4 .or. command_argument_count() > 5) then
        print *, "usage: fortbo_reproduction DIMENSION BUDGET N_INITIAL SEED [ei|ts]"
        error stop 2
    end if
    call get_command_argument(1, argument, length=length)
    read (argument(:length), *) dimension
    call get_command_argument(2, argument, length=length)
    read (argument(:length), *) budget
    call get_command_argument(3, argument, length=length)
    read (argument(:length), *) n_initial
    call get_command_argument(4, argument, length=length)
    read (argument(:length), *) seed
    if (command_argument_count() == 5) then
        call get_command_argument(5, argument, length=length)
        select case (trim(adjustl(argument(:length))))
        case ("ei", "qei")
            config%acquisition = FORTBO_TURBO_ACQUISITION_EI
        case ("ts")
            config%acquisition = FORTBO_TURBO_ACQUISITION_TS
        case default
            print *, "unknown acquisition: ", trim(argument(:length))
            error stop 2
        end select
    end if

    config%n_regions = 1
    config%batch_size = 1
    config%n_initial = n_initial
    config%lengthscale = 0.25_dp
    ! The synthetic smoke keeps equal scales for continuity, but exercises the
    ! same explicit ARD posterior branch as the parity fixture.
    allocate (config%lengthscales(dimension))
    config%lengthscales = config%lengthscale
    config%use_ard = .true.
    config%success_tolerance = 10
    config%failure_tolerance = max(4, dimension)
    config%improvement_tolerance = 1.0e-3_dp
    call driver%initialize(dimension, config, seed, status)
    if (status%code /= FORTNUM_OK) then
        print *, "ERROR initialize ", trim(status%msg)
        error stop 1
    end if

    allocate (points(1, dimension), values(1), regions(1))
    used = 0
    do while (used < budget)
        call driver%ask(points, regions, status)
        if (status%code /= FORTNUM_OK) then
            print *, "ERROR ask ", trim(status%msg)
            error stop 1
        end if
        values(1) = sum((points(1, :) - 0.25_dp)**2)
        call driver%tell(points, regions, values, status)
        if (status%code /= FORTNUM_OK) then
            print *, "ERROR tell ", trim(status%msg)
            error stop 1
        end if
        used = used + 1
        ! Real values precede the integer counters so the Python adapter can
        ! parse any dimension without relying on Fortran's default precision.
        write (*, '("ROW ", I0, 1X, I0, 1X)', advance='no') &
            used - 1, regions(1)
        write (*, '(*(ES24.16E3, 1X))', advance='no') points(1, :), values(1), &
            driver%best_value, driver%regions(regions(1))%length
        write (*, '(I0, 1X, I0, 1X, I0)') &
            driver%regions(regions(1))%success_counter, &
            driver%regions(regions(1))%failure_counter, &
            driver%regions(regions(1))%restarts
    end do
end program fortbo_reproduction
