program fortbo_b5_protocol
    !! Line-oriented ask/tell bridge for external B5 evaluators.
    !!
    !! The optimizer remains in FortBO. The caller owns the truth evaluator and
    !! may run the points concurrently. Failed truth calls are sent as FAIL
    !! records: the driver keeps them in history and trust accounting, but does
    !! not train a surrogate on an invented objective.

    use, intrinsic :: iso_fortran_env, only: input_unit, output_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_turbo_driver, only: fortbo_turbo_config_t, fortbo_turbo_driver_t, &
        FORTBO_TURBO_ACQUISITION_TS
    implicit none

    integer :: dimension, budget, n_initial, seed, n_regions, batch_size
    integer :: used, i, length, ios, count, index
    character(len=512) :: argument, line, command
    real(dp), allocatable :: points(:, :), values(:)
    integer, allocatable :: regions(:)
    logical, allocatable :: successful(:)
    type(fortbo_turbo_config_t) :: config
    type(fortbo_turbo_driver_t) :: driver
    type(fortnum_status_t) :: status

    if (command_argument_count() /= 6) then
        error stop "usage: fortbo_b5_protocol DIMENSION BUDGET N_INITIAL SEED REGIONS BATCH_SIZE"
    end if
    call read_argument(1, dimension)
    call read_argument(2, budget)
    call read_argument(3, n_initial)
    call read_argument(4, seed)
    call read_argument(5, n_regions)
    call read_argument(6, batch_size)
    if (dimension < 1 .or. budget < 1 .or. n_initial < 1 .or. &
            n_regions < 1 .or. batch_size < 1 .or. mod(budget, batch_size) /= 0) then
        error stop "fortbo_b5_protocol: invalid configuration"
    end if

    config%n_regions = n_regions
    config%batch_size = batch_size
    config%n_initial = n_initial
    config%acquisition = FORTBO_TURBO_ACQUISITION_TS
    config%success_tolerance = 10
    config%failure_tolerance = dimension
    config%improvement_tolerance = 0.0_dp
    call driver%initialize(dimension, config, seed, status)
    if (status%code /= FORTNUM_OK) then
        error stop "fortbo_b5_protocol: driver initialization failed"
    end if

    allocate (points(batch_size, dimension), values(batch_size))
    allocate (regions(batch_size), successful(batch_size))
    used = 0
    do while (used < budget)
        call driver%ask(points, regions, status)
        if (status%code /= FORTNUM_OK) then
            error stop "fortbo_b5_protocol: driver ask failed"
        end if
        write (output_unit, '("ASK ", I0)') batch_size
        do i = 1, batch_size
            write (output_unit, '("POINT ", I0, 1X, I0, 1X, *(ES24.16E3, 1X))') &
                i, regions(i), points(i, :)
        end do
        flush (output_unit)

        call read_line(line, ios)
        if (ios /= 0) error stop "fortbo_b5_protocol: missing TELL header"
        read (line, *, iostat=ios) command, count
        if (ios /= 0 .or. trim(command) /= "TELL" .or. count /= batch_size) then
            error stop "fortbo_b5_protocol: malformed TELL header"
        end if
        values = huge(1.0_dp)
        successful = .false.
        do i = 1, batch_size
            call read_line(line, ios)
            if (ios /= 0) error stop "fortbo_b5_protocol: missing result"
            read (line, *, iostat=ios) command, index
            if (ios /= 0 .or. index < 1 .or. index > batch_size) then
                error stop "fortbo_b5_protocol: malformed result index"
            end if
            select case (trim(command))
            case ("VALUE")
                read (line, *, iostat=ios) command, index, values(index)
                if (ios /= 0) error stop "fortbo_b5_protocol: malformed VALUE"
                successful(index) = .true.
            case ("FAIL")
                successful(index) = .false.
            case default
                error stop "fortbo_b5_protocol: unknown result command"
            end select
        end do
        call driver%tell(points, regions, values, status, successful=successful)
        if (status%code /= FORTNUM_OK) then
            error stop "fortbo_b5_protocol: driver tell failed"
        end if
        used = used + batch_size
    end do
    write (output_unit, '("DONE ", I0, 1X, ES24.16E3)') used, driver%best_value
    flush (output_unit)

contains

    subroutine read_argument(number, value)
        integer, intent(in) :: number
        integer, intent(out) :: value

        call get_command_argument(number, argument, length=length)
        read (argument(:length), *) value
    end subroutine read_argument

    subroutine read_line(value, io_status)
        character(len=*), intent(out) :: value
        integer, intent(out) :: io_status

        read (input_unit, '(A)', iostat=io_status) value
    end subroutine read_line

end program fortbo_b5_protocol
