program fortbo_b5_completion_protocol
    !! Completion-driven ask/tell bridge for external B5 evaluators.
    !!
    !! Up to WORKERS one-point asks are in flight. Each completion identifies
    !! its candidate id, so the evaluator may tell results in any order. The
    !! FortBO driver keeps pending points, uses posterior-mean fantasies while
    !! asking, and records failures without inventing objective values.

    use, intrinsic :: iso_fortran_env, only: input_unit, output_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_turbo_driver, only: fortbo_turbo_config_t, fortbo_turbo_driver_t, &
        FORTBO_TURBO_ACQUISITION_TS
    implicit none

    integer :: dimension, budget, n_initial, seed, n_regions, workers
    integer :: length, ios, slot, candidate_id, result_id
    integer :: dispatched, completed, pending_count
    character(len=512) :: argument, line, command
    real(dp), allocatable :: point(:, :), tell_point(:, :), value(:)
    real(dp), allocatable :: pending_points(:, :)
    integer, allocatable :: region(:), tell_region(:), pending_region(:)
    integer, allocatable :: pending_id(:)
    logical, allocatable :: pending_used(:), successful(:)
    type(fortbo_turbo_config_t) :: config
    type(fortbo_turbo_driver_t) :: driver
    type(fortnum_status_t) :: status

    if (command_argument_count() /= 6) then
        error stop "usage: fortbo_b5_completion_protocol DIMENSION BUDGET "// &
            "N_INITIAL SEED REGIONS WORKERS"
    end if
    call read_argument(1, dimension)
    call read_argument(2, budget)
    call read_argument(3, n_initial)
    call read_argument(4, seed)
    call read_argument(5, n_regions)
    call read_argument(6, workers)
    if (dimension < 1 .or. budget < 1 .or. n_initial < 1 .or. &
            n_regions < 1 .or. workers < 1) then
        error stop "fortbo_b5_completion_protocol: invalid configuration"
    end if

    config%n_regions = n_regions
    config%batch_size = 1
    config%completion_driven = .true.
    config%max_pending = workers
    config%n_initial = n_initial
    config%acquisition = FORTBO_TURBO_ACQUISITION_TS
    config%success_tolerance = 10
    config%failure_tolerance = dimension
    config%improvement_tolerance = 0.0_dp
    call driver%initialize(dimension, config, seed, status)
    if (status%code /= FORTNUM_OK) then
        error stop "fortbo_b5_completion_protocol: driver initialization failed"
    end if

    allocate (point(1, dimension), tell_point(1, dimension), value(1))
    allocate (region(1), tell_region(1), successful(1))
    allocate (pending_points(workers, dimension), pending_region(workers), &
        pending_id(workers), pending_used(workers))
    pending_points = 0.0_dp
    pending_region = 0
    pending_id = -1
    pending_used = .false.
    dispatched = 0
    completed = 0
    pending_count = 0

    do while (completed < budget .or. pending_count > 0)
        if (dispatched < budget .and. pending_count < workers) then
            call driver%ask(point, region, status)
            if (status%code /= FORTNUM_OK) then
                if (pending_count < 1 .or. index(status%msg, &
                        "no region has a surrogate") == 0) then
                    error stop "fortbo_b5_completion_protocol: driver ask failed"
                end if
                write (output_unit, '("WAIT")')
                flush (output_unit)
            else
                slot = first_free_slot(pending_used)
                if (slot == 0) error stop "completion protocol: no pending slot"
                candidate_id = dispatched
                pending_used(slot) = .true.
                pending_id(slot) = candidate_id
                pending_points(slot, :) = point(1, :)
                pending_region(slot) = region(1)
                pending_count = pending_count + 1
                dispatched = dispatched + 1
                write (output_unit, '("ASK 1")')
                write (output_unit, '("POINT ", I0, 1X, I0, 1X, *(ES24.16E3, 1X))') &
                    candidate_id, region(1), point(1, :)
                flush (output_unit)
                cycle
            end if
        end if

        call read_line(line, ios)
        if (ios /= 0) error stop "fortbo_completion_protocol: missing TELL"
        read (line, *, iostat=ios) command, result_id
        if (ios /= 0 .or. trim(command) /= "TELL") then
            error stop "fortbo_completion_protocol: malformed TELL"
        end if
        slot = pending_slot(pending_used, pending_id, result_id)
        if (slot == 0) error stop "completion protocol: unknown candidate id"

        call read_line(line, ios)
        if (ios /= 0) error stop "fortbo_completion_protocol: missing result"
        read (line, *, iostat=ios) command, candidate_id
        if (ios /= 0 .or. candidate_id /= result_id) then
            error stop "fortbo_completion_protocol: malformed result id"
        end if
        value(1) = huge(1.0_dp)
        successful(1) = .false.
        select case (trim(command))
        case ("VALUE")
            read (line, *, iostat=ios) command, candidate_id, value(1)
            if (ios /= 0) error stop "fortbo_completion_protocol: malformed VALUE"
            successful(1) = .true.
        case ("FAIL")
            continue
        case default
            error stop "fortbo_completion_protocol: unknown result command"
        end select

        tell_point(1, :) = pending_points(slot, :)
        tell_region(1) = pending_region(slot)
        call driver%tell(tell_point, tell_region, value, status, &
            successful=successful)
        if (status%code /= FORTNUM_OK) then
            error stop "fortbo_completion_protocol: driver tell failed"
        end if
        pending_used(slot) = .false.
        pending_id(slot) = -1
        pending_count = pending_count - 1
        completed = completed + 1
    end do

    write (output_unit, '("DONE ", I0, 1X, ES24.16E3)') &
        completed, driver%best_value
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

    integer function first_free_slot(used) result(slot)
        logical, intent(in) :: used(:)
        integer :: i

        slot = 0
        do i = 1, size(used)
            if (.not. used(i)) then
                slot = i
                return
            end if
        end do
    end function first_free_slot

    integer function pending_slot(used, ids, wanted) result(slot)
        logical, intent(in) :: used(:)
        integer, intent(in) :: ids(:)
        integer, intent(in) :: wanted
        integer :: i

        slot = 0
        do i = 1, size(used)
            if (used(i) .and. ids(i) == wanted) then
                slot = i
                return
            end if
        end do
    end function pending_slot

end program fortbo_b5_completion_protocol
