module fortbo_ordering_bench
    !! The TuRBO ordering comparison, as a reusable runner.
    !!
    !! This lives in the library rather than in the test suite because the same
    !! comparison has to run in two places with two different budgets. Inside
    !! `fo`'s five-minute cap on a slow test, the rover and Ackley arms can only
    !! reach budgets where a GP holds a couple of dozen points in sixty or two
    !! hundred dimensions -- too few for a trust region to do anything a random
    !! sweep does not. Outside it, through `app/fortbo_ordering_bench`, the same
    !! code runs long enough for the ordering to be testable at all.
    !!
    !! One implementation for both, so a published benchmark number and a test
    !! assertion cannot be measuring subtly different things.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_benchmarks, only: fortbo_benchmark_t, FORTBO_BENCH_ACKLEY
    use fortbo_rover, only: fortbo_rover_t, FORTBO_ROVER_DIMENSION
    use fortbo_push, only: fortbo_push_t, FORTBO_PUSH_DIMENSION
    use fortbo_turbo_driver, only: fortbo_turbo_driver_t, fortbo_turbo_config_t
    implicit none
    private

    integer, parameter, public :: FORTBO_BENCH_PROBLEM_ACKLEY = 1
    integer, parameter, public :: FORTBO_BENCH_PROBLEM_ROVER = 2
    integer, parameter, public :: FORTBO_BENCH_PROBLEM_PUSH = 3

    public :: bench_problem_id
    public :: bench_problem_width
    public :: bench_run_turbo
    public :: bench_run_random
    public :: bench_median

contains

    !! Name to identifier, so a benchmark is selected by a word on a command
    !! line rather than by a number nobody can read back.
    integer function bench_problem_id(name, ok) result(problem)
        character(len=*), intent(in) :: name
        logical, intent(out) :: ok

        ok = .true.
        select case (name)
        case ("ackley")
            problem = FORTBO_BENCH_PROBLEM_ACKLEY
        case ("rover")
            problem = FORTBO_BENCH_PROBLEM_ROVER
        case ("push")
            problem = FORTBO_BENCH_PROBLEM_PUSH
        case default
            problem = 0
            ok = .false.
        end select
    end function bench_problem_id

    pure integer function bench_problem_width(problem) result(width)
        integer, intent(in) :: problem

        select case (problem)
        case (FORTBO_BENCH_PROBLEM_ACKLEY)
            width = 200
        case (FORTBO_BENCH_PROBLEM_ROVER)
            width = FORTBO_ROVER_DIMENSION
        case default
            width = FORTBO_PUSH_DIMENSION
        end select
    end function bench_problem_width

    subroutine bench_bounds(problem, lower, upper)
        integer, intent(in) :: problem
        real(dp), intent(out) :: lower(:), upper(:)
        type(fortbo_benchmark_t) :: ackley
        type(fortbo_rover_t) :: rover
        type(fortbo_push_t) :: push
        type(fortnum_status_t) :: status

        select case (problem)
        case (FORTBO_BENCH_PROBLEM_ACKLEY)
            ackley%kind = FORTBO_BENCH_ACKLEY
            ackley%dimension = size(lower)
            call ackley%bounds(lower, upper, status)
        case (FORTBO_BENCH_PROBLEM_ROVER)
            call rover%bounds(lower, upper, status)
        case default
            call push%bounds(lower, upper, status)
        end select
    end subroutine bench_bounds

    subroutine bench_value(problem, x, value, status)
        integer, intent(in) :: problem
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_benchmark_t) :: ackley
        type(fortbo_rover_t) :: rover
        type(fortbo_push_t) :: push

        select case (problem)
        case (FORTBO_BENCH_PROBLEM_ACKLEY)
            ackley%kind = FORTBO_BENCH_ACKLEY
            ackley%dimension = size(x)
            call ackley%value(x, value, status)
        case (FORTBO_BENCH_PROBLEM_ROVER)
            call rover%value(x, value, status)
        case default
            call push%value(x, value, status)
        end select
    end subroutine bench_value

    subroutine bench_run_turbo(problem, n_regions, seed, budget, n_initial, best, status)
        integer, intent(in) :: problem, n_regions, seed, budget, n_initial
        real(dp), intent(out) :: best
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_turbo_driver_t) :: driver
        type(fortbo_turbo_config_t) :: config
        real(dp), allocatable :: lower(:), upper(:)
        real(dp), allocatable :: points(:, :), values(:), scaled(:)
        integer, allocatable :: regions(:)
        integer :: n_inputs
        integer :: evaluations, k

        n_inputs = bench_problem_width(problem)
        allocate (lower(n_inputs), upper(n_inputs), scaled(n_inputs))
        call bench_bounds(problem, lower, upper)

        config%n_regions = n_regions
        config%batch_size = 1
        ! Stated, not defaulted to the paper's 2*d: at 200 dimensions that rule
        ! would spend 400 evaluations per region on the initial design alone
        ! and the comparison would be entirely about initial designs. The same
        ! number is used for every arm, which is what makes the arms
        ! comparable even though it is not the paper's number.
        config%n_initial = n_initial
        config%use_gradients = .false.
        config%quasi_random = .true.

        call driver%initialize(n_inputs, config, seed, status)
        if (status%code /= FORTNUM_OK) return

        allocate (points(1, n_inputs), regions(1), values(1))
        best = huge(1.0_dp)
        evaluations = 0
        do while (evaluations < budget)
            call driver%ask(points, regions, status)
            if (status%code /= FORTNUM_OK) return
            ! The driver works on the unit cube; the problems have their own
            ! boxes.
            do k = 1, n_inputs
                scaled(k) = lower(k) + points(1, k)*(upper(k) - lower(k))
            end do
            call bench_value(problem, scaled, values(1), status)
            if (status%code /= FORTNUM_OK) return
            call driver%tell(points, regions, values, status)
            if (status%code /= FORTNUM_OK) return
            best = min(best, values(1))
            evaluations = evaluations + 1
        end do
    end subroutine bench_run_turbo

    subroutine bench_run_random(problem, seed, budget, best, status)
        integer, intent(in) :: problem, seed, budget
        real(dp), intent(out) :: best
        type(fortnum_status_t), intent(out) :: status
        type(rng_t) :: generator
        real(dp), allocatable :: lower(:), upper(:), scaled(:)
        real(dp) :: value, draw
        integer :: n_inputs, evaluations, k

        n_inputs = bench_problem_width(problem)
        allocate (lower(n_inputs), upper(n_inputs), scaled(n_inputs))
        call bench_bounds(problem, lower, upper)

        call rng_seed(generator, int(seed, kind(1_8)), status)
        best = huge(1.0_dp)
        do evaluations = 1, budget
            do k = 1, n_inputs
                call rng_uniform(generator, draw)
                scaled(k) = lower(k) + draw*(upper(k) - lower(k))
            end do
            call bench_value(problem, scaled, value, status)
            if (status%code /= FORTNUM_OK) return
            best = min(best, value)
        end do
    end subroutine bench_run_random

    pure real(dp) function bench_median(values) result(middle)
        real(dp), intent(in) :: values(:)
        real(dp) :: sorted(size(values))
        real(dp) :: swap
        integer :: i, j

        sorted = values
        do i = 1, size(sorted) - 1
            do j = i + 1, size(sorted)
                if (sorted(j) < sorted(i)) then
                    swap = sorted(i)
                    sorted(i) = sorted(j)
                    sorted(j) = swap
                end if
            end do
        end do
        middle = sorted((size(sorted) + 1)/2)
    end function bench_median

end module fortbo_ordering_bench
