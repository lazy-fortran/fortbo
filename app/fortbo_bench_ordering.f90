program fortbo_bench_ordering
    !! Run one TuRBO ordering comparison at a budget the test suite cannot
    !! afford, and print the result for `fortbo-bench` to record.
    !!
    !! This exists because of a measurement, not a preference. `fo` caps a slow
    !! test at five minutes, and the paper's own candidate rule --
    !! `min(100d, 5000)` per region per step -- makes each ask at 200
    !! dimensions cost about fifty times what it costs at fourteen. Inside that
    !! cap the rover and Ackley arms can only reach budgets where a GP has a
    !! couple of dozen points in sixty or two hundred dimensions, and at that
    !! size a trust region contracts around an arbitrary point while undirected
    !! search still covers the space. The ordering is not testable there, and
    !! the test suite says so rather than asserting it.
    !!
    !! So the comparison moves here, where it can run for as long as it needs,
    !! and the *result* is what gets recorded and checked. A benchmark that
    !! takes an hour is a benchmark; a test that takes an hour is not a test.
    !!
    !! Usage:
    !!
    !!     fo exec fortbo_bench_ordering PROBLEM BUDGET N_INITIAL N_REGIONS SEEDS
    !!
    !! where PROBLEM is `ackley`, `rover` or `push`. Every parameter is on the
    !! command line rather than compiled in, so the budget a published number
    !! came from is visible in the command that produced it.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_ordering_bench, only: bench_run_turbo, bench_run_random, &
        bench_problem_id, bench_median
    implicit none

    character(len=64) :: argument
    character(len=32) :: problem_name
    integer :: problem, budget, n_initial, n_regions, n_seeds
    integer :: s, length, status_code
    real(dp), allocatable :: single(:), several(:), random(:)
    real(dp) :: value
    type(fortnum_status_t) :: status
    logical :: ok

    if (command_argument_count() < 5) then
        print *, "usage: fortbo_bench_ordering PROBLEM BUDGET N_INITIAL "// &
            "N_REGIONS SEEDS"
        error stop 2
    end if

    call get_command_argument(1, problem_name, length=length)
    call get_command_argument(2, argument, length=length)
    read (argument(:length), *) budget
    call get_command_argument(3, argument, length=length)
    read (argument(:length), *) n_initial
    call get_command_argument(4, argument, length=length)
    read (argument(:length), *) n_regions
    call get_command_argument(5, argument, length=length)
    read (argument(:length), *) n_seeds

    problem = bench_problem_id(trim(problem_name), ok)
    if (.not. ok) then
        print *, "unknown problem: ", trim(problem_name)
        error stop 2
    end if

    ! The budget must clear the initial designs or the multi-region arm emits
    ! nothing but its initial design -- and because those come from the same
    ! seeded stream the random arm draws from, the two then agree exactly. That
    ! failure is silent and its output looks like a tie, so it is refused here
    ! rather than discovered in the numbers.
    if (budget <= 2*n_regions*n_initial) then
        print *, "budget must exceed twice n_regions*n_initial; got", budget, &
            "against", 2*n_regions*n_initial
        error stop 2
    end if

    allocate (single(n_seeds), several(n_seeds), random(n_seeds))
    do s = 1, n_seeds
        call bench_run_turbo(problem, 1, 100 + s, budget, n_initial, &
            single(s), status)
        if (status%code /= FORTNUM_OK) then
            print *, "turbo-1 failed: ", trim(status%msg)
            error stop 1
        end if
        call bench_run_turbo(problem, n_regions, 100 + s, budget, n_initial, &
            several(s), status)
        if (status%code /= FORTNUM_OK) then
            print *, "turbo-m failed: ", trim(status%msg)
            error stop 1
        end if
        call bench_run_random(problem, 100 + s, budget, random(s), status)
        if (status%code /= FORTNUM_OK) then
            print *, "random failed: ", trim(status%msg)
            error stop 1
        end if
        print *, "seed", 100 + s, " turbo1", single(s), " turbom", several(s), &
            " random", random(s)
    end do

    ! Machine-readable, so the recording script does not have to parse prose.
    print *, "RESULT ", trim(problem_name), budget, n_initial, n_regions, &
        n_seeds, bench_median(single), bench_median(several), &
        bench_median(random)

end program fortbo_bench_ordering
