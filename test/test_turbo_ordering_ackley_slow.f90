program test_turbo_ordering_ackley_slow
    !! The 200-dimensional Ackley arm of the TuRBO ordering comparison. See
    !! `turbo_ordering_harness` for what is claimed and what is not.

    use turbo_ordering_harness, only: check_ordering, PROBLEM_ACKLEY
    implicit none

    integer :: failures

    failures = 0
    call check_ordering(PROBLEM_ACKLEY, "ackley-200", failures, &
        single_region_only=.true., report_only=.true.)

    if (failures == 0) then
        print *, "test_turbo_ordering_ackley_slow: PASS"
    else
        print *, "test_turbo_ordering_ackley_slow: FAIL", failures
        error stop 1
    end if

end program test_turbo_ordering_ackley_slow
