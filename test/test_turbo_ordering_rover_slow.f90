program test_turbo_ordering_rover_slow
    !! The 60-dimensional rover arm of the TuRBO ordering comparison. See
    !! `turbo_ordering_harness` for what is claimed and what is not.

    use turbo_ordering_harness, only: check_ordering, PROBLEM_ROVER
    implicit none

    integer :: failures

    failures = 0
    call check_ordering(PROBLEM_ROVER, "rover-60", failures, &
        single_region_only=.true., report_only=.true.)

    if (failures == 0) then
        print *, "test_turbo_ordering_rover_slow: PASS"
    else
        print *, "test_turbo_ordering_rover_slow: FAIL", failures
        error stop 1
    end if

end program test_turbo_ordering_rover_slow
