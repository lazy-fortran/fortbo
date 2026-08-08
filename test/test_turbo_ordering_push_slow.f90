program test_turbo_ordering_push_slow
    !! One arm of the TuRBO ordering comparison. See `turbo_ordering_harness`
    !! for what is being claimed and what is deliberately not.

    use turbo_ordering_harness, only: check_ordering, PROBLEM_PUSH
    implicit none

    integer :: failures

    failures = 0
    call check_ordering(PROBLEM_PUSH, "push-14", failures)

    if (failures == 0) then
        print *, "test_turbo_ordering_push_slow: PASS"
    else
        print *, "test_turbo_ordering_push_slow: FAIL", failures
        error stop 1
    end if

end program test_turbo_ordering_push_slow
