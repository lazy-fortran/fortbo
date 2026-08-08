program test_turbo_ordering_ackley_slow
    !! The 200-dimensional Ackley arm, and the one place FortBO's TuRBO is
    !! compared against the pinned `uber-research/TuRBO`. See
    !! `turbo_ordering_harness` for what is claimed and what is not.

    use fortnum_kinds, only: dp
    use turbo_ordering_harness, only: check_ordering, &
        check_against_pinned_reference, PROBLEM_ACKLEY
    implicit none

    integer :: failures
    real(dp) :: fortbo_best

    failures = 0
    call check_ordering(PROBLEM_ACKLEY, "ackley-200", failures, &
        single_region_only=.true., report_only=.true., &
        single_median=fortbo_best)
    call check_against_pinned_reference(fortbo_best, failures)

    if (failures == 0) then
        print *, "test_turbo_ordering_ackley_slow: PASS"
    else
        print *, "test_turbo_ordering_ackley_slow: FAIL", failures
        error stop 1
    end if

end program test_turbo_ordering_ackley_slow
