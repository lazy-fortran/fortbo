program test_recorded_ordering
    !! BO5: the TuRBO paper's ordering, asserted against recorded benchmark runs.
    !!
    !! **Why this reads a fixture instead of running the comparison.** The
    !! budgets where the ordering is actually testable do not fit inside `fo`'s
    !! five-minute cap on a slow test, and that is a measurement rather than an
    !! excuse. The paper's own candidate rule -- `min(100d, 5000)` per region
    !! per step -- makes each ask expensive, and cost grows roughly with the
    !! square of the budget. The consequence was measured, not assumed:
    !!
    !!   * rover-60 at 22 evaluations: TuRBO-1 scores 1331 against random
    !!     search's 1190. It **loses**, and it should — a GP holding two dozen
    !!     points in sixty dimensions carries almost no information, so the
    !!     trust region contracts around an arbitrary point while undirected
    !!     search still covers the space.
    !!   * rover-60 at 60 evaluations with one region: 1047 against 1191. It
    !!     wins by about twelve percent, but says nothing about several
    !!     regions.
    !!   * rover-60 at 80 evaluations with three regions: the paper's *full*
    !!     ordering, TuRBO-m ahead of TuRBO-1 ahead of random.
    !!
    !! So the comparison runs in `fortbo-bench` where it can take as long as it
    !! needs, and its result is committed. This test asserts against that
    !! record. That is a **weaker** guarantee than a live run and is labelled
    !! as such rather than presented as equivalent: it catches a regression in
    !! the recorded numbers, not one introduced since they were recorded. The
    !! regenerating command is in the fixture so the numbers cannot become
    !! folklore.
    !!
    !! A missing fixture is a failure, not a skip. A test that passes with
    !! nothing to check against looks exactly like one that ran.

    use fortnum_kinds, only: dp
    implicit none

    integer :: failures

    failures = 0
    call check_recorded_ordering(failures)

    if (failures == 0) then
        print *, "test_recorded_ordering: PASS"
    else
        print *, "test_recorded_ordering: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_recorded_ordering(failures)
        integer, intent(inout) :: failures
        character(len=32) :: problem
        real(dp) :: turbo1, turbom, random
        integer :: count, budget, regions, unit, ios, k
        logical :: present_on_disk, saw_rover, saw_full_ordering

        inquire (file="test/fixtures/fortbo_ordering.txt", exist=present_on_disk)
        if (.not. present_on_disk) then
            print *, "  FAIL: the recorded ordering fixture is missing; "// &
                "regenerate with fortbo-bench/scripts/record_fortbo_ordering.py"
            failures = failures + 1
            return
        end if

        open (newunit=unit, file="test/fixtures/fortbo_ordering.txt", &
            status="old", action="read", iostat=ios)
        if (ios /= 0) then
            print *, "  FAIL: the recorded ordering fixture could not be opened"
            failures = failures + 1
            return
        end if

        call skip_comments(unit)
        read (unit, *) count
        call expect(count >= 1, "the fixture records at least one run", failures)

        saw_rover = .false.
        saw_full_ordering = .false.
        call skip_comments(unit)
        do k = 1, count
            read (unit, *, iostat=ios) problem, budget, regions, turbo1, &
                turbom, random
            if (ios /= 0) then
                print *, "  FAIL: the fixture is truncated"
                failures = failures + 1
                exit
            end if

            print *, "  ", trim(problem), " budget", budget, " regions", &
                regions
            print *, "      turbo-1", turbo1, " turbo-m", turbom, &
                " random", random

            ! The claim the paper makes and every recorded run must show:
            ! directed local search beats undirected search. If this fails the
            ! method is not doing anything, whatever else is true.
            call expect(turbo1 < random, &
                trim(problem)//": one trust region beats quasi-random search", &
                failures)

            if (regions > 1) then
                call expect(turbom < random, &
                    trim(problem)//": several trust regions beat "// &
                    "quasi-random search", failures)
            end if

            if (trim(problem) == "rover") then
                saw_rover = .true.
                ! The paper's full ordering, where the budget allows it to
                ! appear at all. This is the assertion that was deliberately
                ! *not* made at smaller budgets, because there TuRBO-1 wins
                ! and tuning until the paper's ordering showed would have made
                ! the test a record of that search rather than evidence.
                if (regions > 1 .and. budget >= 80) then
                    saw_full_ordering = turbom < turbo1
                    call expect(saw_full_ordering, &
                        "rover: several trust regions beat one at a budget "// &
                        "that lets them", failures)
                end if
            end if
        end do
        close (unit)

        call expect(saw_rover, &
            "the rover arm is present, not quietly dropped", failures)
        call expect(saw_full_ordering, &
            "the paper's full ordering is reproduced somewhere in the record", &
            failures)
    end subroutine check_recorded_ordering

    subroutine skip_comments(unit)
        integer, intent(in) :: unit
        character(len=256) :: line
        integer :: ios

        do
            read (unit, "(a)", iostat=ios) line
            if (ios /= 0) return
            if (len_trim(line) == 0) cycle
            if (line(1:1) == "#") cycle
            backspace (unit)
            return
        end do
    end subroutine skip_comments

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_recorded_ordering
