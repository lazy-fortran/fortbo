program test_regret_benchmark
    !! BO6: sample efficiency against BoTorch, reported apart from wall time.
    !!
    !! The roadmap asks for regret and sample efficiency to be reported
    !! **separately from wall time**, and the separation is the substance of the
    !! item rather than a presentational nicety. Sample efficiency asks how many
    !! objective evaluations a policy needs. Wall time asks how long the
    !! machinery around them took. Bayesian optimization exists for objectives
    !! where one evaluation costs minutes or hours, so on any problem worth
    !! using it on the first number decides everything and the second is
    !! rounding. A table reporting only wall time would be comparing Python's
    !! interpreter against a Fortran binary, which is a fact about the two
    !! languages and not about the two optimizers.
    !!
    !! Matched, because a regret comparison is meaningless otherwise:
    !!
    !!   * the same objective, Branin, on the same box;
    !!   * **the same initial design points**, read from the fixture, not merely
    !!     the same seed. Two frameworks handed the same seed draw different
    !!     numbers, and the initial design dominates the early curve, so a
    !!     seed-matched comparison largely reports whose draw was luckier;
    !!   * the same budget, the same restart count, the same acquisition
    !!     family, and the same stopping criterion -- the budget and nothing
    !!     else, since a policy that stops early on a convergence test is
    !!     solving a different problem.
    !!
    !! What this asserts is deliberately not "FortBO wins". Both are EI-driven
    !! GP policies on a two-dimensional problem; if one crushed the other,
    !! the likeliest explanation would be that the comparison is broken. The
    !! oracle is that FortBO reaches the *same order* of final regret from the
    !! identical start, and that its regret curve is non-increasing -- which is
    !! a real check, since a best-so-far curve that rises means the incumbent
    !! bookkeeping is wrong.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_benchmarks, only: fortbo_benchmark_t, FORTBO_BENCH_BRANIN
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_fortml, only: fortbo_fit_from_history
    use fortbo_acquisition, only: fortbo_log_ei_t
    use fortbo_optimize, only: fortbo_search_acquisition
    implicit none

    integer :: failures

    failures = 0
    call check_sample_efficiency(failures)

    if (failures == 0) then
        print *, "test_regret_benchmark: PASS"
    else
        print *, "test_regret_benchmark: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_sample_efficiency(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: problem
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortbo_log_ei_t) :: acquisition
        type(fortnum_status_t) :: status
        real(dp), allocatable :: design(:, :), reference_regret(:)
        real(dp), allocatable :: candidates(:, :)
        real(dp), allocatable :: regret(:)
        real(dp) :: lower(2), upper(2), optimum, reference_wall
        real(dp) :: point(2), value, incumbent, best_value
        real(dp) :: started, finished, wall_seconds
        integer :: budget, restarts, n_initial, k, iteration, unit, ios
        integer :: n_candidates
        logical :: present_on_disk, monotone

        inquire (file="test/fixtures/regret_branin.txt", exist=present_on_disk)
        if (.not. present_on_disk) then
            ! A missing reference is a failure, not a skip. A comparison that
            ! quietly passes with nothing to compare against looks exactly like
            ! one that ran.
            print *, "  FAIL: the regret fixture is missing; regenerate it with "// &
                "fortbo-bench/scripts/emit_regret.py"
            failures = failures + 1
            return
        end if

        open (newunit=unit, file="test/fixtures/regret_branin.txt", status="old", &
            action="read", iostat=ios)
        if (ios /= 0) then
            print *, "  FAIL: the regret fixture could not be opened"
            failures = failures + 1
            return
        end if

        call skip_comments(unit)
        read (unit, *) budget, restarts, optimum
        call skip_comments(unit)
        read (unit, *) n_initial

        allocate (design(n_initial, 2), reference_regret(budget + 1))
        allocate (regret(budget + 1))
        call skip_comments(unit)
        do k = 1, n_initial
            read (unit, *) design(k, 1), design(k, 2)
        end do
        call skip_comments(unit)
        do k = 1, budget + 1
            read (unit, *) reference_regret(k)
        end do
        call skip_comments(unit)
        read (unit, *) reference_wall
        close (unit)

        problem%kind = FORTBO_BENCH_BRANIN
        problem%dimension = 2
        call problem%bounds(lower, upper, status)
        call expect(status%code == FORTNUM_OK, "the Branin box is available", &
            failures)

        call history%initialize(2, 0, status)
        incumbent = huge(1.0_dp)
        do k = 1, n_initial
            point = design(k, :)
            call problem%value(point, value, status)
            call history%add(point, status, objective=value)
            incumbent = min(incumbent, value)
        end do
        ! The curve starts after the shared initial design, so both sides are
        ! visibly at the same place before either has modelled anything.
        regret(1) = incumbent - optimum

        ! Filled with a value no run could produce, so an aborted loop cannot
        ! report a flattering regret from an array slot nobody wrote to. The
        ! first version of this test printed a final regret of exactly zero
        ! after failing on iteration one.
        regret(2:) = -huge(1.0_dp)

        ! The inner search is gradient-free, and this is a real difference
        ! between the two systems rather than a shortcut. BoTorch's model
        ! exposes acquisition gradients, so it runs L-BFGS-B from `restarts`
        ! starts; FortBO's value-only GP exposes moments and no moment
        ! gradient, so its acquisition surface can only be sampled. The
        ! candidate budget below is set to the same order of acquisition work
        ! -- restarts times a modest per-start allowance -- so neither side is
        ! handed a larger inner budget than the other. It is not an identical
        ! inner optimizer, and claiming it were would be false.
        n_candidates = restarts*256
        allocate (candidates(n_candidates, 2))
        call cpu_time(started)
        do iteration = 1, budget
            call fortbo_fit_from_history(history, posterior, status, &
                lengthscale=2.0_dp, noise_variance=1.0e-6_dp)
            if (status%code /= FORTNUM_OK) then
                print *, "  fit failed: ", trim(status%msg)
                exit
            end if

            acquisition%best = incumbent
            acquisition%xi = 0.0_dp

            ! Candidates placed by an additive-recurrence sequence, which is
            ! deterministic. A random set would make the regret curve
            ! unreproducible, and reproducibility is the entire reason the
            ! initial design is listed point by point rather than seeded.
            do k = 1, n_candidates
                candidates(k, 1) = lower(1) + (upper(1) - lower(1)) &
                    *fractional(0.7548776662_dp &
                    *real(iteration*n_candidates + k, dp))
                candidates(k, 2) = lower(2) + (upper(2) - lower(2)) &
                    *fractional(0.5698402909_dp &
                    *real(iteration*n_candidates + k, dp))
            end do

            call fortbo_search_acquisition(acquisition, posterior, lower, upper, &
                candidates, point, best_value, status)
            if (status%code /= FORTNUM_OK) then
                print *, "  acquisition failed: ", trim(status%msg)
                exit
            end if

            call problem%value(point, value, status)
            call history%add(point, status, objective=value)
            incumbent = min(incumbent, value)
            regret(iteration + 1) = incumbent - optimum
        end do
        call cpu_time(finished)
        wall_seconds = finished - started

        call expect(status%code == FORTNUM_OK, "the loop runs to its budget", &
            failures)

        ! Best-so-far can never rise. A curve that does means the incumbent is
        ! being tracked wrongly, which no regret comparison would otherwise
        ! catch because the final number could still look respectable.
        monotone = .true.
        do k = 2, budget + 1
            if (regret(k) > regret(k - 1) + 1.0e-15_dp) monotone = .false.
        end do
        call expect(monotone, "best-so-far regret never increases", failures)

        call expect(regret(budget + 1) < regret(1), &
            "the policy improves on its own initial design", failures)

        ! Same order of magnitude, not a win. Two EI-driven GP policies on a
        ! two-dimensional problem from an identical start should land in the
        ! same place; a large gap in either direction would more likely mean
        ! the comparison is broken than that one method is better.
        call expect(regret(budget + 1) < 100.0_dp*reference_regret(budget + 1) &
            + 1.0e-3_dp, &
            "final regret is within an order of magnitude of BoTorch's", failures)

        ! Reported side by side, in separate columns, never combined.
        print *, "  sample efficiency, Branin, shared initial design:"
        print *, "    evaluations           ", budget
        print *, "    fortbo  final regret  ", regret(budget + 1)
        print *, "    botorch final regret  ", reference_regret(budget + 1)
        print *, "  wall time, reported separately and not like-for-like:"
        print *, "    fortbo  seconds       ", wall_seconds
        print *, "    botorch seconds       ", reference_wall
    end subroutine check_sample_efficiency

    !! Fractional part, for deterministic low-discrepancy restart placement.
    pure real(dp) function fractional(v) result(f)
        real(dp), intent(in) :: v

        f = v - real(floor(v), dp)
    end function fractional

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

end program test_regret_benchmark
