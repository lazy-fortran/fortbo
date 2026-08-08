program test_metrics
    !! BO6: run metrics.
    !!
    !! The oracle is the recorded run: the test knows what it fed in, so it
    !! knows what the metrics must say. What is checked is the set of decisions
    !! that separate an honest benchmark record from a flattering one:
    !!
    !!   * an infeasible point never sets the incumbent, however good its
    !!     objective value;
    !!   * regret is *unavailable*, not large, before anything feasible exists;
    !!   * cumulative regret charges every evaluation, so a method that finds
    !!     the optimum and then wanders is distinguishable from one that stops;
    !!   * effective sample size is checked against the two cases whose answers
    !!     are known exactly, uniform and degenerate weights.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_metrics, only: fortbo_metrics_t, fortbo_effective_sample_size
    implicit none

    integer :: failures

    failures = 0
    call check_infeasible_never_sets_the_incumbent(failures)
    call check_regret_is_unavailable_before_feasibility(failures)
    call check_cumulative_regret_charges_every_evaluation(failures)
    call check_counters_and_violations(failures)
    call check_effective_sample_size(failures)
    call check_growth(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_metrics: PASS"
    else
        print *, "test_metrics: FAIL", failures
        error stop 1
    end if

contains

    !! The decision that matters most in a constrained record. An infeasible
    !! point with a wonderful value is not a solution, and if it sets the
    !! incumbent the run reports solving a problem it never solved.
    subroutine check_infeasible_never_sets_the_incumbent(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: metrics
        type(fortnum_status_t) :: status
        real(dp) :: regret
        logical :: available

        call metrics%initialize(status, optimum=0.0_dp)
        call expect(status%code == FORTNUM_OK, "metrics initialize", failures)

        call metrics%record(5.0_dp, status, feasible=.true.)
        ! Far better, and inadmissible.
        call metrics%record(-100.0_dp, status, feasible=.false., violation=3.0_dp)
        call metrics%record(4.0_dp, status, feasible=.true.)

        call expect(metrics%rows(2)%best_feasible == 5.0_dp, &
            "an infeasible point does not become the incumbent", failures)
        call expect(metrics%rows(3)%best_feasible == 4.0_dp, &
            "a feasible improvement does become the incumbent", failures)

        call metrics%simple_regret(regret, available)
        call expect(available .and. regret == 4.0_dp, &
            "final regret is measured from the best feasible value", failures)
        call expect(metrics%feasible_count() == 2, &
            "the feasible evaluations are counted", failures)
    end subroutine check_infeasible_never_sets_the_incumbent

    !! Before anything feasible exists there is no regret to report, and
    !! reporting a large number instead would be indistinguishable from having
    !! found something bad.
    subroutine check_regret_is_unavailable_before_feasibility(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: metrics
        type(fortnum_status_t) :: status
        real(dp) :: regret
        logical :: available

        call metrics%initialize(status, optimum=0.0_dp)
        call metrics%record(1.0_dp, status, feasible=.false., violation=0.5_dp)
        call metrics%record(2.0_dp, status, feasible=.false., violation=0.25_dp)

        call expect(.not. metrics%rows(1)%regret_available, &
            "regret is unavailable while nothing feasible has been seen", failures)
        call metrics%simple_regret(regret, available)
        call expect(.not. available, &
            "the final regret is reported as unavailable, not as a number", &
            failures)

        call metrics%record(3.0_dp, status, feasible=.true.)
        call expect(metrics%rows(3)%regret_available, &
            "the first feasible point makes regret available", failures)
        call metrics%simple_regret(regret, available)
        call expect(available .and. regret == 3.0_dp, &
            "regret becomes measurable once feasibility is reached", failures)

        ! A run with no known optimum gets best-value curves, not regret against
        ! a guess.
        block
            type(fortbo_metrics_t) :: unknown
            call unknown%initialize(status)
            call unknown%record(1.0_dp, status)
            call unknown%simple_regret(regret, available)
            call expect(.not. available, &
                "a run without a known optimum reports no regret", failures)
            call expect(unknown%rows(1)%best_feasible == 1.0_dp, &
                "the best value is still recorded without an optimum", failures)
        end block
    end subroutine check_regret_is_unavailable_before_feasibility

    !! Two runs reaching the same answer at the same evaluation, one of which
    !! then keeps searching badly. Simple regret cannot tell them apart;
    !! cumulative regret must.
    subroutine check_cumulative_regret_charges_every_evaluation(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: quick, wandering
        type(fortnum_status_t) :: status
        integer :: k
        real(dp) :: quick_regret, wandering_regret
        logical :: available_a, available_b

        call quick%initialize(status, optimum=0.0_dp)
        call wandering%initialize(status, optimum=0.0_dp)

        ! Both find 1.0 immediately.
        call quick%record(1.0_dp, status)
        call wandering%record(1.0_dp, status)
        ! Both then spend nine more evaluations. Neither improves.
        do k = 1, 9
            call quick%record(1.0_dp, status)
            call wandering%record(50.0_dp, status)
        end do

        call quick%simple_regret(quick_regret, available_a)
        call wandering%simple_regret(wandering_regret, available_b)
        call expect(quick_regret == wandering_regret, &
            "simple regret cannot distinguish the two runs", failures)
        call expect(abs(quick%cumulative_regret - 10.0_dp) < 1.0e-12_dp, &
            "cumulative regret charges every evaluation, not just improvements", &
            failures)
        call expect(abs(wandering%cumulative_regret &
            - quick%cumulative_regret) < 1.0e-12_dp, &
            "cumulative regret follows the incumbent, not the sampled value", &
            failures)

        ! Cumulative regret must never decrease.
        block
            logical :: monotone
            monotone = .true.
            do k = 2, quick%count
                if (quick%rows(k)%cumulative_regret &
                    < quick%rows(k - 1)%cumulative_regret) monotone = .false.
            end do
            call expect(monotone, "cumulative regret never decreases", failures)
        end block
    end subroutine check_cumulative_regret_charges_every_evaluation

    subroutine check_counters_and_violations(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: metrics
        type(fortnum_status_t) :: status

        call metrics%initialize(status, optimum=0.0_dp)
        call metrics%record(1.0_dp, status, acquisition_evaluations=5000, &
            gradient_evaluations=12, wall_seconds=0.25_dp, peak_megabytes=64.0_dp, &
            host_device_transfers=3)
        call metrics%record(0.5_dp, status, feasible=.false., violation=1.5_dp, &
            acquisition_evaluations=4000, gradient_evaluations=8)

        ! The whole point of separate counters: one objective evaluation can
        ! hide thousands of acquisition evaluations, and a regret-per-objective
        ! plot would not show it.
        call expect(metrics%rows(1)%acquisition_evaluations == 5000 .and. &
            metrics%rows(1)%gradient_evaluations == 12, &
            "acquisition and gradient work are counted separately", failures)
        call expect(abs(metrics%total_violations() - 1.5_dp) < 1.0e-12_dp, &
            "constraint violations are accumulated", failures)
        call expect(metrics%rows(1)%wall_seconds == 0.25_dp .and. &
            metrics%rows(1)%peak_megabytes == 64.0_dp .and. &
            metrics%rows(1)%host_device_transfers == 3, &
            "machine metrics are recorded alongside the method's", failures)
    end subroutine check_counters_and_violations

    subroutine check_effective_sample_size(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: weights(8), ess

        ! Uniform weights: every draw counts, so the ESS is the sample size.
        weights = 1.0_dp
        call fortbo_effective_sample_size(weights, ess, status)
        call expect(status%code == FORTNUM_OK .and. abs(ess - 8.0_dp) < 1.0e-12_dp, &
            "uniform weights give an effective sample size of the full sample", &
            failures)

        ! Scale invariance: the weights need not be normalized.
        weights = 37.5_dp
        call fortbo_effective_sample_size(weights, ess, status)
        call expect(abs(ess - 8.0_dp) < 1.0e-12_dp, &
            "the effective sample size does not depend on the weight scale", &
            failures)

        ! One weight carrying everything: the average rests on a single draw.
        weights = 0.0_dp
        weights(3) = 1.0_dp
        call fortbo_effective_sample_size(weights, ess, status)
        call expect(abs(ess - 1.0_dp) < 1.0e-12_dp, &
            "a degenerate weight set has an effective sample size of one", &
            failures)

        ! Anything in between must lie strictly between the two.
        weights = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 20.0_dp]
        call fortbo_effective_sample_size(weights, ess, status)
        call expect(ess > 1.0_dp .and. ess < 8.0_dp, &
            "an uneven weight set falls between the two extremes", failures)
    end subroutine check_effective_sample_size

    subroutine check_growth(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: metrics
        type(fortnum_status_t) :: status
        integer, parameter :: n = 500
        integer :: k
        logical :: ordered

        call metrics%initialize(status, optimum=0.0_dp)
        do k = 1, n
            call metrics%record(real(n - k, dp), status)
        end do
        call expect(metrics%count == n, "every evaluation is kept", failures)

        ordered = .true.
        do k = 1, n
            if (metrics%rows(k)%evaluation /= k) ordered = .false.
        end do
        call expect(ordered, "rows stay in run order across a resize", failures)
        call expect(metrics%rows(n)%best_feasible == 0.0_dp, &
            "the incumbent survives the resize", failures)
    end subroutine check_growth

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_metrics_t) :: metrics, fresh
        type(fortnum_status_t) :: status
        real(dp) :: ess
        real(dp) :: empty(0)

        call fresh%record(1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "recording into uninitialized metrics is refused", failures)

        call metrics%initialize(status, optimum=0.0_dp)
        call metrics%record(1.0_dp, status, violation=-1.0_dp)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative violation is refused", failures)

        call fortbo_effective_sample_size(empty, ess, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an empty weight set is refused", failures)

        call fortbo_effective_sample_size([0.0_dp, 0.0_dp], ess, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "all-zero weights are refused rather than dividing by zero", failures)

        call fortbo_effective_sample_size([-1.0_dp, 1.0_dp], ess, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative weight is refused", failures)
    end subroutine check_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_metrics
