program test_stopping
    !! BO4: stopping rules.
    !!
    !! Oracles are behavioral. Each rule is driven to its boundary and checked
    !! on both sides of it, and the priority order is checked by constructing
    !! states where two rules would both fire — a rule that reports the wrong
    !! reason is as bad as one that fires at the wrong time, because the reason
    !! is what tells a caller whether to trust the answer.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR
    use fortbo_stopping, only: fortbo_stopping_rule_t, fortbo_stopping_state_t, &
        fortbo_stop_reason_name, FORTBO_CONTINUE, FORTBO_STOP_BUDGET, &
        FORTBO_STOP_COST, FORTBO_STOP_WALL_TIME, FORTBO_STOP_TARGET_REACHED, &
        FORTBO_STOP_NO_IMPROVEMENT, FORTBO_STOP_UNCERTAINTY, FORTBO_STOP_ACQUISITION
    implicit none

    integer :: failures

    failures = 0
    call check_defaults_never_stop(failures)
    call check_budget(failures)
    call check_target(failures)
    call check_patience(failures)
    call check_uncertainty_and_acquisition(failures)
    call check_priority_order(failures)
    call check_reason_names(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_stopping: PASS"
    else
        print *, "test_stopping: FAIL", failures
        error stop 1
    end if

contains

    !! A default rule must never stop a run. Anything else would mean a caller
    !! that forgot to configure stopping silently gets a truncated run.
    subroutine check_defaults_never_stop(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason, k

        do k = 1, 100000, 9999
            state%evaluations = k
            state%total_cost = real(k, dp)
            state%wall_seconds = real(k, dp)
            state%incumbent_value = -real(k, dp)
            state%stalled_iterations = k
            state%max_posterior_sd = 0.0_dp
            state%best_acquisition = 0.0_dp
            state%has_feasible = .true.
            call rule%evaluate(state, reason, status)
            call expect(reason == FORTBO_CONTINUE, &
                "an unconfigured rule never stops", failures)
        end do
    end subroutine check_defaults_never_stop

    subroutine check_budget(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason

        rule%max_evaluations = 10
        state%evaluations = 9
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, "below the budget the run continues", &
            failures)
        state%evaluations = 10
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_BUDGET, "at the budget the run stops", &
            failures)

        rule%max_evaluations = huge(1)
        rule%max_cost = 5.0_dp
        state%evaluations = 1
        state%total_cost = 4.999_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, "below the cost limit continues", &
            failures)
        state%total_cost = 5.0_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_COST, "at the cost limit stops", failures)

        rule%max_cost = huge(1.0_dp)
        rule%max_wall_seconds = 60.0_dp
        state%total_cost = 0.0_dp
        state%wall_seconds = 60.5_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_WALL_TIME, "the wall clock stops the run", &
            failures)
    end subroutine check_budget

    !! The target must not fire before anything feasible has been seen. An
    !! uninitialized incumbent is not an achievement.
    subroutine check_target(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason

        rule%target_value = 1.0_dp

        state%has_feasible = .false.
        state%incumbent_value = -huge(1.0_dp)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, &
            "the target cannot fire before a feasible point exists", failures)

        state%has_feasible = .true.
        state%incumbent_value = 1.5_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, "above the target the run continues", &
            failures)
        state%incumbent_value = 1.0_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_TARGET_REACHED, "reaching the target stops", &
            failures)
    end subroutine check_target

    !! Patience counts iterations without a *meaningful* improvement. A run
    !! creeping down by amounts below the tolerance must still be judged stalled.
    subroutine check_patience(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason, k

        rule%patience = 5
        rule%improvement_tolerance = 1.0e-6_dp

        call state%record(10.0_dp, 1.0_dp, rule%improvement_tolerance)
        call expect(state%stalled_iterations == 0, "the first record is progress", &
            failures)

        ! Genuine improvements keep resetting the counter.
        do k = 1, 4
            call state%record(10.0_dp - real(k, dp), 1.0_dp, rule%improvement_tolerance)
        end do
        call expect(state%stalled_iterations == 0, &
            "real improvements reset the stall counter", failures)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, "an improving run continues", failures)

        ! Improvements below the tolerance count as stalls.
        do k = 1, 5
            call state%record(state%incumbent_value - 1.0e-9_dp, 1.0_dp, &
                rule%improvement_tolerance)
        end do
        call expect(state%stalled_iterations == 5, &
            "sub-tolerance improvements count as stalls", failures)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_NO_IMPROVEMENT, "a stalled run stops", &
            failures)

        ! The incumbent still tracks the best value even while stalled.
        call expect(state%incumbent_value < 6.0_dp, &
            "the incumbent keeps improving while stalled", failures)
    end subroutine check_patience

    subroutine check_uncertainty_and_acquisition(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason

        state%max_posterior_sd = 1.0e-3_dp
        state%best_acquisition = 1.0e-3_dp

        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, &
            "a negative tolerance leaves the rule disabled", failures)

        rule%uncertainty_tolerance = 1.0e-4_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_CONTINUE, &
            "uncertainty above the tolerance continues", failures)
        rule%uncertainty_tolerance = 1.0e-2_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_UNCERTAINTY, &
            "uncertainty below the tolerance stops", failures)

        rule%uncertainty_tolerance = -1.0_dp
        rule%acquisition_tolerance = 1.0e-2_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_ACQUISITION, &
            "a vanishing acquisition stops", failures)

        ! A zero tolerance is a legitimate, enabled setting.
        rule%acquisition_tolerance = 0.0_dp
        state%best_acquisition = 0.0_dp
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_ACQUISITION, &
            "a zero acquisition tolerance is enabled, not disabled", failures)
    end subroutine check_uncertainty_and_acquisition

    !! When several rules would fire, the resource limit wins. Stopping because
    !! the budget ran out and stopping because the model converged say opposite
    !! things about the answer's quality.
    subroutine check_priority_order(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason

        rule%max_evaluations = 5
        rule%target_value = 100.0_dp
        rule%patience = 1
        rule%uncertainty_tolerance = 10.0_dp
        rule%acquisition_tolerance = 10.0_dp

        state%evaluations = 5
        state%has_feasible = .true.
        state%incumbent_value = 0.0_dp
        state%stalled_iterations = 50
        state%max_posterior_sd = 0.0_dp
        state%best_acquisition = 0.0_dp

        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_BUDGET, &
            "the budget outranks every convergence rule", failures)

        rule%max_evaluations = huge(1)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_TARGET_REACHED, &
            "the target outranks the stall and tolerance rules", failures)

        rule%target_value = -huge(1.0_dp)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_NO_IMPROVEMENT, &
            "the stall rule outranks the tolerance rules", failures)

        rule%patience = huge(1)
        call rule%evaluate(state, reason, status)
        call expect(reason == FORTBO_STOP_UNCERTAINTY, &
            "uncertainty outranks the acquisition rule", failures)
    end subroutine check_priority_order

    subroutine check_reason_names(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_stop_reason_name(FORTBO_CONTINUE) == "continue", &
            "continue is named", failures)
        call expect(fortbo_stop_reason_name(FORTBO_STOP_BUDGET) == "budget_exhausted", &
            "the budget reason is named", failures)
        call expect(fortbo_stop_reason_name(FORTBO_STOP_NO_IMPROVEMENT) &
            == "no_improvement", "the stall reason is named", failures)
        call expect(fortbo_stop_reason_name(-7) == "unknown", &
            "an unrecognized reason is named unknown", failures)
    end subroutine check_reason_names

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_stopping_rule_t) :: rule
        type(fortbo_stopping_state_t) :: state
        type(fortnum_status_t) :: status
        integer :: reason

        rule%max_evaluations = -1
        call rule%evaluate(state, reason, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative budget is refused", failures)

        rule%max_evaluations = 10
        rule%improvement_tolerance = -1.0_dp
        call rule%evaluate(state, reason, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative improvement tolerance is refused", failures)
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

end program test_stopping
