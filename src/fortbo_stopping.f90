module fortbo_stopping
    !! Stopping rules and diagnostics (ROADMAP BO4).
    !!
    !! A stopping rule decides when further evaluations are not worth their
    !! cost. Every rule here answers with a *reason*, not merely a boolean,
    !! because "the run stopped" is not a result anyone can act on: stopping
    !! because the budget ran out and stopping because the posterior collapsed
    !! mean opposite things about whether to trust the answer.
    !!
    !! The rules are deliberately separate and combinable. A run typically wants
    !! several at once — stop on convergence, but never exceed the budget — and
    !! the first rule to fire is reported, in a fixed priority order so that a
    !! replayed run stops for the same recorded reason.
    !!
    !! Every threshold is an absolute quantity in the objective's own units
    !! except where noted. Relative thresholds are offered separately rather
    !! than inferred, because dividing by an incumbent that happens to be near
    !! zero produces a criterion that fires at random.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    ! Reasons, in the order they are tested.
    integer, parameter, public :: FORTBO_CONTINUE = 0
    integer, parameter, public :: FORTBO_STOP_BUDGET = 1
    integer, parameter, public :: FORTBO_STOP_COST = 2
    integer, parameter, public :: FORTBO_STOP_WALL_TIME = 3
    integer, parameter, public :: FORTBO_STOP_TARGET_REACHED = 4
    integer, parameter, public :: FORTBO_STOP_NO_IMPROVEMENT = 5
    integer, parameter, public :: FORTBO_STOP_UNCERTAINTY = 6
    integer, parameter, public :: FORTBO_STOP_ACQUISITION = 7

    public :: fortbo_stopping_rule_t
    public :: fortbo_stopping_state_t
    public :: fortbo_stop_reason_name

    !! Thresholds. A rule is inactive when its threshold is left at the
    !! disabling default, so a caller enables exactly what it means to use.
    type :: fortbo_stopping_rule_t
        integer :: max_evaluations = huge(1)
        real(dp) :: max_cost = huge(1.0_dp)
        real(dp) :: max_wall_seconds = huge(1.0_dp)
        !! Stop once the incumbent is at or below this objective value.
        real(dp) :: target_value = -huge(1.0_dp)
        !! Stop after this many consecutive iterations without an improvement
        !! larger than `improvement_tolerance`.
        integer :: patience = huge(1)
        real(dp) :: improvement_tolerance = 0.0_dp
        !! Stop once the largest posterior standard deviation over the candidate
        !! set falls below this.
        real(dp) :: uncertainty_tolerance = -1.0_dp
        !! Stop once the best acquisition value falls below this.
        real(dp) :: acquisition_tolerance = -1.0_dp
    contains
        procedure, public :: evaluate => rule_evaluate
    end type fortbo_stopping_rule_t

    !! Everything a rule needs to see, gathered by the driver.
    type :: fortbo_stopping_state_t
        integer :: evaluations = 0
        real(dp) :: total_cost = 0.0_dp
        real(dp) :: wall_seconds = 0.0_dp
        real(dp) :: incumbent_value = huge(1.0_dp)
        !! Consecutive iterations without a meaningful improvement.
        integer :: stalled_iterations = 0
        real(dp) :: max_posterior_sd = huge(1.0_dp)
        real(dp) :: best_acquisition = huge(1.0_dp)
        logical :: has_feasible = .false.
    contains
        procedure, public :: record => state_record
    end type fortbo_stopping_state_t

contains

    pure function fortbo_stop_reason_name(reason) result(name)
        integer, intent(in) :: reason
        character(len=:), allocatable :: name

        select case (reason)
        case (FORTBO_CONTINUE)
            name = "continue"
        case (FORTBO_STOP_BUDGET)
            name = "budget_exhausted"
        case (FORTBO_STOP_COST)
            name = "cost_exhausted"
        case (FORTBO_STOP_WALL_TIME)
            name = "wall_time_exhausted"
        case (FORTBO_STOP_TARGET_REACHED)
            name = "target_reached"
        case (FORTBO_STOP_NO_IMPROVEMENT)
            name = "no_improvement"
        case (FORTBO_STOP_UNCERTAINTY)
            name = "uncertainty_below_tolerance"
        case (FORTBO_STOP_ACQUISITION)
            name = "acquisition_below_tolerance"
        case default
            name = "unknown"
        end select
    end function fortbo_stop_reason_name

    !! Fold one completed iteration into the state, maintaining the stall
    !! counter. An improvement must exceed `tolerance` to reset the counter;
    !! otherwise numerical noise in the incumbent keeps a stalled run alive
    !! forever.
    subroutine state_record(self, objective, cost, tolerance)
        class(fortbo_stopping_state_t), intent(inout) :: self
        real(dp), intent(in) :: objective
        real(dp), intent(in) :: cost
        real(dp), intent(in) :: tolerance

        self%evaluations = self%evaluations + 1
        self%total_cost = self%total_cost + cost
        if (objective < self%incumbent_value - tolerance) then
            self%incumbent_value = objective
            self%stalled_iterations = 0
        else
            if (objective < self%incumbent_value) self%incumbent_value = objective
            self%stalled_iterations = self%stalled_iterations + 1
        end if
        self%has_feasible = .true.
    end subroutine state_record

    !! Decide. Resource limits are tested first: a run that has spent its budget
    !! must stop for that reason even if a convergence rule would also have
    !! fired, because the two carry different confidence in the answer.
    subroutine rule_evaluate(self, state, reason, status)
        class(fortbo_stopping_rule_t), intent(in) :: self
        type(fortbo_stopping_state_t), intent(in) :: state
        integer, intent(out) :: reason
        type(fortnum_status_t), intent(out) :: status

        reason = FORTBO_CONTINUE
        if (self%max_evaluations < 0 .or. self%patience < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo stopping: negative limit")
            return
        end if
        if (self%improvement_tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo stopping: improvement tolerance must not be negative")
            return
        end if

        if (state%evaluations >= self%max_evaluations) then
            reason = FORTBO_STOP_BUDGET
        else if (state%total_cost >= self%max_cost) then
            reason = FORTBO_STOP_COST
        else if (state%wall_seconds >= self%max_wall_seconds) then
            reason = FORTBO_STOP_WALL_TIME
        else if (state%has_feasible .and. state%incumbent_value <= self%target_value) then
            reason = FORTBO_STOP_TARGET_REACHED
        else if (state%stalled_iterations >= self%patience) then
            reason = FORTBO_STOP_NO_IMPROVEMENT
        else if (self%uncertainty_tolerance >= 0.0_dp .and. &
                state%max_posterior_sd <= self%uncertainty_tolerance) then
            reason = FORTBO_STOP_UNCERTAINTY
        else if (self%acquisition_tolerance >= 0.0_dp .and. &
                state%best_acquisition <= self%acquisition_tolerance) then
            reason = FORTBO_STOP_ACQUISITION
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine rule_evaluate

end module fortbo_stopping
