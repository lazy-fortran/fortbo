module fortbo_metrics
    !! Run metrics for benchmark records (ROADMAP BO6).
    !!
    !! Every number a benchmark reports is recorded here, per evaluation, in run
    !! order. The design decisions worth stating:
    !!
    !!   * **Simple regret is over feasible points only.** An infeasible point
    !!     with a wonderful objective value is not a solution, and letting it set
    !!     the incumbent makes a constrained run look like it solved a problem it
    !!     never solved. When nothing feasible has been seen the regret is
    !!     reported as unavailable rather than as some large number, because
    !!     "we have not found a feasible point" and "we found a bad one" call for
    !!     different responses.
    !!   * **Cumulative regret sums the regret of every evaluation**, including
    !!     the ones that did not improve. Simple regret says how good the answer
    !!     is; cumulative regret says what the search cost along the way, and a
    !!     method that finds the optimum immediately and then wanders is
    !!     indistinguishable from one that finds it and stops unless both are
    !!     recorded.
    !!   * **Acquisition and gradient evaluations are counted separately from
    !!     objective evaluations.** A method that spends a thousand acquisition
    !!     evaluations per objective evaluation is making a real trade, and a
    !!     regret-per-objective-evaluation plot hides it entirely.
    !!   * **Wall time, memory, and transfers are recorded but never used to
    !!     compare correctness.** They belong to the machine, not the method.
    !!
    !! Effective sample size uses the standard `1 / sum(w^2)` for normalized
    !! weights, which equals the sample count exactly when the weights are
    !! uniform and falls toward one as a single weight dominates. It is the
    !! diagnostic that says whether a sampled policy's average was carried by
    !! its whole sample or by three draws of it.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_metrics_t
    public :: fortbo_metric_row_t
    public :: fortbo_effective_sample_size

    integer, parameter :: INITIAL_CAPACITY = 64

    type :: fortbo_metric_row_t
        integer :: evaluation = 0
        real(dp) :: objective = 0.0_dp
        logical :: feasible = .true.
        !! Total constraint violation at this point; zero when feasible.
        real(dp) :: violation = 0.0_dp
        !! Best feasible objective seen up to and including this evaluation.
        real(dp) :: best_feasible = huge(1.0_dp)
        logical :: has_feasible = .false.
        !! `best_feasible - optimum`, and the running sum of per-evaluation
        !! regret. Both unavailable until something feasible exists.
        real(dp) :: simple_regret = huge(1.0_dp)
        real(dp) :: cumulative_regret = 0.0_dp
        logical :: regret_available = .false.
        integer :: acquisition_evaluations = 0
        integer :: gradient_evaluations = 0
        real(dp) :: effective_sample_size = 0.0_dp
        real(dp) :: wall_seconds = 0.0_dp
        real(dp) :: peak_megabytes = 0.0_dp
        integer :: host_device_transfers = 0
    end type fortbo_metric_row_t

    type :: fortbo_metrics_t
        !! The optimum used for regret. A run without a known optimum leaves
        !! this unset and gets best-value curves rather than regret ones, which
        !! is the honest outcome rather than regret against a guess.
        real(dp) :: optimum = 0.0_dp
        logical :: optimum_known = .false.
        integer :: count = 0
        type(fortbo_metric_row_t), allocatable :: rows(:)
        real(dp) :: best_feasible = huge(1.0_dp)
        logical :: has_feasible = .false.
        real(dp) :: cumulative_regret = 0.0_dp
    contains
        procedure, public :: initialize => metrics_initialize
        procedure, public :: record => metrics_record
        procedure, public :: simple_regret => metrics_simple_regret
        procedure, public :: total_violations => metrics_total_violations
        procedure, public :: feasible_count => metrics_feasible_count
    end type fortbo_metrics_t

contains

    !! `1 / sum(w^2)` for weights normalized to sum to one.
    subroutine fortbo_effective_sample_size(weights, ess, status)
        real(dp), intent(in) :: weights(:)
        real(dp), intent(out) :: ess
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: total, sum_squares
        integer :: i

        ess = 0.0_dp
        if (size(weights) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo metrics: at least one weight is required")
            return
        end if
        if (any(weights < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo metrics: weights must not be negative")
            return
        end if
        total = sum(weights)
        if (total <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo metrics: weights must not all be zero")
            return
        end if

        sum_squares = 0.0_dp
        do i = 1, size(weights)
            sum_squares = sum_squares + (weights(i)/total)**2
        end do
        ess = 1.0_dp/sum_squares
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_effective_sample_size

    subroutine metrics_initialize(self, status, optimum)
        class(fortbo_metrics_t), intent(out) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: optimum

        allocate (self%rows(INITIAL_CAPACITY))
        self%count = 0
        self%best_feasible = huge(1.0_dp)
        self%has_feasible = .false.
        self%cumulative_regret = 0.0_dp
        if (present(optimum)) then
            self%optimum = optimum
            self%optimum_known = .true.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine metrics_initialize

    !! Append one evaluation.
    !!
    !! The incumbent advances only on a *feasible* improvement. An infeasible
    !! point with a better objective is not a better answer, and letting it set
    !! the incumbent is how a constrained run comes to report a regret it never
    !! achieved.
    subroutine metrics_record(self, objective, status, feasible, violation, &
            acquisition_evaluations, gradient_evaluations, &
            effective_sample_size, wall_seconds, peak_megabytes, &
            host_device_transfers)
        class(fortbo_metrics_t), intent(inout) :: self
        real(dp), intent(in) :: objective
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: feasible
        real(dp), intent(in), optional :: violation
        integer, intent(in), optional :: acquisition_evaluations
        integer, intent(in), optional :: gradient_evaluations
        real(dp), intent(in), optional :: effective_sample_size
        real(dp), intent(in), optional :: wall_seconds
        real(dp), intent(in), optional :: peak_megabytes
        integer, intent(in), optional :: host_device_transfers
        type(fortbo_metric_row_t), allocatable :: grown(:)
        type(fortbo_metric_row_t) :: row
        logical :: is_feasible

        if (.not. allocated(self%rows)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo metrics: metrics are not initialized")
            return
        end if

        is_feasible = .true.
        if (present(feasible)) is_feasible = feasible
        if (present(violation)) then
            if (violation < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo metrics: violation must not be negative")
                return
            end if
        end if

        row%evaluation = self%count + 1
        row%objective = objective
        row%feasible = is_feasible
        if (present(violation)) row%violation = violation
        if (present(acquisition_evaluations)) &
            row%acquisition_evaluations = acquisition_evaluations
        if (present(gradient_evaluations)) &
            row%gradient_evaluations = gradient_evaluations
        if (present(effective_sample_size)) &
            row%effective_sample_size = effective_sample_size
        if (present(wall_seconds)) row%wall_seconds = wall_seconds
        if (present(peak_megabytes)) row%peak_megabytes = peak_megabytes
        if (present(host_device_transfers)) &
            row%host_device_transfers = host_device_transfers

        if (is_feasible) then
            if (.not. self%has_feasible .or. objective < self%best_feasible) then
                self%best_feasible = objective
                self%has_feasible = .true.
            end if
        end if
        row%best_feasible = self%best_feasible
        row%has_feasible = self%has_feasible

        if (self%optimum_known .and. self%has_feasible) then
            row%simple_regret = self%best_feasible - self%optimum
            ! Cumulative regret charges every evaluation, improving or not.
            ! Charging only improvements would make a method that finds the
            ! optimum and then wanders look identical to one that stops.
            self%cumulative_regret = self%cumulative_regret + row%simple_regret
            row%cumulative_regret = self%cumulative_regret
            row%regret_available = .true.
        end if

        if (self%count == size(self%rows)) then
            allocate (grown(2*size(self%rows)))
            grown(:self%count) = self%rows(:self%count)
            call move_alloc(grown, self%rows)
        end if
        self%count = self%count + 1
        self%rows(self%count) = row
        call status_set(status, FORTNUM_OK, "")
    end subroutine metrics_record

    !! Final simple regret. `available` is false when no feasible point was ever
    !! found, or when the run has no known optimum.
    subroutine metrics_simple_regret(self, regret, available)
        class(fortbo_metrics_t), intent(in) :: self
        real(dp), intent(out) :: regret
        logical, intent(out) :: available

        regret = huge(1.0_dp)
        available = self%optimum_known .and. self%has_feasible
        if (available) regret = self%best_feasible - self%optimum
    end subroutine metrics_simple_regret

    pure real(dp) function metrics_total_violations(self) result(total)
        class(fortbo_metrics_t), intent(in) :: self
        integer :: i

        total = 0.0_dp
        if (.not. allocated(self%rows)) return
        do i = 1, self%count
            total = total + self%rows(i)%violation
        end do
    end function metrics_total_violations

    pure integer function metrics_feasible_count(self) result(total)
        class(fortbo_metrics_t), intent(in) :: self
        integer :: i

        total = 0
        if (.not. allocated(self%rows)) return
        do i = 1, self%count
            if (self%rows(i)%feasible) total = total + 1
        end do
    end function metrics_feasible_count

end module fortbo_metrics
