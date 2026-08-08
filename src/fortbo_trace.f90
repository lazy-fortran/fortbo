module fortbo_trace
    !! Trust-region traces for TuRBO and DTuRBO runs (ROADMAP BO6).
    !!
    !! A regret curve says a run found a good value. It does not say the
    !! trust-region logic worked. A region that never shrank, a restart that
    !! never fired, a bandit that sent every point to one region — all of these
    !! produce perfectly respectable regret curves on an easy problem and are
    !! all broken. The trace is what makes the difference visible.
    !!
    !! One row per recorded batch, carrying the region's radius, both counters,
    !! the ratio-test value where one applies, the adaptation event, and which
    !! region supplied the point. Rows are appended in run order and never
    !! rewritten, so a trace read back is the run as it happened rather than a
    !! summary of it.
    !!
    !! The ratio is stored as a sentinel when no ratio test was performed —
    !! TuRBO's counter rule has no ratio, and storing zero there would be
    !! indistinguishable from a genuinely useless step. `has_ratio` says which.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortbo_trust_region, only: fortbo_trust_region_t, FORTBO_TR_UNCHANGED, &
        FORTBO_TR_EXPANDED, FORTBO_TR_SHRANK, FORTBO_TR_EXHAUSTED
    implicit none
    private

    public :: fortbo_trace_t
    public :: fortbo_trace_row_t
    public :: fortbo_trace_event_name

    integer, parameter :: INITIAL_CAPACITY = 64

    type :: fortbo_trace_row_t
        integer :: batch = 0
        integer :: region = 0
        real(dp) :: radius = 0.0_dp
        integer :: success_counter = 0
        integer :: failure_counter = 0
        integer :: event = FORTBO_TR_UNCHANGED
        !! Positive-is-good decreases, as the ratio test consumes them.
        real(dp) :: actual_decrease = 0.0_dp
        real(dp) :: predicted_decrease = 0.0_dp
        real(dp) :: ratio = 0.0_dp
        logical :: has_ratio = .false.
        !! Best objective value seen anywhere in the run at the time of the row.
        real(dp) :: incumbent = huge(1.0_dp)
        integer :: evaluations = 0
        integer :: restarts = 0
    end type fortbo_trace_row_t

    type :: fortbo_trace_t
        integer :: count = 0
        type(fortbo_trace_row_t), allocatable :: rows(:)
    contains
        procedure, public :: initialize => trace_initialize
        procedure, public :: record => trace_record
        procedure, public :: radius_history => trace_radius_history
        procedure, public :: event_count => trace_event_count
        procedure, public :: region_share => trace_region_share
        procedure, public :: shrank_after_expanding => trace_shrank_after_expanding
    end type fortbo_trace_t

contains

    pure function fortbo_trace_event_name(event) result(name)
        integer, intent(in) :: event
        character(len=:), allocatable :: name

        select case (event)
        case (FORTBO_TR_UNCHANGED)
            name = "unchanged"
        case (FORTBO_TR_EXPANDED)
            name = "expanded"
        case (FORTBO_TR_SHRANK)
            name = "shrank"
        case (FORTBO_TR_EXHAUSTED)
            name = "exhausted"
        case default
            name = "unknown"
        end select
    end function fortbo_trace_event_name

    subroutine trace_initialize(self, status)
        class(fortbo_trace_t), intent(out) :: self
        type(fortnum_status_t), intent(out) :: status

        allocate (self%rows(INITIAL_CAPACITY))
        self%count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine trace_initialize

    !! Append one row.
    !!
    !! The ratio is computed here rather than taken from the caller so that a
    !! trace can never disagree with the decreases it also stores. When no
    !! predicted decrease is supplied there was no ratio test, and the row says
    !! so instead of recording a zero that would read as a failed step.
    subroutine trace_record(self, batch, region_index, region, event, status, &
            actual_decrease, predicted_decrease, incumbent, &
            evaluations, restarts)
        class(fortbo_trace_t), intent(inout) :: self
        integer, intent(in) :: batch
        integer, intent(in) :: region_index
        type(fortbo_trust_region_t), intent(in) :: region
        integer, intent(in) :: event
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: actual_decrease
        real(dp), intent(in), optional :: predicted_decrease
        real(dp), intent(in), optional :: incumbent
        integer, intent(in), optional :: evaluations
        integer, intent(in), optional :: restarts
        type(fortbo_trace_row_t), allocatable :: grown(:)
        type(fortbo_trace_row_t) :: row

        if (.not. allocated(self%rows)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trace: trace is not initialized")
            return
        end if
        if (region_index < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trace: region index must be positive")
            return
        end if

        row%batch = batch
        row%region = region_index
        row%radius = region%length
        row%success_counter = region%success_counter
        row%failure_counter = region%failure_counter
        row%event = event
        if (present(actual_decrease)) row%actual_decrease = actual_decrease
        if (present(incumbent)) row%incumbent = incumbent
        if (present(evaluations)) row%evaluations = evaluations
        if (present(restarts)) row%restarts = restarts
        if (present(predicted_decrease)) then
            row%predicted_decrease = predicted_decrease
            if (predicted_decrease > 0.0_dp) then
                row%ratio = row%actual_decrease/predicted_decrease
                row%has_ratio = .true.
            end if
        end if

        if (self%count == size(self%rows)) then
            allocate (grown(2*size(self%rows)))
            grown(:self%count) = self%rows(:self%count)
            call move_alloc(grown, self%rows)
        end if
        self%count = self%count + 1
        self%rows(self%count) = row
        call status_set(status, FORTNUM_OK, "")
    end subroutine trace_record

    !! The radius history of one region, in run order. This is the series that
    !! shows whether the region adapted at all.
    subroutine trace_radius_history(self, region_index, history, count, status)
        class(fortbo_trace_t), intent(in) :: self
        integer, intent(in) :: region_index
        real(dp), allocatable, intent(out) :: history(:)
        integer, intent(out) :: count
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        count = 0
        allocate (history(max(self%count, 1)))
        history = 0.0_dp
        if (.not. allocated(self%rows)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trace: trace is not initialized")
            return
        end if
        do i = 1, self%count
            if (self%rows(i)%region /= region_index) cycle
            count = count + 1
            history(count) = self%rows(i)%radius
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine trace_radius_history

    pure integer function trace_event_count(self, event) result(total)
        class(fortbo_trace_t), intent(in) :: self
        integer, intent(in) :: event
        integer :: i

        total = 0
        if (.not. allocated(self%rows)) return
        do i = 1, self%count
            if (self%rows(i)%event == event) total = total + 1
        end do
    end function trace_event_count

    !! Fraction of recorded rows supplied by one region. With several regions
    !! this is the bandit's allocation, which is otherwise invisible.
    pure real(dp) function trace_region_share(self, region_index) result(share)
        class(fortbo_trace_t), intent(in) :: self
        integer, intent(in) :: region_index
        integer :: i, matches

        share = 0.0_dp
        if (.not. allocated(self%rows) .or. self%count == 0) return
        matches = 0
        do i = 1, self%count
            if (self%rows(i)%region == region_index) matches = matches + 1
        end do
        share = real(matches, dp)/real(self%count, dp)
    end function trace_region_share

    !! Whether a region ever shrank after having expanded. A trust region that
    !! only ever grows is not adapting to anything, and this is the cheapest
    !! signal that the counters or the ratio test are actually engaged.
    pure logical function trace_shrank_after_expanding(self, region_index) &
            result(observed)
        class(fortbo_trace_t), intent(in) :: self
        integer, intent(in) :: region_index
        integer :: i
        logical :: expanded

        observed = .false.
        expanded = .false.
        if (.not. allocated(self%rows)) return
        do i = 1, self%count
            if (self%rows(i)%region /= region_index) cycle
            if (self%rows(i)%event == FORTBO_TR_EXPANDED) expanded = .true.
            if (expanded .and. (self%rows(i)%event == FORTBO_TR_SHRANK .or. &
                self%rows(i)%event == FORTBO_TR_EXHAUSTED)) then
                observed = .true.
                return
            end if
        end do
    end function trace_shrank_after_expanding

end module fortbo_trace
