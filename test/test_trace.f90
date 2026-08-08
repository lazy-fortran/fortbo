program test_trace
    !! BO6: trust-region traces.
    !!
    !! The trace exists so that trust-region behavior can be checked rather than
    !! assumed, so the test does exactly that: it drives a real region through a
    !! real adaptation sequence and then asserts, from the trace alone, the
    !! things a regret curve cannot show.
    !!
    !! Oracles are the recorded run itself — the test knows what it fed the
    !! region, so it knows what the trace must contain. Nothing here compares
    !! the trace against the trace.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_trust_region, only: fortbo_trust_region_t, FORTBO_TR_UNCHANGED, &
        FORTBO_TR_EXPANDED, FORTBO_TR_SHRANK, FORTBO_TR_EXHAUSTED
    use fortbo_dturbo, only: fortbo_dturbo_ratio_update
    use fortbo_trace, only: fortbo_trace_t, fortbo_trace_event_name
    implicit none

    integer :: failures

    failures = 0
    call check_records_a_real_adaptation(failures)
    call check_ratio_is_consistent_with_its_decreases(failures)
    call check_absent_ratio_is_distinguishable_from_zero(failures)
    call check_region_shares(failures)
    call check_growth_beyond_capacity(failures)
    call check_event_names(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_trace: PASS"
    else
        print *, "test_trace: FAIL", failures
        error stop 1
    end if

contains

    !! Drive one region up and then down, and recover the whole story from the
    !! trace. This is the check the roadmap asks for: the radius history is what
    !! says the trust-region logic ran, and a regret curve cannot show it.
    subroutine check_records_a_real_adaptation(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace
        type(fortnum_status_t) :: status
        real(dp), allocatable :: history(:)
        real(dp) :: peak
        integer :: event, batch, count
        logical :: rose, fell

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call trace%initialize(status)
        call expect(status%code == FORTNUM_OK, "the trace initializes", failures)

        ! Record the starting state before any adaptation. Without this row the
        ! trace begins at the already-expanded radius and the history cannot
        ! show the region growing at all — the first thing a reader would look
        ! for.
        call trace%record(0, 1, region, FORTBO_TR_UNCHANGED, status, &
            incumbent=huge(1.0_dp), evaluations=0)

        ! Three good batches, then poor ones until the region gives up.
        do batch = 1, 3
            call fortbo_dturbo_ratio_update(region, 1.0_dp, 1.0_dp, event, status)
            call trace%record(batch, 1, region, event, status, &
                actual_decrease=1.0_dp, predicted_decrease=1.0_dp, &
                incumbent=-real(batch, dp), evaluations=batch)
        end do
        do batch = 4, 40
            call fortbo_dturbo_ratio_update(region, 0.0_dp, 1.0_dp, event, status)
            call trace%record(batch, 1, region, event, status, &
                actual_decrease=0.0_dp, predicted_decrease=1.0_dp, &
                incumbent=-3.0_dp, evaluations=batch)
            if (.not. region%active) exit
        end do

        call trace%radius_history(1, history, count, status)
        call expect(status%code == FORTNUM_OK .and. count > 4, &
            "the radius history is recorded", failures)

        rose = .false.
        fell = .false.
        peak = history(1)
        do batch = 2, count
            if (history(batch) > history(batch - 1)) rose = .true.
            if (history(batch) < history(batch - 1)) fell = .true.
            peak = max(peak, history(batch))
        end do
        call expect(rose, "the trace shows the region expanding", failures)
        call expect(fell, "the trace shows the region shrinking", failures)
        call expect(peak > history(count), &
            "the trace shows the radius coming back down from its peak", failures)

        call expect(trace%event_count(FORTBO_TR_EXPANDED) == 3, &
            "every expansion is recorded, and only those", failures)
        call expect(trace%event_count(FORTBO_TR_EXHAUSTED) == 1, &
            "the collapse is recorded exactly once", failures)
        call expect(trace%shrank_after_expanding(1), &
            "the trace shows a region that adapted in both directions", failures)

        ! A region that only ever grew would pass a regret check and fail this.
        call expect(.not. trace%shrank_after_expanding(2), &
            "a region with no rows shows no adaptation", failures)
    end subroutine check_records_a_real_adaptation

    !! The stored ratio must equal the stored decreases. Recording all three
    !! independently is what would let them drift apart.
    subroutine check_ratio_is_consistent_with_its_decreases(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace
        type(fortnum_status_t) :: status
        integer :: i
        logical :: consistent

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call trace%initialize(status)

        call trace%record(1, 1, region, FORTBO_TR_UNCHANGED, status, &
            actual_decrease=0.3_dp, predicted_decrease=1.2_dp)
        call trace%record(2, 1, region, FORTBO_TR_EXPANDED, status, &
            actual_decrease=2.5_dp, predicted_decrease=2.0_dp)

        consistent = .true.
        do i = 1, trace%count
            if (.not. trace%rows(i)%has_ratio) cycle
            if (abs(trace%rows(i)%ratio - trace%rows(i)%actual_decrease &
                /trace%rows(i)%predicted_decrease) > 1.0e-14_dp) consistent = .false.
        end do
        call expect(consistent, "the recorded ratio matches its own decreases", &
            failures)
        call expect(trace%rows(2)%ratio > 1.0_dp, &
            "a step that beat its prediction records a ratio above one", failures)
    end subroutine check_ratio_is_consistent_with_its_decreases

    !! A counter-rule run has no ratio at all. Storing zero would be
    !! indistinguishable from a step that predicted well and delivered nothing,
    !! which is the opposite verdict about the model.
    subroutine check_absent_ratio_is_distinguishable_from_zero(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace
        type(fortnum_status_t) :: status

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call trace%initialize(status)

        ! TuRBO: no ratio test was run.
        call trace%record(1, 1, region, FORTBO_TR_SHRANK, status)
        ! DTuRBO: a ratio test ran and the step was worthless.
        call trace%record(2, 1, region, FORTBO_TR_SHRANK, status, &
            actual_decrease=0.0_dp, predicted_decrease=1.0_dp)

        call expect(.not. trace%rows(1)%has_ratio, &
            "a counter-rule row carries no ratio", failures)
        call expect(trace%rows(2)%has_ratio .and. trace%rows(2)%ratio == 0.0_dp, &
            "a genuinely zero ratio is recorded as present", failures)
        call expect(trace%rows(1)%ratio == trace%rows(2)%ratio, &
            "the two are indistinguishable by value alone, hence the flag", &
            failures)

        ! A zero predicted decrease is also not a ratio: dividing by it would
        ! manufacture a verdict out of nothing.
        call trace%record(3, 1, region, FORTBO_TR_UNCHANGED, status, &
            actual_decrease=1.0_dp, predicted_decrease=0.0_dp)
        call expect(.not. trace%rows(3)%has_ratio, &
            "a vanishing prediction yields no ratio", failures)
    end subroutine check_absent_ratio_is_distinguishable_from_zero

    subroutine check_region_shares(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace
        type(fortnum_status_t) :: status
        integer :: k

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call trace%initialize(status)

        do k = 1, 7
            call trace%record(k, 1, region, FORTBO_TR_UNCHANGED, status)
        end do
        do k = 8, 10
            call trace%record(k, 2, region, FORTBO_TR_UNCHANGED, status)
        end do

        call expect(abs(trace%region_share(1) - 0.7_dp) < 1.0e-12_dp, &
            "the share of the busy region is recorded", failures)
        call expect(abs(trace%region_share(2) - 0.3_dp) < 1.0e-12_dp, &
            "the share of the quiet region is recorded", failures)
        call expect(trace%region_share(3) == 0.0_dp, &
            "a region that never proposed has no share", failures)
        call expect(abs(trace%region_share(1) + trace%region_share(2) - 1.0_dp) &
            < 1.0e-12_dp, "the shares account for every row", failures)
    end subroutine check_region_shares

    !! Rows must survive the buffer growing, in order and unaltered.
    subroutine check_growth_beyond_capacity(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace
        type(fortnum_status_t) :: status
        integer, parameter :: n = 500
        integer :: k
        logical :: ordered, preserved

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call trace%initialize(status)

        do k = 1, n
            call trace%record(k, 1 + mod(k, 3), region, FORTBO_TR_UNCHANGED, &
                status, incumbent=-real(k, dp), evaluations=k)
        end do
        call expect(trace%count == n, "every row is kept", failures)

        ordered = .true.
        preserved = .true.
        do k = 1, n
            if (trace%rows(k)%batch /= k) ordered = .false.
            if (trace%rows(k)%evaluations /= k) preserved = .false.
            if (trace%rows(k)%region /= 1 + mod(k, 3)) preserved = .false.
        end do
        call expect(ordered, "rows stay in run order across a resize", failures)
        call expect(preserved, "row contents survive a resize", failures)
    end subroutine check_growth_beyond_capacity

    subroutine check_event_names(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_trace_event_name(FORTBO_TR_EXPANDED) == "expanded", &
            "expansion is named", failures)
        call expect(fortbo_trace_event_name(FORTBO_TR_EXHAUSTED) == "exhausted", &
            "exhaustion is named", failures)
        call expect(fortbo_trace_event_name(-9) == "unknown", &
            "an unrecognized event is named unknown", failures)
    end subroutine check_event_names

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortbo_trace_t) :: trace, fresh
        type(fortnum_status_t) :: status
        real(dp), allocatable :: history(:)
        integer :: count

        call region%initialize(2, 1, status)
        call trace%initialize(status)

        call trace%record(1, 0, region, FORTBO_TR_UNCHANGED, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-positive region index is refused", failures)

        call fresh%record(1, 1, region, FORTBO_TR_UNCHANGED, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "recording into an uninitialized trace is refused", failures)

        call fresh%radius_history(1, history, count, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "reading an uninitialized trace is refused", failures)
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

end program test_trace
