program test_provenance
    !! BO6: lane separation and provenance on benchmark rows.
    !!
    !! Every check here is aimed at a specific way benchmark tables mislead,
    !! and the oracle is the recorded table: the test knows what it put in, so
    !! it knows what the summaries must say.
    !!
    !!   * a refusal must not be averaged as a zero, which would drag every
    !!     summary toward it invisibly;
    !!   * a lane with no measured rows must report *unavailable*, not zero —
    !!     "fast" and "never ran" are different claims;
    !!   * two rows from different source revisions must not be comparable, the
    !!     most common way a speedup is manufactured by accident;
    !!   * a refusal must carry a reason, or it is indistinguishable from a row
    !!     nobody filled in.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_provenance, only: fortbo_provenance_table_t, fortbo_lane_name, &
        FORTBO_LANE_CPU, FORTBO_LANE_GPU_TRANSFER, FORTBO_LANE_GPU_RESIDENT, &
        FORTBO_LANE_REFUSED
    implicit none

    integer :: failures

    failures = 0
    call check_lanes_stay_separate(failures)
    call check_a_refusal_is_not_a_zero(failures)
    call check_missing_lane_is_unavailable_not_fast(failures)
    call check_comparability_requires_matching_provenance(failures)
    call check_lane_names(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_provenance: PASS"
    else
        print *, "test_provenance: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_lanes_stay_separate(failures)
        integer, intent(inout) :: failures
        type(fortbo_provenance_table_t) :: table
        type(fortnum_status_t) :: status
        real(dp) :: mean
        logical :: available

        call table%initialize(status)
        call expect(status%code == FORTNUM_OK, "the table initializes", failures)

        call table%record(FORTBO_LANE_CPU, "branin", "abc1234", "gfortran-16.1", &
            "cpu", 1.0_dp, status)
        call table%record(FORTBO_LANE_GPU_TRANSFER, "branin", "abc1234", &
            "nvfortran-26.5", "RTX 5060 Ti", 0.6_dp, status)
        call table%record(FORTBO_LANE_GPU_RESIDENT, "branin", "abc1234", &
            "nvfortran-26.5", "RTX 5060 Ti", 0.1_dp, status)
        call expect(status%code == FORTNUM_OK, "the three lanes record", failures)

        call expect(table%lane_count(FORTBO_LANE_CPU) == 1 .and. &
            table%lane_count(FORTBO_LANE_GPU_TRANSFER) == 1 .and. &
            table%lane_count(FORTBO_LANE_GPU_RESIDENT) == 1, &
            "each lane holds its own row", failures)

        ! The distinction that matters: resident and transfer-inclusive are
        ! different numbers answering different questions, and averaging them
        ! together would report neither.
        call table%lane_mean_seconds(FORTBO_LANE_GPU_RESIDENT, mean, available)
        call expect(available .and. abs(mean - 0.1_dp) < 1.0e-12_dp, &
            "the resident lane reports only resident timings", failures)
        call table%lane_mean_seconds(FORTBO_LANE_GPU_TRANSFER, mean, available)
        call expect(available .and. abs(mean - 0.6_dp) < 1.0e-12_dp, &
            "the transfer-inclusive lane reports only its own", failures)
    end subroutine check_lanes_stay_separate

    !! The failure this design exists to prevent. A refused configuration
    !! recorded as a zero would make the lane look six times faster than it is.
    subroutine check_a_refusal_is_not_a_zero(failures)
        integer, intent(inout) :: failures
        type(fortbo_provenance_table_t) :: table
        type(fortnum_status_t) :: status
        real(dp) :: mean
        logical :: available

        call table%initialize(status)
        call table%record(FORTBO_LANE_GPU_RESIDENT, "ackley", "abc1234", &
            "nvfortran-26.5", "RTX 5060 Ti", 0.6_dp, status)
        call table%refuse("ackley-fp32", "abc1234", "nvfortran-26.5", &
            "RTX 5060 Ti", "single precision is not supported by the kernel", &
            status)
        call expect(status%code == FORTNUM_OK, "the refusal records", failures)

        call expect(table%lane_count(FORTBO_LANE_REFUSED) == 1, &
            "the refusal is kept in the table rather than dropped", failures)

        call table%lane_mean_seconds(FORTBO_LANE_GPU_RESIDENT, mean, available)
        call expect(available .and. abs(mean - 0.6_dp) < 1.0e-12_dp, &
            "the refusal does not enter the resident lane's mean", failures)

        ! And it carries no timing at all, so nothing downstream can average it.
        call expect(.not. table%rows(2)%has_timing, &
            "a refusal carries no timing, not a zero", failures)
        call expect(len_trim(table%rows(2)%refusal_reason) > 0, &
            "a refusal says why", failures)
    end subroutine check_a_refusal_is_not_a_zero

    !! "Fast" and "never ran" are different claims, and a zero conflates them.
    subroutine check_missing_lane_is_unavailable_not_fast(failures)
        integer, intent(inout) :: failures
        type(fortbo_provenance_table_t) :: table
        type(fortnum_status_t) :: status
        real(dp) :: mean
        logical :: available

        call table%initialize(status)
        call table%record(FORTBO_LANE_CPU, "levy", "abc1234", "gfortran-16.1", &
            "cpu", 2.0_dp, status)
        call table%refuse("levy", "abc1234", "gfortran-16.1", "cpu", &
            "no device available for a resident kernel", status)

        call table%lane_mean_seconds(FORTBO_LANE_GPU_RESIDENT, mean, available)
        call expect(.not. available, &
            "a lane that never ran reports unavailable, not zero", failures)
        call expect(mean == 0.0_dp, &
            "and its mean is left at zero only alongside that flag", failures)

        call table%lane_mean_seconds(FORTBO_LANE_CPU, mean, available)
        call expect(available .and. abs(mean - 2.0_dp) < 1.0e-12_dp, &
            "the lane that did run reports its mean", failures)
    end subroutine check_missing_lane_is_unavailable_not_fast

    !! Comparing across revisions measures the revisions, not the lanes.
    subroutine check_comparability_requires_matching_provenance(failures)
        integer, intent(inout) :: failures
        type(fortbo_provenance_table_t) :: table
        type(fortnum_status_t) :: status

        call table%initialize(status)
        call table%record(FORTBO_LANE_CPU, "branin", "abc1234", "gfortran-16.1", &
            "cpu", 1.0_dp, status)
        call table%record(FORTBO_LANE_GPU_RESIDENT, "branin", "abc1234", &
            "nvfortran-26.5", "RTX 5060 Ti", 0.2_dp, status)
        call table%record(FORTBO_LANE_GPU_RESIDENT, "branin", "def5678", &
            "nvfortran-26.5", "RTX 5060 Ti", 0.1_dp, status)
        call table%record(FORTBO_LANE_CPU, "branin", "abc1234", "gfortran-16.1", &
            "cpu", 1.0_dp, status, precision="float32")
        call table%refuse("branin", "abc1234", "gfortran-16.1", "cpu", &
            "unsupported", status)

        ! Different lanes, same everything else: comparable, and that is the
        ! comparison the table exists for.
        call expect(table%comparable(1, 2), &
            "two lanes at the same revision and precision are comparable", &
            failures)
        ! Different revision: not comparable, however tempting the numbers.
        call expect(.not. table%comparable(1, 3), &
            "rows from different revisions are not comparable", failures)
        ! Different precision: not comparable.
        call expect(.not. table%comparable(1, 4), &
            "rows at different precisions are not comparable", failures)
        ! A refusal has nothing to compare.
        call expect(.not. table%comparable(1, 5), &
            "a refusal is not comparable with a timing", failures)
        call expect(.not. table%comparable(1, 99), &
            "an out-of-range row is not comparable", failures)
    end subroutine check_comparability_requires_matching_provenance

    subroutine check_lane_names(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_lane_name(FORTBO_LANE_GPU_RESIDENT) == "gpu_resident", &
            "the resident lane is nameable", failures)
        call expect(fortbo_lane_name(FORTBO_LANE_GPU_TRANSFER) &
            == "gpu_transfer_inclusive", &
            "the transfer-inclusive lane says so in its name", failures)
        call expect(fortbo_lane_name(FORTBO_LANE_REFUSED) == "refused", &
            "the refusal lane is nameable", failures)
        call expect(fortbo_lane_name(0) == "unknown", &
            "an unrecognized lane is named unknown", failures)
    end subroutine check_lane_names

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_provenance_table_t) :: table, fresh
        type(fortnum_status_t) :: status

        call fresh%record(FORTBO_LANE_CPU, "x", "r", "t", "d", 1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "recording into an uninitialized table is refused", failures)

        call table%initialize(status)

        call table%record(FORTBO_LANE_CPU, "", "abc", "gfortran", "cpu", 1.0_dp, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a row without a case name is refused", failures)

        call table%record(FORTBO_LANE_CPU, "branin", "", "gfortran", "cpu", &
            1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a row without a source revision is refused", failures)

        call table%record(FORTBO_LANE_CPU, "branin", "abc", "gfortran", "cpu", &
            -1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative wall time is refused", failures)

        ! A refusal must go through `refuse`, which requires the reason that a
        ! measured row has nowhere to put.
        call table%record(FORTBO_LANE_REFUSED, "branin", "abc", "gfortran", &
            "cpu", 0.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a refusal cannot be recorded as a timing", failures)

        call table%refuse("branin", "abc", "gfortran", "cpu", "", status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a refusal without a reason is refused", failures)
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

end program test_provenance
