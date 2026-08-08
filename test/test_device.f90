program test_device
    !! BO5: device-resident acquisition kernels.
    !!
    !! The claim under test is not speed. It is that the device path returns
    !! **bit-identical** answers to the host path, and refuses by name when it
    !! cannot run. A GPU kernel that is fast and disagrees with the host in the
    !! last bit makes a run unreplayable, which the roadmap forbids independently
    !! of any performance claim.
    !!
    !! Oracles:
    !!
    !!   * the host path against expected improvement computed independently
    !!     here, so the reference the device is checked against is itself
    !!     checked;
    !!   * the device path against the host path, exactly — not within a
    !!     tolerance. A tolerance would hide precisely the reduction-order
    !!     nondeterminism the design exists to prevent;
    !!   * ties broken by the lowest index on both paths, which is what makes
    !!     "exactly" achievable at all;
    !!   * a typed refusal when no device is present, rather than a silent host
    !!     fallback that would let a benchmark row claim a device number it did
    !!     not earn.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_acquisition, only: fortbo_expected_improvement
    use fortbo_device, only: fortbo_device_available, fortbo_device_name, &
        fortbo_device_score_and_select, fortbo_host_score_and_select, &
        FORTBO_EXECUTED_HOST, FORTBO_EXECUTED_DEVICE
    implicit none

    integer :: failures

    failures = 0
    call check_host_matches_the_acquisition(failures)
    call check_ties_go_to_the_lowest_index(failures)
    call check_device_agrees_exactly_or_refuses(failures)
    call check_refusals(failures)
    call report_device_state()

    if (failures == 0) then
        print *, "test_device: PASS"
    else
        print *, "test_device: FAIL", failures
        error stop 1
    end if

contains

    !! The host path is the device's reference, so it is checked against the
    !! package's own expected improvement rather than trusted.
    subroutine check_host_matches_the_acquisition(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 200
        real(dp) :: mean(n), sd(n), value, expected, best_seen
        integer :: chosen, k, expected_index

        do k = 1, n
            mean(k) = sin(0.37_dp*real(k, dp))
            sd(k) = 0.2_dp + 0.15_dp*(1.0_dp + cos(0.11_dp*real(k, dp)))
        end do

        call fortbo_host_score_and_select(mean, sd, 0.5_dp, 0.0_dp, chosen, value, &
            status)
        call expect(status%code == FORTNUM_OK, "the host path runs", failures)

        ! Independent sweep, using the package's acquisition directly.
        best_seen = -huge(1.0_dp)
        expected_index = 0
        do k = 1, n
            call fortbo_expected_improvement(mean(k), sd(k), 0.5_dp, 0.0_dp, &
                expected)
            if (expected > best_seen) then
                best_seen = expected
                expected_index = k
            end if
        end do

        call expect(chosen == expected_index, &
            "the host path selects the same candidate as the acquisition", &
            failures)
        call expect(abs(value - best_seen) < 1.0e-14_dp, &
            "the host path reports the acquisition's own value", failures)
    end subroutine check_host_matches_the_acquisition

    !! Ties are what make an order-independent answer possible. Without a stated
    !! rule, two candidates with identical scores could be chosen differently by
    !! a sequential sweep and a parallel reduction, and no tolerance would hide
    !! it because the *index* would differ.
    subroutine check_ties_go_to_the_lowest_index(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(5), sd(5), value, device_value
        integer :: chosen, device_chosen, executed

        ! Three candidates identical and best.
        mean = [0.0_dp, -1.0_dp, 0.5_dp, -1.0_dp, -1.0_dp]
        sd = [0.3_dp, 0.4_dp, 0.3_dp, 0.4_dp, 0.4_dp]

        call fortbo_host_score_and_select(mean, sd, 0.0_dp, 0.0_dp, chosen, value, &
            status)
        call expect(chosen == 2, "the host breaks a tie to the lowest index", &
            failures)

        if (fortbo_device_available()) then
            call fortbo_device_score_and_select(mean, sd, 0.0_dp, 0.0_dp, &
                device_chosen, device_value, executed, status)
            call expect(status%code == FORTNUM_OK .and. device_chosen == chosen, &
                "the device breaks the same tie the same way", failures)
        end if
    end subroutine check_ties_go_to_the_lowest_index

    !! Exact agreement, not agreement within a tolerance. A tolerance would
    !! accept precisely the reduction-order variation the design prevents.
    subroutine check_device_agrees_exactly_or_refuses(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 5000
        real(dp) :: mean(n), sd(n)
        real(dp) :: host_value, device_value, repeat_value
        integer :: host_chosen, device_chosen, repeat_chosen, executed, k

        do k = 1, n
            mean(k) = 0.5_dp*sin(0.013_dp*real(k, dp)) &
                + 0.2_dp*cos(0.0071_dp*real(k, dp))
            sd(k) = 0.05_dp + 0.4_dp*abs(sin(0.0037_dp*real(k, dp)))
        end do

        call fortbo_host_score_and_select(mean, sd, 0.3_dp, 0.0_dp, host_chosen, &
            host_value, status)
        call expect(status%code == FORTNUM_OK, "the host reference runs", failures)

        call fortbo_device_score_and_select(mean, sd, 0.3_dp, 0.0_dp, &
            device_chosen, device_value, executed, status)

        if (fortbo_device_available()) then
            call expect(status%code == FORTNUM_OK, "the device path runs", failures)
            call expect(executed == FORTBO_EXECUTED_DEVICE, &
                "the device path reports that it ran on a device", failures)
            call expect(device_chosen == host_chosen, &
                "the device selects exactly the candidate the host selects", &
                failures)
            call expect(device_value == host_value, &
                "the device value is bit-identical to the host's", failures)

            ! Twice, because a reduction whose order varied between launches
            ! would pass a single comparison and fail replay.
            call fortbo_device_score_and_select(mean, sd, 0.3_dp, 0.0_dp, &
                repeat_chosen, repeat_value, executed, status)
            call expect(repeat_chosen == device_chosen .and. &
                repeat_value == device_value, &
                "two device runs agree bit-for-bit with each other", failures)
        else
            ! The refusal is the contract when there is no device.
            call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
                "without a device the resident path refuses by name", failures)
            call expect(executed == FORTBO_EXECUTED_HOST, &
                "a refused device call does not claim to have used a device", &
                failures)
        end if
    end subroutine check_device_agrees_exactly_or_refuses

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(3), sd(3), value
        integer :: chosen, executed

        mean = 0.0_dp
        sd = [0.1_dp, -0.1_dp, 0.1_dp]

        call fortbo_host_score_and_select(mean, sd, 0.0_dp, 0.0_dp, chosen, value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative standard deviation is refused on the host", failures)

        call fortbo_device_score_and_select(mean, sd, 0.0_dp, 0.0_dp, chosen, &
            value, executed, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "and on the device path, before any offload is attempted", failures)

        call fortbo_host_score_and_select(mean, [0.1_dp, 0.1_dp], 0.0_dp, 0.0_dp, &
            chosen, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched moment arrays are refused", failures)

        call expect(fortbo_device_name(FORTBO_EXECUTED_DEVICE) == "device" .and. &
            fortbo_device_name(FORTBO_EXECUTED_HOST) == "host", &
            "the execution site is nameable for a benchmark row", failures)
        call expect(fortbo_device_name(-1) == "unknown", &
            "an unrecognized execution site is named unknown", failures)
    end subroutine check_refusals

    !! Say which path was exercised. A suite that silently skipped its device
    !! checks would look identical to one that ran them.
    subroutine report_device_state()
        if (fortbo_device_available()) then
            print *, "  device kernels exercised against the host reference"
        else
            print *, "  no device available; refusal path exercised instead"
        end if
    end subroutine report_device_state

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_device
