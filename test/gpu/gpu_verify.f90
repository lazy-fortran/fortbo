program gpu_verify
    !! Real-hardware verification of the OpenACC device kernels.
    !!
    !! The in-tree `test_device` runs under gfortran, where `_OPENACC` is not
    !! defined, so the device branch compiles out and only the refusal path is
    !! exercised. That test is honest about it -- it prints which branch ran --
    !! but it cannot prove the kernels work, because it never builds them.
    !!
    !! This builds them with nvfortran and OpenACC against the real GPU. It
    !! cannot live in the main suite: a full nvfortran build of the stack is
    !! blocked by internal compiler errors in FortAD ("Deferred-length
    !! character symbol must have descriptor", and a segfault in fort1), which
    !! are nvfortran defects rather than ours. `fortbo_device` depends only on
    !! FortNum's kinds and status, so it compiles standalone.
    !!
    !! What is checked is what the design claims: the device answer is *bit
    !! identical* to the host's, repeatedly. A tolerance would accept exactly
    !! the reduction-order nondeterminism the pooled single-reduction design
    !! exists to prevent.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_device, only: fortbo_device_available, fortbo_device_name, &
        fortbo_device_score_and_select, fortbo_host_score_and_select, &
        fortbo_device_turbo_select, fortbo_host_turbo_select, &
        FORTBO_EXECUTED_DEVICE
    implicit none

    integer, parameter :: n = 20000, n_regions = 5
    real(dp), allocatable :: mean(:), sd(:), draw(:)
    logical, allocatable :: mask(:)
    integer, allocatable :: region_of(:)
    real(dp) :: host_value, device_value, repeat_value
    real(dp) :: host_turbo, device_turbo
    integer :: host_chosen, device_chosen, repeat_chosen, executed, k
    integer :: host_region, device_region
    type(fortnum_status_t) :: status
    integer :: failures

    failures = 0
    print *, "device available: ", fortbo_device_available()
    if (.not. fortbo_device_available()) then
        print *, "FAIL: built with OpenACC but no device visible"
        error stop 1
    end if

    allocate (mean(n), sd(n), draw(n), mask(n), region_of(n))
    do k = 1, n
        mean(k) = 0.5_dp*sin(0.013_dp*real(k, dp)) + 0.2_dp*cos(0.0071_dp*real(k, dp))
        sd(k) = 0.05_dp + 0.4_dp*abs(sin(0.0037_dp*real(k, dp)))
        draw(k) = 1.3_dp*sin(0.31_dp*real(k, dp))
        region_of(k) = 1 + mod(k - 1, n_regions)
        mask(k) = mod(k, 3) /= 0
    end do

    call fortbo_host_score_and_select(mean, sd, 0.3_dp, 0.0_dp, host_chosen, &
        host_value, status)
    call fortbo_device_score_and_select(mean, sd, 0.3_dp, 0.0_dp, &
        device_chosen, device_value, executed, status)
    call expect(status%code == FORTNUM_OK, "device selection runs")
    call expect(executed == FORTBO_EXECUTED_DEVICE, "it really ran on device")
    call expect(device_chosen == host_chosen, "device picks the host's index")
    call expect(device_value == host_value, "device value is bit identical")

    call fortbo_device_score_and_select(mean, sd, 0.3_dp, 0.0_dp, &
        repeat_chosen, repeat_value, executed, status)
    call expect(repeat_chosen == device_chosen .and. repeat_value == device_value, &
        "two device launches agree bit for bit")

    call fortbo_host_turbo_select(mean, sd, draw, mask, region_of, &
        host_chosen, host_region, host_turbo, status)
    call fortbo_device_turbo_select(mean, sd, draw, mask, region_of, &
        device_chosen, device_region, device_turbo, executed, status)
    call expect(status%code == FORTNUM_OK .and. executed == FORTBO_EXECUTED_DEVICE, &
        "resident pooled kernel runs on device")
    call expect(device_chosen == host_chosen .and. device_region == host_region &
        .and. device_turbo == host_turbo, &
        "pooled device reduction is bit identical to the host")

    if (failures == 0) then
        print *, "gpu_verify: PASS on ", fortbo_device_name(FORTBO_EXECUTED_DEVICE)
    else
        print *, "gpu_verify: FAIL", failures
        error stop 1
    end if

contains

    subroutine expect(condition, description)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        else
            print *, "  ok: ", description
        end if
    end subroutine expect
end program gpu_verify
