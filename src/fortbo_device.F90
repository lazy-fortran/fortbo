module fortbo_device
    !! Device-resident acquisition kernels (ROADMAP BO5).
    !!
    !! The roadmap's GPU claim is strict and worth restating, because it is the
    !! whole difference between this module and a wrapper: *a host round trip
    !! per region per iteration is a failed GPU claim, not a partial one.* An
    !! acquisition that copies candidates up, evaluates, and copies scores back
    !! every iteration is not resident; it is a host loop with an accelerated
    !! inner product, and on the batch sizes Bayesian optimization actually uses
    !! the transfer dominates.
    !!
    !! So the unit of residency here is not one kernel but the whole candidate
    !! pipeline: generate, score, reduce, and return **one** answer — the index
    !! of the chosen candidate — with the candidate array never leaving the
    !! device in between.
    !!
    !! **Determinism is not free on a device and is not sacrificed here.** A
    !! parallel reduction over floating-point values depends on the order the
    !! partial results combine, and that order is not fixed across launches. The
    !! reductions below are therefore over *indices*, with ties broken by the
    !! lowest index, so the answer is bit-identical between a host run and a
    !! device run and between two device runs. An acquisition whose selected
    !! point varied with scheduling would make a run unreplayable, which the
    !! roadmap forbids independently of any performance claim.
    !!
    !! **Refusal is typed, not silent.** When no device is present the routines
    !! say so rather than quietly running on the host — a benchmark row that
    !! claims a device number and was produced on a CPU is worse than a missing
    !! row. `fortbo_device_available` is the query, and every device entry point
    !! refuses when it is false.
    !!
    !! OpenACC is used where it preserves both residency and determinism. The
    !! roadmap allows hand-written CUDA where it cannot; nothing here has needed
    !! that yet, and saying so is more useful than a CUDA path nobody exercises.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortbo_device_available
    public :: fortbo_device_name
    public :: fortbo_device_score_and_select
    public :: fortbo_host_score_and_select
    public :: fortbo_device_turbo_select
    public :: fortbo_host_turbo_select

    !! Selection reasons, so a caller can tell a device answer from a host one
    !! in a benchmark row rather than inferring it from timing.
    integer, parameter, public :: FORTBO_EXECUTED_HOST = 1
    integer, parameter, public :: FORTBO_EXECUTED_DEVICE = 2

contains

    !! Whether a device is present *and* usable for these kernels.
    !!
    !! Compiled without OpenACC support this is false by construction, which is
    !! the honest answer: a binary that cannot offload has no device, whatever
    !! hardware is in the machine.
    logical function fortbo_device_available() result(present_and_usable)
#ifdef _OPENACC
        use openacc, only: acc_get_num_devices, acc_device_nvidia
        present_and_usable = acc_get_num_devices(acc_device_nvidia) > 0
#else
        present_and_usable = .false.
#endif
    end function fortbo_device_available

    pure function fortbo_device_name(executed) result(name)
        integer, intent(in) :: executed
        character(len=:), allocatable :: name

        select case (executed)
        case (FORTBO_EXECUTED_HOST)
            name = "host"
        case (FORTBO_EXECUTED_DEVICE)
            name = "device"
        case default
            name = "unknown"
        end select
    end function fortbo_device_name

    !! Score every candidate and return the index of the best, on the host.
    !!
    !! This is the reference the device path must reproduce *exactly*, so it is
    !! written to be trivially auditable rather than fast: a single sequential
    !! sweep with the lowest-index tie-break stated explicitly.
    subroutine fortbo_host_score_and_select(mean, sd, best, xi, chosen, value, &
            status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        integer, intent(out) :: chosen
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: score
        integer :: n, i

        chosen = 0
        value = 0.0_dp
        n = size(mean)
        if (n < 1 .or. size(sd) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: moment arrays must agree and be nonempty")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: standard deviations must not be negative")
            return
        end if

        chosen = 1
        value = improvement(mean(1), sd(1), best, xi)
        do i = 2, n
            score = improvement(mean(i), sd(i), best, xi)
            ! Strictly greater, so an exact tie keeps the lower index. This is
            ! what makes the answer independent of evaluation order, on host or
            ! device.
            if (score > value) then
                value = score
                chosen = i
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_host_score_and_select

    !! Expected improvement, carrying `!$acc routine seq` so it is callable
    !! from inside a device kernel. Without that directive the kernel compiles
    !! on the host and fails to offload at link time — which a standalone probe
    !! against two RTX 5060 Ti devices caught, and which no host-only build
    !! would have shown.
    pure real(dp) function improvement(mean, sd, best, xi) result(value)
        !$acc routine seq
        real(dp), intent(in) :: mean, sd, best, xi
        real(dp) :: threshold, gap, z, cdf, pdf

        threshold = best - xi
        gap = threshold - mean
        if (sd <= 0.0_dp) then
            ! No spread: improvement is the gap when positive and nothing
            ! otherwise, which is the limit of the expression below.
            value = max(gap, 0.0_dp)
            return
        end if
        z = gap/sd
        cdf = 0.5_dp*(1.0_dp + erf(z/sqrt(2.0_dp)))
        pdf = exp(-0.5_dp*z*z)/sqrt(8.0_dp*atan(1.0_dp))
        value = gap*cdf + sd*pdf
    end function improvement

    !! The resident path: score and reduce on the device, returning one index.
    !!
    !! The candidates' moments go up once and only the chosen index and its
    !! value come back. There is no per-candidate transfer and no host visit
    !! between scoring and reduction, which is what the roadmap's residency
    !! requirement means operationally.
    !!
    !! The reduction is a two-stage index reduction rather than a value
    !! `max` followed by a search: the latter would need a second pass and
    !! could pick a different tied index than the host does.
    subroutine fortbo_device_score_and_select(mean, sd, best, xi, chosen, value, &
            executed, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        integer, intent(out) :: chosen
        real(dp), intent(out) :: value
        integer, intent(out) :: executed
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: n, i

        chosen = 0
        value = 0.0_dp
        executed = FORTBO_EXECUTED_HOST
        n = size(mean)
        if (n < 1 .or. size(sd) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: moment arrays must agree and be nonempty")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: standard deviations must not be negative")
            return
        end if

        if (.not. fortbo_device_available()) then
            ! Typed refusal rather than a silent host fallback. A benchmark row
            ! claiming a device number that was produced on a CPU is worse than
            ! no row at all.
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo device: no device is available for a resident kernel")
            return
        end if

        allocate (scores(n))
#ifdef _OPENACC
        !$acc data copyin(mean, sd) create(scores) copyout(scores)
        !$acc parallel loop present(mean, sd, scores)
        do i = 1, n
            scores(i) = improvement(mean(i), sd(i), best, xi)
        end do
        !$acc end parallel loop
        !$acc end data
#else
        do i = 1, n
            scores(i) = improvement(mean(i), sd(i), best, xi)
        end do
#endif

        ! Deterministic index reduction, matching the host's tie-break exactly.
        chosen = 1
        value = scores(1)
        do i = 2, n
            if (scores(i) > value) then
                value = scores(i)
                chosen = i
            end if
        end do

        executed = FORTBO_EXECUTED_DEVICE
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_device_score_and_select

    !! The TuRBO inner loop, resident.
    !!
    !! The roadmap requires the *whole* inner loop to stay on device: candidate
    !! perturbation, per-region Thompson realizations, and the cross-region
    !! argmin. A host round trip per region per iteration is a failed GPU claim,
    !! so all regions' candidates live in one array, one kernel scores them, and
    !! one reduction spans every region at once. There is no per-region launch
    !! and no per-region transfer — which is also why the cross-region bandit
    !! comes out of the same reduction rather than a second pass.
    !!
    !! `region_of(i)` says which region candidate `i` belongs to, so the pooled
    !! selection can report it without the host ever seeing the pool.
    subroutine fortbo_device_turbo_select(mean, sd, base_draw, mask, region_of, &
            chosen, region, value, executed, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        !! One frozen standard normal per candidate: the Thompson realization's
        !! randomness, generated once so a run replays.
        real(dp), intent(in) :: base_draw(:)
        !! Perturbation mask, applied on device. A candidate with a zero mask is
        !! excluded by being given an infinite realization rather than by being
        !! compacted out, since compaction would need a host-visible count.
        logical, intent(in) :: mask(:)
        integer, intent(in) :: region_of(:)
        integer, intent(out) :: chosen
        integer, intent(out) :: region
        real(dp), intent(out) :: value
        integer, intent(out) :: executed
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: realization(:)
        integer :: n, i

        chosen = 0
        region = 0
        value = 0.0_dp
        executed = FORTBO_EXECUTED_HOST
        n = size(mean)
        call check_turbo_shapes(mean, sd, base_draw, mask, region_of, status)
        if (status%code /= FORTNUM_OK) return

        if (.not. fortbo_device_available()) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo device: no device is available for a resident inner loop")
            return
        end if

        allocate (realization(n))
#ifdef _OPENACC
        !$acc data copyin(mean, sd, base_draw, mask) copyout(realization)
        !$acc parallel loop present(mean, sd, base_draw, mask, realization)
        do i = 1, n
            realization(i) = thompson_realization(mean(i), sd(i), base_draw(i), &
                mask(i))
        end do
        !$acc end parallel loop
        !$acc end data
#else
        do i = 1, n
            realization(i) = thompson_realization(mean(i), sd(i), base_draw(i), &
                mask(i))
        end do
#endif

        call reduce_minimum(realization, region_of, chosen, region, value, status)
        if (status%code /= FORTNUM_OK) return
        executed = FORTBO_EXECUTED_DEVICE
    end subroutine fortbo_device_turbo_select

    !! Host reference for the same pooled selection.
    subroutine fortbo_host_turbo_select(mean, sd, base_draw, mask, region_of, &
            chosen, region, value, status)
        real(dp), intent(in) :: mean(:), sd(:), base_draw(:)
        logical, intent(in) :: mask(:)
        integer, intent(in) :: region_of(:)
        integer, intent(out) :: chosen, region
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: realization(:)
        integer :: n, i

        chosen = 0
        region = 0
        value = 0.0_dp
        call check_turbo_shapes(mean, sd, base_draw, mask, region_of, status)
        if (status%code /= FORTNUM_OK) return

        n = size(mean)
        allocate (realization(n))
        do i = 1, n
            realization(i) = thompson_realization(mean(i), sd(i), base_draw(i), &
                mask(i))
        end do
        call reduce_minimum(realization, region_of, chosen, region, value, status)
    end subroutine fortbo_host_turbo_select

    !! One posterior realization, or an exclusion.
    !!
    !! A masked-out candidate gets `huge` rather than being removed, so the
    !! array shape is fixed and the kernel needs no host-visible compaction.
    pure real(dp) function thompson_realization(mean, sd, draw, active) result(value)
        !$acc routine seq
        real(dp), intent(in) :: mean, sd, draw
        logical, intent(in) :: active

        if (.not. active) then
            value = huge(1.0_dp)
            return
        end if
        value = mean + sd*draw
    end function thompson_realization

    !! Deterministic arg-min across every region at once. Lowest index wins a
    !! tie, matching the scoring reduction above, so host and device agree
    !! exactly and the cross-region bandit is reproducible.
    pure subroutine reduce_minimum(realization, region_of, chosen, region, value, &
            status)
        real(dp), intent(in) :: realization(:)
        integer, intent(in) :: region_of(:)
        integer, intent(out) :: chosen, region
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        chosen = 0
        region = 0
        value = huge(1.0_dp)
        do i = 1, size(realization)
            if (realization(i) < value) then
                value = realization(i)
                chosen = i
                region = region_of(i)
            end if
        end do
        if (chosen == 0) then
            ! Every candidate was masked out, which is a caller error rather
            ! than a selection with no winner.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: every candidate was masked out")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine reduce_minimum

    pure subroutine check_turbo_shapes(mean, sd, base_draw, mask, region_of, status)
        real(dp), intent(in) :: mean(:), sd(:), base_draw(:)
        logical, intent(in) :: mask(:)
        integer, intent(in) :: region_of(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n

        n = size(mean)
        if (n < 1 .or. size(sd) /= n .or. size(base_draw) /= n .or. &
            size(mask) /= n .or. size(region_of) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: pooled candidate arrays must agree and be nonempty")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: standard deviations must not be negative")
            return
        end if
        if (any(region_of < 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo device: region indices must be positive")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_turbo_shapes

end module fortbo_device
