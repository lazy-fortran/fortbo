module fortbo_placement
    !! Where an acquisition graph is allowed to run (ROADMAP BO5).
    !!
    !! The rule this module enforces is the roadmap's: **a FortAD-bearing
    !! acquisition graph stays on FortAD/FortSym until complete device
    !! JVP/VJP/HVP products exist for the surrogate underneath it.** The word
    !! that matters is *complete*. A partial set is worse than none, because it
    !! runs: a graph placed on a device that has a forward product but not a
    !! reverse one does not fail, it silently falls back for the missing piece,
    !! and the result is a run whose derivatives came from two different code
    !! paths with no record of which.
    !!
    !! **The precondition is not met today, and this module says so rather than
    !! assuming it.** FortML carries device JVP and VJP products for several
    !! models -- the multi-output GP, the variational classifiers, the ELBO --
    !! and device HVP products only for the linear pipelines. The exact GP
    !! regression that FortBO's own posteriors are built on has no device
    !! derivative products at all. So every derivative-bearing acquisition
    !! graph over a GP surrogate belongs on the host right now, and that is a
    !! statement about what exists rather than a preference.
    !!
    !! **Placement is refused by name, never downgraded silently.** A caller
    !! that asks for a device placement it cannot have gets a refusal carrying
    !! the reason. The alternative -- quietly returning a host placement -- is
    !! how a benchmark table ends up with a device row that never touched a
    !! device, which `fortbo_provenance` exists to prevent and which this
    !! module must not reintroduce one layer up.
    !!
    !! **On CUDA.** The roadmap allows dropping to CUDA for fixed
    !! sampling/reduction kernels where OpenACC cannot preserve residency or
    !! determinism. Measured, it can: `fortbo_device` pools every region's
    !! candidates into one array, runs one kernel, and takes one reduction with
    !! an index tie-break, and `test_device` checks the result is *bit
    !! identical* to the host across repeated launches. So the escape hatch is
    !! not taken, and the reason is recorded here rather than left as an
    !! absence -- writing CUDA for kernels OpenACC already handles
    !! deterministically would add a second code path to keep in agreement for
    !! no measured gain.

    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENT_GRADIENT, &
        FORTBO_CAP_MOMENT_HESSIAN, FORTBO_CAP_MEAN_HESSIAN
    implicit none
    private

    public :: fortbo_placement_t
    public :: fortbo_device_derivative_support_t
    public :: fortbo_decide_placement
    public :: fortbo_placement_name

    integer, parameter, public :: FORTBO_PLACE_HOST = 1
    integer, parameter, public :: FORTBO_PLACE_DEVICE = 2

    !! What the surrogate underneath actually offers on device.
    !!
    !! All three flags, separately, because "has device derivatives" is not a
    !! single fact. A graph that needs a Hessian-vector product is not served
    !! by a forward product however good it is.
    type :: fortbo_device_derivative_support_t
        logical :: jvp = .false.
        logical :: vjp = .false.
        logical :: hvp = .false.
    contains
        procedure, public :: complete => support_complete
        procedure, public :: missing => support_missing
    end type fortbo_device_derivative_support_t

    type :: fortbo_placement_t
        integer :: site = FORTBO_PLACE_HOST
        !! Why this site. Always populated, including for host placements,
        !! because "host because the graph carries no derivatives" and "host
        !! because the device products are incomplete" are different facts and
        !! a reader of a benchmark table needs to tell them apart.
        character(len=160) :: reason = ""
        !! True when the graph carries FortAD derivative work at all.
        logical :: derivative_bearing = .false.
    end type fortbo_placement_t

contains

    pure logical function support_complete(self) result(complete)
        class(fortbo_device_derivative_support_t), intent(in) :: self

        complete = self%jvp .and. self%vjp .and. self%hvp
    end function support_complete

    !! Which products are absent, named. A refusal that says "incomplete" and
    !! stops is not actionable; one that says which product is missing points
    !! at the work.
    pure function support_missing(self) result(text)
        class(fortbo_device_derivative_support_t), intent(in) :: self
        character(len=:), allocatable :: text

        text = ""
        if (.not. self%jvp) text = text//"jvp "
        if (.not. self%vjp) text = text//"vjp "
        if (.not. self%hvp) text = text//"hvp "
        if (len_trim(text) == 0) then
            text = "none"
        else
            text = trim(text)
        end if
    end function support_missing

    pure function fortbo_placement_name(site) result(name)
        integer, intent(in) :: site
        character(len=:), allocatable :: name

        select case (site)
        case (FORTBO_PLACE_HOST)
            name = "host"
        case (FORTBO_PLACE_DEVICE)
            name = "device"
        case default
            name = "unknown"
        end select
    end function fortbo_placement_name

    !! Decide where an acquisition graph runs.
    !!
    !! `wants_device` is the caller's request, not a decision. A request that
    !! cannot be honoured is refused with `FORTNUM_NOT_IMPLEMENTED` and a
    !! reason; the returned placement is still filled in with the host site so
    !! a caller that logs it regardless records something true.
    subroutine fortbo_decide_placement(posterior, support, wants_device, &
            device_available, placement, status)
        class(fortbo_posterior_t), intent(in) :: posterior
        type(fortbo_device_derivative_support_t), intent(in) :: support
        logical, intent(in) :: wants_device
        logical, intent(in) :: device_available
        type(fortbo_placement_t), intent(out) :: placement
        type(fortnum_status_t), intent(out) :: status
        logical :: bearing

        ! A graph is derivative-bearing exactly when the surrogate below it
        ! offers derivative moments for the acquisition to differentiate
        ! through. Asking the posterior rather than taking a flag from the
        ! caller means the answer cannot drift from what the model can do.
        bearing = posterior%supports(FORTBO_CAP_MOMENT_GRADIENT) &
            .or. posterior%supports(FORTBO_CAP_MOMENT_HESSIAN) &
            .or. posterior%supports(FORTBO_CAP_MEAN_HESSIAN)

        placement%site = FORTBO_PLACE_HOST
        placement%derivative_bearing = bearing

        if (.not. wants_device) then
            placement%reason = "host requested"
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        if (.not. device_available) then
            placement%reason = "no device present"
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo placement: a device placement was requested but no "// &
                "device is present")
            return
        end if

        if (bearing .and. .not. support%complete()) then
            ! The roadmap's rule, enforced. Naming the missing products makes
            ! the refusal actionable instead of merely correct.
            placement%reason = "derivative-bearing graph, device products "// &
                "incomplete: missing "//support%missing()
            ! Short enough to survive `fortnum_status_t`'s fixed-width
            ! message field. An earlier version spelled out the whole rule and
            ! was truncated before the word "missing", which removed the only
            ! actionable part of the refusal -- the rule itself is in this
            ! module's documentation, the missing products are not.
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo placement: device products incomplete, missing "// &
                support%missing())
            return
        end if

        placement%site = FORTBO_PLACE_DEVICE
        if (bearing) then
            placement%reason = "derivative-bearing graph, device products complete"
        else
            ! A value-only graph never needed the derivative products, so their
            ! absence is irrelevant to it. Saying so keeps a reader from
            ! concluding the products exist.
            placement%reason = "value-only graph, no derivative products required"
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_decide_placement

end module fortbo_placement
