program test_placement
    !! BO5: FortAD-bearing acquisition graphs stay on the host until complete
    !! device derivative products exist.
    !!
    !! The failure this prevents is not a crash. A derivative-bearing graph
    !! placed on a device with a forward product but no reverse one *runs* --
    !! it falls back for the missing piece and produces a run whose derivatives
    !! came from two code paths, with nothing recording which. So the checks
    !! here are about the decision being made on what exists rather than on
    !! what was asked for:
    !!
    !!   * a partial product set is refused exactly as firmly as an empty one,
    !!     since it is the partial case that would otherwise silently work;
    !!   * the refusal names the missing products, because "incomplete" does
    !!     not tell anyone what to build;
    !!   * a value-only graph is *not* blocked, since it never needed the
    !!     derivative products -- blocking it would be a different rule and a
    !!     needlessly conservative one;
    !!   * a refused request never comes back as a quiet host placement, which
    !!     is how a benchmark table acquires a device row that never ran on a
    !!     device.

    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use placement_stubs, only: value_only_t, differentiable_t
    use fortbo_placement, only: fortbo_placement_t, &
        fortbo_device_derivative_support_t, fortbo_decide_placement, &
        fortbo_placement_name, FORTBO_PLACE_HOST, FORTBO_PLACE_DEVICE
    implicit none

    integer :: failures

    failures = 0
    call check_partial_products_are_refused(failures)
    call check_complete_products_are_allowed(failures)
    call check_value_only_graphs_are_not_blocked(failures)
    call check_absent_device_is_refused(failures)
    call check_host_requests_are_honoured(failures)
    call check_names(failures)

    if (failures == 0) then
        print *, "test_placement: PASS"
    else
        print *, "test_placement: FAIL", failures
        error stop 1
    end if

contains

    !! Every incomplete combination, not just the empty one. The partial cases
    !! are the dangerous ones: they are the combinations under which a device
    !! placement would run and quietly mix code paths.
    subroutine check_partial_products_are_refused(failures)
        integer, intent(inout) :: failures
        type(differentiable_t) :: posterior
        type(fortbo_device_derivative_support_t) :: support
        type(fortbo_placement_t) :: placement
        type(fortnum_status_t) :: status
        logical :: jvp, vjp, hvp, all_refused, named
        integer :: combination

        all_refused = .true.
        named = .true.
        do combination = 0, 6
            jvp = btest(combination, 0)
            vjp = btest(combination, 1)
            hvp = btest(combination, 2)
            ! Seven of the eight combinations are incomplete; the eighth is
            ! checked separately.
            support%jvp = jvp
            support%vjp = vjp
            support%hvp = hvp

            call fortbo_decide_placement(posterior, support, .true., .true., &
                placement, status)
            if (status%code /= FORTNUM_NOT_IMPLEMENTED) all_refused = .false.
            ! And it must not come back as a device placement anyway.
            if (placement%site /= FORTBO_PLACE_HOST) all_refused = .false.
            if (index(status%msg, "missing") == 0) named = .false.
        end do

        call expect(all_refused, &
            "every incomplete device product set is refused", failures)
        call expect(named, "and the refusal names what is missing", failures)

        ! Specifically: the missing product is named, not merely mentioned.
        support%jvp = .true.
        support%vjp = .true.
        support%hvp = .false.
        call fortbo_decide_placement(posterior, support, .true., .true., &
            placement, status)
        call expect(index(status%msg, "hvp") > 0, &
            "a missing hvp is named as hvp", failures)
        call expect(index(placement%reason, "hvp") > 0, &
            "and the placement carries the same reason", failures)
        call expect(placement%derivative_bearing, &
            "the graph is recorded as derivative-bearing", failures)
    end subroutine check_partial_products_are_refused

    subroutine check_complete_products_are_allowed(failures)
        integer, intent(inout) :: failures
        type(differentiable_t) :: posterior
        type(fortbo_device_derivative_support_t) :: support
        type(fortbo_placement_t) :: placement
        type(fortnum_status_t) :: status

        support%jvp = .true.
        support%vjp = .true.
        support%hvp = .true.
        call expect(support%complete(), "a full set reports complete", failures)
        call expect(support%missing() == "none", &
            "and reports nothing missing", failures)

        call fortbo_decide_placement(posterior, support, .true., .true., &
            placement, status)
        call expect(status%code == FORTNUM_OK, &
            "a complete product set permits a device placement", failures)
        call expect(placement%site == FORTBO_PLACE_DEVICE, &
            "and the placement really is the device", failures)
    end subroutine check_complete_products_are_allowed

    !! A value-only graph has no derivatives to be missing, so the rule does
    !! not apply to it. Blocking it too would be a different and needlessly
    !! conservative rule -- and would have made the device kernels in
    !! `fortbo_device` unreachable, which are value-only by construction.
    subroutine check_value_only_graphs_are_not_blocked(failures)
        integer, intent(inout) :: failures
        type(value_only_t) :: posterior
        type(fortbo_device_derivative_support_t) :: support
        type(fortbo_placement_t) :: placement
        type(fortnum_status_t) :: status

        ! No device derivative products at all: the situation FortML's exact
        ! GP regression is actually in today.
        support%jvp = .false.
        support%vjp = .false.
        support%hvp = .false.

        call fortbo_decide_placement(posterior, support, .true., .true., &
            placement, status)
        call expect(status%code == FORTNUM_OK, &
            "a value-only graph is allowed on device without derivative "// &
            "products", failures)
        call expect(placement%site == FORTBO_PLACE_DEVICE, &
            "and is placed there", failures)
        call expect(.not. placement%derivative_bearing, &
            "and is recorded as carrying no derivative work", failures)
        ! The reason must not imply the products exist.
        call expect(index(placement%reason, "no derivative products required") &
            > 0, "the reason says why the products were irrelevant", failures)
    end subroutine check_value_only_graphs_are_not_blocked

    subroutine check_absent_device_is_refused(failures)
        integer, intent(inout) :: failures
        type(value_only_t) :: posterior
        type(fortbo_device_derivative_support_t) :: support
        type(fortbo_placement_t) :: placement
        type(fortnum_status_t) :: status

        support%jvp = .true.
        support%vjp = .true.
        support%hvp = .true.
        call fortbo_decide_placement(posterior, support, .true., .false., &
            placement, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "asking for a device that is not there is refused", failures)
        call expect(placement%site == FORTBO_PLACE_HOST, &
            "and the placement does not claim a device", failures)
        call expect(index(placement%reason, "no device") > 0, &
            "and says the device was absent, not that products were missing", &
            failures)
    end subroutine check_absent_device_is_refused

    !! A host request is not a failure and must not be reported as one.
    subroutine check_host_requests_are_honoured(failures)
        integer, intent(inout) :: failures
        type(differentiable_t) :: posterior
        type(fortbo_device_derivative_support_t) :: support
        type(fortbo_placement_t) :: placement
        type(fortnum_status_t) :: status

        support%jvp = .true.
        support%vjp = .true.
        support%hvp = .true.
        call fortbo_decide_placement(posterior, support, .false., .true., &
            placement, status)
        call expect(status%code == FORTNUM_OK, &
            "a host request succeeds even where a device was available", &
            failures)
        call expect(placement%site == FORTBO_PLACE_HOST, &
            "and is honoured rather than upgraded", failures)
        call expect(len_trim(placement%reason) > 0, &
            "a host placement still records why", failures)
    end subroutine check_host_requests_are_honoured

    subroutine check_names(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_placement_name(FORTBO_PLACE_HOST) == "host" .and. &
            fortbo_placement_name(FORTBO_PLACE_DEVICE) == "device", &
            "placements are nameable for a benchmark row", failures)
        call expect(fortbo_placement_name(-1) == "unknown", &
            "an unrecognized placement is named unknown", failures)
    end subroutine check_names

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_placement
