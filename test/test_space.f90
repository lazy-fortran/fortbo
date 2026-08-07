program test_space
    !! BO0: normalized search spaces.
    !!
    !! Oracles:
    !!   * continuous and log-scaled encodings are checked against values
    !!     computed by hand in the test, not against the encoder's own inverse;
    !!   * the round trip is checked over a deterministic sweep of the unit
    !!     cube, which catches any encode/decode pair that is merely
    !!     self-consistent rather than correct;
    !!   * integer rounding is checked at the cell boundaries, where an
    !!     off-by-one in the affine map shows up and nowhere else;
    !!   * conditional activation is checked by the observable rule that two
    !!     unit points differing only in an inactive coordinate must decode to
    !!     the same user point.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_space, only: fortbo_space_t
    implicit none

    integer :: failures

    failures = 0
    call check_continuous(failures)
    call check_log_scale(failures)
    call check_integer_rounding(failures)
    call check_categorical(failures)
    call check_mixed_round_trip(failures)
    call check_differentiable_mask(failures)
    call check_conditionals(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_space: PASS"
    else
        print *, "test_space: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_continuous(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: values(1), unit(1)

        call space%add_continuous("x", -2.0_dp, 6.0_dp, status)
        call space%finalize(status)
        call expect(space%n_coordinates() == 1, "one coordinate per continuous", &
            failures)

        call space%to_unit([-2.0_dp], unit, status)
        call expect_close(unit(1), 0.0_dp, 1.0e-14_dp, "the lower bound maps to zero", &
            failures)
        call space%to_unit([6.0_dp], unit, status)
        call expect_close(unit(1), 1.0_dp, 1.0e-14_dp, "the upper bound maps to one", &
            failures)
        call space%to_unit([0.0_dp], unit, status)
        call expect_close(unit(1), 0.25_dp, 1.0e-14_dp, &
            "an interior value maps affinely", failures)

        call space%from_unit([0.75_dp], values, status)
        call expect_close(values(1), 4.0_dp, 1.0e-14_dp, "decoding is the inverse map", &
            failures)
    end subroutine check_continuous

    !! A log-scaled variable must be uniform in the exponent, so the geometric
    !! mean of the bounds sits at the midpoint of the unit interval.
    subroutine check_log_scale(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: values(1), unit(1)

        call space%add_continuous("rate", 1.0e-4_dp, 1.0e0_dp, status, log_scale=.true.)
        call space%finalize(status)

        call space%to_unit([1.0e-2_dp], unit, status)
        call expect_close(unit(1), 0.5_dp, 1.0e-12_dp, &
            "the geometric mean is the unit midpoint", failures)
        call space%from_unit([0.25_dp], values, status)
        call expect_close(values(1), 1.0e-3_dp, 1.0e-12_dp, &
            "a quarter of the way is one decade up", failures)
        call space%from_unit([0.0_dp], values, status)
        call expect_close(values(1), 1.0e-4_dp, 1.0e-14_dp, &
            "zero decodes to the lower bound", failures)
    end subroutine check_log_scale

    !! The affine map assigns each integer a cell of width 1/(upper-lower).
    !! Checking just inside each cell boundary is what catches an off-by-one.
    subroutine check_integer_rounding(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: values(1), unit(1)
        integer :: k
        logical :: exact

        call space%add_integer("n", 2, 6, status)
        call space%finalize(status)

        call space%from_unit([0.0_dp], values, status)
        call expect_close(values(1), 2.0_dp, 1.0e-14_dp, "zero decodes to the lower", &
            failures)
        call space%from_unit([1.0_dp], values, status)
        call expect_close(values(1), 6.0_dp, 1.0e-14_dp, "one decodes to the upper", &
            failures)

        exact = .true.
        do k = 2, 6
            call space%to_unit([real(k, dp)], unit, status)
            call space%from_unit(unit, values, status)
            if (abs(values(1) - real(k, dp)) > 0.0_dp) exact = .false.
        end do
        call expect(exact, "every integer round-trips exactly", failures)

        call space%from_unit([0.5_dp/4.0_dp - 1.0e-6_dp], values, status)
        call expect_close(values(1), 2.0_dp, 1.0e-14_dp, &
            "just below the first boundary decodes down", failures)
        call space%from_unit([0.5_dp/4.0_dp + 1.0e-6_dp], values, status)
        call expect_close(values(1), 3.0_dp, 1.0e-14_dp, &
            "just above the first boundary decodes up", failures)
    end subroutine check_integer_rounding

    subroutine check_categorical(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: values(1), unit(3)

        call space%add_categorical("colour", 3, status)
        call space%finalize(status)
        call expect(space%n_coordinates() == 3, &
            "a categorical occupies one coordinate per category", failures)

        call space%to_unit([2.0_dp], unit, status)
        call expect_close(unit(1), 0.0_dp, 1.0e-14_dp, "one-hot is zero elsewhere", &
            failures)
        call expect_close(unit(2), 1.0_dp, 1.0e-14_dp, "one-hot is one at the category", &
            failures)
        call expect_close(unit(3), 0.0_dp, 1.0e-14_dp, "one-hot is zero after", failures)

        call space%from_unit([0.2_dp, 0.1_dp], values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a unit vector of the wrong width is refused", failures)
        call space%from_unit([0.2_dp, 0.1_dp, 0.9_dp], values, status)
        call expect_close(values(1), 3.0_dp, 1.0e-14_dp, "decoding takes the arg-max", &
            failures)
        call space%from_unit([0.5_dp, 0.5_dp, 0.1_dp], values, status)
        call expect_close(values(1), 1.0_dp, 1.0e-14_dp, &
            "a tie goes to the lowest index", failures)
    end subroutine check_categorical

    !! Sweep the cube deterministically and require that decode-then-encode is
    !! a projection: encoding a decoded point and decoding again must land on
    !! the same user point.
    subroutine check_mixed_round_trip(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: unit(6), unit_again(6), values(4), values_again(4)
        integer :: trial, i
        logical :: stable
        real(dp) :: phase

        call space%add_continuous("x", 0.0_dp, 10.0_dp, status)
        call space%add_integer("n", 1, 8, status)
        call space%add_categorical("mode", 3, status)
        call space%add_continuous("rate", 1.0e-3_dp, 1.0e1_dp, status, log_scale=.true.)
        call space%finalize(status)
        call expect(space%n_coordinates() == 6, "the layout packs to six coordinates", &
            failures)

        stable = .true.
        do trial = 0, 200
            phase = real(trial, dp)
            do i = 1, 6
                unit(i) = 0.5_dp + 0.5_dp*sin(0.7_dp*phase + 1.3_dp*real(i, dp))
            end do
            call space%from_unit(unit, values, status)
            if (status%code /= FORTNUM_OK) stable = .false.
            call space%to_unit(values, unit_again, status)
            if (status%code /= FORTNUM_OK) stable = .false.
            call space%from_unit(unit_again, values_again, status)
            if (status%code /= FORTNUM_OK) stable = .false.
            if (maxval(abs(values - values_again)) > 1.0e-9_dp) stable = .false.
        end do
        call expect(stable, "decode/encode is a stable projection over the cube", &
            failures)
    end subroutine check_mixed_round_trip

    subroutine check_differentiable_mask(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        logical :: mask(6)

        call space%add_continuous("x", 0.0_dp, 1.0_dp, status)
        call space%add_integer("n", 1, 8, status)
        call space%add_categorical("mode", 3, status)
        call space%add_continuous("y", 0.0_dp, 1.0_dp, status)
        call space%finalize(status)

        call space%differentiable_mask(mask, status)
        call expect(status%code == FORTNUM_OK, "the mask is reported", failures)
        call expect(mask(1), "a continuous coordinate is differentiable", failures)
        call expect(.not. mask(2), "an integer coordinate is not differentiable", &
            failures)
        call expect(.not. any(mask(3:5)), &
            "no one-hot coordinate is differentiable", failures)
        call expect(mask(6), "the trailing continuous coordinate is differentiable", &
            failures)
        call expect(count(mask) == 2, "exactly the continuous coordinates are marked", &
            failures)
    end subroutine check_differentiable_mask

    !! Observable rule: an inactive coordinate cannot change the decoded point.
    subroutine check_conditionals(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: unit_a(4), unit_b(4), values_a(3), values_b(3)
        logical :: mask(3)

        call space%add_categorical("kernel", 2, status)
        call space%add_continuous("nu", 0.5_dp, 2.5_dp, status)
        call space%add_continuous("degree", 1.0_dp, 5.0_dp, status)
        call space%finalize(status)
        call space%set_condition(2, 1, 1, status, default_value=0.5_dp)
        call expect(status%code == FORTNUM_OK, "a condition can be declared", failures)
        call space%set_condition(3, 1, 2, status, default_value=1.0_dp)

        ! Kernel category 1: nu is active, degree is not.
        unit_a = [1.0_dp, 0.0_dp, 0.9_dp, 0.2_dp]
        unit_b = [1.0_dp, 0.0_dp, 0.9_dp, 0.8_dp]
        call space%from_unit(unit_a, values_a, status)
        call space%from_unit(unit_b, values_b, status)
        call expect(maxval(abs(values_a - values_b)) == 0.0_dp, &
            "an inactive coordinate cannot change the decoded point", failures)
        call expect_close(values_a(3), 1.0_dp, 1.0e-14_dp, &
            "an inactive variable takes its default", failures)
        call expect_close(values_a(2), 0.5_dp + 0.9_dp*2.0_dp, 1.0e-12_dp, &
            "an active variable decodes normally", failures)

        call space%active_mask(values_a, mask, status)
        call expect(mask(1), "the parent is always active", failures)
        call expect(mask(2), "the selected child is active", failures)
        call expect(.not. mask(3), "the unselected child is inactive", failures)

        ! Kernel category 2 flips which child is active.
        unit_a = [0.0_dp, 1.0_dp, 0.9_dp, 0.5_dp]
        call space%from_unit(unit_a, values_a, status)
        call space%active_mask(values_a, mask, status)
        call expect(.not. mask(2), "the other child becomes inactive", failures)
        call expect(mask(3), "the other child becomes active", failures)
        call expect_close(values_a(2), 0.5_dp, 1.0e-14_dp, &
            "the newly inactive variable takes its default", failures)
    end subroutine check_conditionals

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space, other
        type(fortnum_status_t) :: status
        real(dp) :: unit(1), values(1)

        call space%add_continuous("bad", 1.0_dp, 1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an empty continuous range is refused", failures)
        call space%add_continuous("bad_log", -1.0_dp, 1.0_dp, status, log_scale=.true.)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-positive log-scaled bound is refused", failures)
        call space%add_categorical("bad_cat", 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a single-category variable is refused", failures)

        call space%add_continuous("x", 0.0_dp, 1.0_dp, status)
        call space%to_unit([0.5_dp], unit, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unfinalized space refuses encoding", failures)
        call space%finalize(status)
        call space%add_continuous("late", 0.0_dp, 1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a finalized space refuses new variables", failures)

        call space%to_unit([2.0_dp], unit, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an out-of-bounds value is refused", failures)

        call other%add_integer("n", 1, 4, status)
        call other%finalize(status)
        call other%to_unit([2.5_dp], unit, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-integral integer value is refused", failures)
        call other%from_unit([0.5_dp], values, status)
        call expect(status%code == FORTNUM_OK, "a valid decode still succeeds", failures)

        call other%set_condition(1, 1, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a self-parent condition is refused", failures)
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

    subroutine expect_close(actual, expected, tolerance, description, failures)
        real(dp), intent(in) :: actual
        real(dp), intent(in) :: expected
        real(dp), intent(in) :: tolerance
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (abs(actual - expected) > tolerance) then
            failures = failures + 1
            print *, "  FAIL: ", description, actual, expected
        end if
    end subroutine expect_close

end program test_space
