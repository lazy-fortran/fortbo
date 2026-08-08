program test_feasible
    !! BO3: fixed choices and constraint penalties.
    !!
    !! Oracles:
    !!
    !!   * a fixed choice must survive the round trip through the unit cube,
    !!     checked by decoding every candidate. That is what "no candidate can
    !!     violate it" means operationally;
    !!   * the two penalties must differ in the way the theory says they do,
    !!     and the test constructs the case that shows it: on a problem whose
    !!     unconstrained optimum is infeasible, the exact penalty reaches the
    !!     constraint boundary at a finite weight while the quadratic one lands
    !!     at `1 + 1/(2 rho)`, outside it for every finite weight;
    !!   * a slack constraint must never offset a violated one.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_space, only: fortbo_space_t
    use fortbo_feasible, only: fortbo_fixed_choice_t, fortbo_apply_fixed_choices, &
        fortbo_violation, fortbo_penalized_score, fortbo_penalty_name, &
        fortbo_penalty_is_differentiable, FORTBO_PENALTY_QUADRATIC, &
        FORTBO_PENALTY_EXACT
    implicit none

    integer :: failures

    failures = 0
    call check_fixed_choices_survive_the_round_trip(failures)
    call check_violation_does_not_offset(failures)
    call check_exact_penalty_binds_where_quadratic_does_not(failures)
    call check_penalty_reporting(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_feasible: PASS"
    else
        print *, "test_feasible: FAIL", failures
        error stop 1
    end if

contains

    subroutine build_space(space, status)
        type(fortbo_space_t), intent(out) :: space
        type(fortnum_status_t), intent(out) :: status

        call space%add_continuous("x", 0.0_dp, 1.0_dp, status)
        if (status%code /= FORTNUM_OK) return
        call space%add_integer("n", 1, 8, status)
        if (status%code /= FORTNUM_OK) return
        call space%add_categorical("mode", 4, status)
        if (status%code /= FORTNUM_OK) return
        call space%finalize(status)
    end subroutine build_space

    !! Pinning is a reparameterization, so *every* candidate must decode to the
    !! pinned value. Checking one would not distinguish a working pin from a
    !! lucky draw.
    subroutine check_fixed_choices_survive_the_round_trip(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        type(fortbo_fixed_choice_t) :: choices(2)
        real(dp), allocatable :: points(:, :), decoded(:)
        real(dp) :: before_first
        integer :: k, j
        logical :: pinned, continuous_untouched

        call build_space(space, status)
        call expect(status%code == FORTNUM_OK, "the space builds", failures)

        allocate (points(20, space%n_coordinates()))
        allocate (decoded(space%n_variables()))
        do k = 1, 20
            do j = 1, space%n_coordinates()
                points(k, j) = real(mod(7*k + 3*j, 97), dp)/97.0_dp
            end do
        end do
        before_first = points(1, 1)

        choices(1)%variable = 2
        choices(1)%value = 5.0_dp
        choices(2)%variable = 3
        choices(2)%value = 2.0_dp

        call fortbo_apply_fixed_choices(space, choices, points, status)
        call expect(status%code == FORTNUM_OK, "the fixed choices apply", failures)

        pinned = .true.
        do k = 1, 20
            call space%from_unit(points(k, :), decoded, status)
            if (status%code /= FORTNUM_OK) pinned = .false.
            if (abs(decoded(2) - 5.0_dp) > 1.0e-9_dp) pinned = .false.
            if (abs(decoded(3) - 2.0_dp) > 1.0e-9_dp) pinned = .false.
        end do
        call expect(pinned, "every candidate decodes to the pinned values", failures)

        ! The unpinned coordinate must be left alone: pinning removes a variable
        ! from the search, it does not reset the rest of the point.
        continuous_untouched = abs(points(1, 1) - before_first) < 1.0e-9_dp
        call expect(continuous_untouched, &
            "an unpinned coordinate is not disturbed", failures)
    end subroutine check_fixed_choices_survive_the_round_trip

    !! Comfortably satisfying one constraint must not buy the right to break
    !! another. Only positive parts count.
    subroutine check_violation_does_not_offset(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: constraints(2, 3), total(3)

        ! Point 1: both satisfied. Point 2: one badly violated, one very slack.
        ! Point 3: both mildly violated.
        constraints(:, 1) = [-1.0_dp, -2.0_dp]
        constraints(:, 2) = [3.0_dp, -100.0_dp]
        constraints(:, 3) = [0.5_dp, 0.5_dp]

        call fortbo_violation(constraints, total, status)
        call expect(status%code == FORTNUM_OK, "the violation computes", failures)
        call expect(total(1) == 0.0_dp, "a feasible point has no violation", &
            failures)
        call expect(abs(total(2) - 3.0_dp) < 1.0e-14_dp, &
            "a slack constraint does not offset a violated one", failures)
        call expect(abs(total(3) - 1.0_dp) < 1.0e-14_dp, &
            "violations accumulate across constraints", failures)
        call expect(total(2) > total(3), &
            "the badly infeasible point is ranked worse", failures)
    end subroutine check_violation_does_not_offset

    !! The behavioural difference between the two penalties.
    !!
    !! Minimize `base(t) = -t` subject to `t <= 1`, over `t` in `[0, 3]`. The
    !! unconstrained optimum is at the upper end and infeasible, so the penalty
    !! decides where the minimum lands. For the exact penalty any `rho > 1`
    !! puts it exactly on the boundary; for the quadratic penalty the minimum
    !! sits at `1 + 1/(2 rho)`, strictly outside for every finite weight.
    subroutine check_exact_penalty_binds_where_quadratic_does_not(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 3001
        real(dp) :: base(n), violation(n), score(n), grid(n)
        real(dp) :: rho
        integer :: k, best
        logical :: quadratic_always_outside

        do k = 1, n
            grid(k) = 3.0_dp*real(k - 1, dp)/real(n - 1, dp)
            base(k) = -grid(k)
            violation(k) = max(grid(k) - 1.0_dp, 0.0_dp)
        end do

        ! Exact penalty, weight above the threshold: the minimum is on the
        ! boundary.
        call fortbo_penalized_score(base, violation, 2.0_dp, FORTBO_PENALTY_EXACT, &
            score, status)
        call expect(status%code == FORTNUM_OK, "the exact penalty computes", &
            failures)
        best = minloc(score, dim=1)
        call expect(abs(grid(best) - 1.0_dp) < 1.0e-3_dp, &
            "the exact penalty puts the optimum on the constraint boundary", &
            failures)

        ! Below the threshold it does not bind, which is the honest caveat: the
        ! penalty is exact only above a finite weight.
        call fortbo_penalized_score(base, violation, 0.5_dp, FORTBO_PENALTY_EXACT, &
            score, status)
        best = minloc(score, dim=1)
        call expect(grid(best) > 1.0_dp + 1.0e-3_dp, &
            "an exact penalty below its threshold does not bind", failures)

        ! Quadratic penalty: the minimum sits at 1 + 1/(2 rho), outside the
        ! feasible region for every finite weight.
        !
        ! An earlier version asserted "outside for every weight tried" over
        ! rho up to 1e4. That was measuring the grid, not the penalty: the
        ! offset 1/(2 rho) falls to 5e-5 there while the grid spacing is 1e-3,
        ! so the reported minimizer lands exactly on the boundary. The claim is
        ! true in exact arithmetic and unobservable below grid resolution — and
        ! that is the practical point, since what a caller needs is a weight
        ! large enough to push the residual infeasibility under its tolerance.
        quadratic_always_outside = .true.
        do k = 1, 3
            rho = 10.0_dp**real(k - 1, dp)
            call fortbo_penalized_score(base, violation, rho, &
                FORTBO_PENALTY_QUADRATIC, score, status)
            best = minloc(score, dim=1)
            ! Resolvable while 1/(2 rho) exceeds the 1e-3 grid spacing.
            if (grid(best) <= 1.0_dp) quadratic_always_outside = .false.
            if (abs(grid(best) - (1.0_dp + 0.5_dp/rho)) > 2.0e-3_dp) &
                quadratic_always_outside = .false.
        end do
        call expect(quadratic_always_outside, &
            "the quadratic optimum sits at 1 + 1/(2 rho), outside the region", &
            failures)

        ! Raising the weight brings it in, so the two penalties really do
        ! describe the same constraint.
        call fortbo_penalized_score(base, violation, 1.0_dp, &
            FORTBO_PENALTY_QUADRATIC, score, status)
        best = minloc(score, dim=1)
        call expect(grid(best) > 1.4_dp, &
            "a weak quadratic weight leaves the optimum well outside", failures)
        call fortbo_penalized_score(base, violation, 100.0_dp, &
            FORTBO_PENALTY_QUADRATIC, score, status)
        best = minloc(score, dim=1)
        call expect(grid(best) < 1.01_dp, &
            "a larger quadratic weight brings the optimum closer", failures)

        ! A feasible point is untouched by either penalty.
        call fortbo_penalized_score([1.0_dp], [0.0_dp], 1000.0_dp, &
            FORTBO_PENALTY_EXACT, score(1:1), status)
        call expect(score(1) == 1.0_dp, &
            "a feasible point keeps its unpenalized score", failures)
    end subroutine check_exact_penalty_binds_where_quadratic_does_not

    !! A caller intending to use gradients must be told about the kink before
    !! it starts, not discover it from a line search that will not converge.
    subroutine check_penalty_reporting(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_penalty_is_differentiable(FORTBO_PENALTY_QUADRATIC), &
            "the quadratic penalty is reported as smooth", failures)
        call expect(.not. fortbo_penalty_is_differentiable(FORTBO_PENALTY_EXACT), &
            "the exact penalty is reported as non-smooth", failures)
        call expect(fortbo_penalty_name(FORTBO_PENALTY_EXACT) == "exact", &
            "the penalty is nameable for the run record", failures)
        call expect(fortbo_penalty_name(0) == "unknown", &
            "an unrecognized penalty is named unknown", failures)
    end subroutine check_penalty_reporting

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        type(fortbo_fixed_choice_t) :: choices(1)
        real(dp), allocatable :: points(:, :)
        real(dp) :: score(2)

        call build_space(space, status)
        allocate (points(2, space%n_coordinates()))
        points = 0.5_dp

        choices(1)%variable = 99
        choices(1)%value = 1.0_dp
        call fortbo_apply_fixed_choices(space, choices, points, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an out-of-range variable is refused", failures)

        choices(1)%variable = 3
        choices(1)%value = 9.0_dp
        call fortbo_apply_fixed_choices(space, choices, points, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a category outside the variable's range is refused", failures)

        choices(1)%variable = 2
        choices(1)%value = 100.0_dp
        call fortbo_apply_fixed_choices(space, choices, points, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an integer outside its bounds is refused", failures)

        call fortbo_penalized_score([1.0_dp, 1.0_dp], [0.0_dp, 0.0_dp], -1.0_dp, &
            FORTBO_PENALTY_EXACT, score, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative penalty weight is refused, since it rewards violation", &
            failures)

        call fortbo_penalized_score([1.0_dp, 1.0_dp], [-1.0_dp, 0.0_dp], 1.0_dp, &
            FORTBO_PENALTY_EXACT, score, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative violation is refused", failures)

        call fortbo_penalized_score([1.0_dp, 1.0_dp], [0.0_dp, 0.0_dp], 1.0_dp, &
            42, score, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unknown penalty kind is refused", failures)
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

end program test_feasible
