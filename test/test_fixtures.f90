program test_fixtures
    !! BO6: constrained and multi-objective fixtures.
    !!
    !! Oracles:
    !!
    !!   * the analytic Pareto fronts are checked against the fixtures
    !!     themselves — every front point must be attainable by a real input,
    !!     and no sampled input may dominate a front point. A stated front that
    !!     nothing attains, or that a random point beats, is worse than no front
    !!     at all;
    !!   * the front's hypervolume is checked against `fortbo_pareto`, so the
    !!     fixture and the indicator agree;
    !!   * the constrained fixtures are checked for the property that makes them
    !!     worth having: the constraint must be *active* at the optimum, since a
    !!     fixture whose constraint is slack there cannot distinguish a method
    !!     that respects it from one that ignores it;
    !!   * the concave front must be shown unreachable by weighted-sum
    !!     scalarization, which is why it is in the suite.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_fixtures, only: fortbo_constrained_fixture_t, &
        fortbo_multi_objective_fixture_t, FORTBO_CONSTRAINED_TOWNSEND, &
        FORTBO_CONSTRAINED_GARDNER, FORTBO_MULTI_ZDT1, FORTBO_MULTI_ZDT2
    use fortbo_pareto, only: fortbo_dominates, fortbo_hypervolume
    implicit none

    integer :: failures

    failures = 0
    call check_front_points_are_attainable(failures)
    call check_nothing_dominates_the_front(failures)
    call check_front_hypervolume_agrees_with_pareto(failures)
    call check_concave_front_defeats_weighted_sum(failures)
    call check_constraints_are_active_at_the_optimum(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_fixtures: PASS"
    else
        print *, "test_fixtures: FAIL", failures
        error stop 1
    end if

contains

    !! Every stated front point must be produced by a real input: the front is
    !! attained where the trailing coordinates are zero.
    subroutine check_front_points_are_attainable(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_objective_fixture_t) :: fixture
        type(fortnum_status_t) :: status
        real(dp) :: point(5), objectives(2), front(2)
        integer :: kinds(2), c, k
        logical :: attained

        kinds = [FORTBO_MULTI_ZDT1, FORTBO_MULTI_ZDT2]
        do c = 1, 2
            fixture%kind = kinds(c)
            fixture%dimension = 5
            attained = .true.
            do k = 0, 20
                point = 0.0_dp
                point(1) = real(k, dp)/20.0_dp
                call fixture%evaluate(point, objectives, status)
                if (status%code /= FORTNUM_OK) attained = .false.
                call fixture%front_point(point(1), front, status)
                if (status%code /= FORTNUM_OK) attained = .false.
                if (maxval(abs(objectives - front)) > 1.0e-12_dp) attained = .false.
            end do
            call expect(attained, &
                "every stated front point is attained by a real input", failures)
        end do
    end subroutine check_front_points_are_attainable

    !! No input may dominate a front point. A "front" that a random draw beats
    !! is not a front, and would make every hypervolume comparison meaningless.
    subroutine check_nothing_dominates_the_front(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_objective_fixture_t) :: fixture
        type(fortnum_status_t) :: status
        real(dp) :: point(4), objectives(2), front(2)
        integer :: kinds(2), c, k, j, seed
        logical :: safe

        kinds = [FORTBO_MULTI_ZDT1, FORTBO_MULTI_ZDT2]
        do c = 1, 2
            fixture%kind = kinds(c)
            fixture%dimension = 4
            safe = .true.
            seed = 1
            do k = 1, 400
                ! A deterministic scatter over the cube; no RNG needed and the
                ! sweep is reproducible.
                do j = 1, 4
                    seed = mod(seed*1103515245 + 12345, 2147483647)
                    point(j) = real(abs(seed), dp)/2147483647.0_dp
                end do
                call fixture%evaluate(point, objectives, status)
                if (status%code /= FORTNUM_OK) safe = .false.
                call fixture%front_point(objectives(1), front, status)
                if (status%code /= FORTNUM_OK) cycle
                if (fortbo_dominates(objectives, front)) safe = .false.
            end do
            call expect(safe, "no sampled input dominates the stated front", &
                failures)
        end do
    end subroutine check_nothing_dominates_the_front

    !! The fixture's front and the package's hypervolume must agree, so a
    !! benchmark comparing a run's hypervolume against the front's is comparing
    !! like with like.
    subroutine check_front_hypervolume_agrees_with_pareto(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_objective_fixture_t) :: fixture
        type(fortnum_status_t) :: status
        integer, parameter :: n = 200
        real(dp) :: front(n, 2), reference(2), volume, partial
        integer :: k

        fixture%kind = FORTBO_MULTI_ZDT1
        fixture%dimension = 3
        call fixture%reference_point(reference)

        do k = 1, n
            call fixture%front_point(real(k - 1, dp)/real(n - 1, dp), &
                front(k, :), status)
        end do
        call fortbo_hypervolume(front, reference, volume, status)
        call expect(status%code == FORTNUM_OK, "the front's hypervolume computes", &
            failures)
        call expect(volume > 0.0_dp, "the front dominates the reference point", &
            failures)

        ! A dense front must beat a sparse one: hypervolume is monotone in the
        ! set, so this catches a reference point on the wrong side.
        call fortbo_hypervolume(front(1:10, :), reference, partial, status)
        call expect(partial < volume, &
            "a denser front has the larger hypervolume", failures)

        ! The analytic value of the ZDT1 front's hypervolume against (1.1, 1.1)
        ! is 1.1*1.1 minus the area under f2 = 1 - sqrt(f1) shifted into the
        ! box, which the dense sum must approach from below.
        call expect(volume < 1.21_dp, &
            "the hypervolume cannot exceed the reference box", failures)
    end subroutine check_front_hypervolume_agrees_with_pareto

    !! Why the concave fixture is in the suite. A weighted sum can only ever
    !! find points on the convex hull of the front, so on ZDT2 it collapses to
    !! the two endpoints however the weights are chosen.
    subroutine check_concave_front_defeats_weighted_sum(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_objective_fixture_t) :: convex, concave
        type(fortnum_status_t) :: status
        integer, parameter :: n = 401
        real(dp) :: front(2), best_value, value, weight
        real(dp) :: chosen_convex, chosen_concave
        integer :: k, w
        logical :: interior_reached

        convex%kind = FORTBO_MULTI_ZDT1
        concave%kind = FORTBO_MULTI_ZDT2
        call expect(convex%front_is_convex(), "ZDT1's front is reported convex", &
            failures)
        call expect(.not. concave%front_is_convex(), &
            "ZDT2's front is reported concave", failures)

        ! Sweep the weights. On the convex front the scalarizer's minimizer
        ! moves across the interior; on the concave one it never leaves an end.
        interior_reached = .false.
        do w = 1, 9
            weight = 0.1_dp*real(w, dp)
            best_value = huge(1.0_dp)
            chosen_convex = 0.0_dp
            do k = 1, n
                call convex%front_point(real(k - 1, dp)/real(n - 1, dp), front, &
                    status)
                value = weight*front(1) + (1.0_dp - weight)*front(2)
                if (value < best_value) then
                    best_value = value
                    chosen_convex = front(1)
                end if
            end do
            if (chosen_convex > 0.05_dp .and. chosen_convex < 0.95_dp) &
                interior_reached = .true.
        end do
        call expect(interior_reached, &
            "a weighted sum reaches the interior of a convex front", failures)

        interior_reached = .false.
        do w = 1, 9
            weight = 0.1_dp*real(w, dp)
            best_value = huge(1.0_dp)
            chosen_concave = 0.0_dp
            do k = 1, n
                call concave%front_point(real(k - 1, dp)/real(n - 1, dp), front, &
                    status)
                value = weight*front(1) + (1.0_dp - weight)*front(2)
                if (value < best_value) then
                    best_value = value
                    chosen_concave = front(1)
                end if
            end do
            if (chosen_concave > 0.05_dp .and. chosen_concave < 0.95_dp) &
                interior_reached = .true.
        end do
        call expect(.not. interior_reached, &
            "a weighted sum never reaches the interior of a concave front", &
            failures)
    end subroutine check_concave_front_defeats_weighted_sum

    !! A fixture whose constraint is slack at the optimum cannot distinguish a
    !! method that respects the constraint from one that ignores it, so it is
    !! worthless as evidence. Both fixtures are checked to have a constraint
    !! that actually bites: the unconstrained optimum over the box must be
    !! infeasible.
    subroutine check_constraints_are_active_at_the_optimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_constrained_fixture_t) :: fixture
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), point(2)
        real(dp) :: value, best_free, best_feasible
        real(dp) :: free_point(2)
        integer, parameter :: n = 120
        integer :: kinds(2), c, i, j
        logical :: feasible, free_is_infeasible, any_feasible

        kinds = [FORTBO_CONSTRAINED_GARDNER, FORTBO_CONSTRAINED_TOWNSEND]
        do c = 1, 2
            fixture%kind = kinds(c)
            call fixture%bounds(lower, upper, status)
            call expect(status%code == FORTNUM_OK, "the fixture has bounds", &
                failures)

            best_free = huge(1.0_dp)
            best_feasible = huge(1.0_dp)
            free_point = lower
            any_feasible = .false.
            do i = 0, n
                do j = 0, n
                    point(1) = lower(1) + (upper(1) - lower(1))*real(i, dp) &
                        /real(n, dp)
                    point(2) = lower(2) + (upper(2) - lower(2))*real(j, dp) &
                        /real(n, dp)
                    call fixture%objective(point, value, status)
                    if (status%code /= FORTNUM_OK) cycle
                    if (value < best_free) then
                        best_free = value
                        free_point = point
                    end if
                    call fixture%is_feasible(point, feasible, status)
                    if (.not. feasible) cycle
                    any_feasible = .true.
                    best_feasible = min(best_feasible, value)
                end do
            end do

            call expect(any_feasible, "the feasible region is not empty", failures)
            call fixture%is_feasible(free_point, feasible, status)
            free_is_infeasible = .not. feasible
            call expect(free_is_infeasible, &
                "the unconstrained optimum is infeasible, so the constraint bites", &
                failures)
            call expect(best_feasible > best_free, &
                "the constraint costs something at the optimum", failures)
        end do
    end subroutine check_constraints_are_active_at_the_optimum

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_objective_fixture_t) :: fixture
        type(fortbo_constrained_fixture_t) :: constrained
        type(fortnum_status_t) :: status
        real(dp) :: objectives(2), point(3), values(1), value

        fixture%kind = FORTBO_MULTI_ZDT1
        fixture%dimension = 3

        call fixture%front_point(1.5_dp, objectives, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a front parameter outside [0,1] is refused", failures)

        point = [0.5_dp, 2.0_dp, 0.5_dp]
        call fixture%evaluate(point, objectives, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a point outside the unit cube is refused", failures)

        fixture%kind = 99
        call fixture%evaluate([0.5_dp, 0.5_dp, 0.5_dp], objectives, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unknown multi-objective fixture is refused", failures)

        constrained%kind = 99
        call constrained%objective([0.0_dp, 0.0_dp], value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unknown constrained fixture is refused", failures)

        constrained%kind = FORTBO_CONSTRAINED_GARDNER
        call constrained%constraints([0.0_dp, 0.0_dp, 0.0_dp], values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-sized constrained point is refused", failures)
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

end program test_fixtures
