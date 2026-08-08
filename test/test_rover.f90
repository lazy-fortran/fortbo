program test_rover
    !! BO6: the 60-dimensional rover trajectory fixture.
    !!
    !! Oracles:
    !!
    !!   * the reward decomposes exactly as the paper states, checked by
    !!     constructing trajectories whose collision count and endpoint
    !!     distances are known by hand, so the arithmetic is verified against
    !!     the formula rather than against itself;
    !!   * the fixture is *hard in the right way*: a straight line from start to
    !!     goal must collide, or the obstacle map would be decorative and the
    !!     problem would be a 60-dimensional way of writing down two endpoints;
    !!   * the gradient is refused rather than returned, since the collision
    !!     term is piecewise constant;
    !!   * the spline is a spline: it responds to interior control points, which
    !!     a fixture that only read its endpoints would not.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_rover, only: fortbo_rover_t, FORTBO_ROVER_DIMENSION, &
        FORTBO_ROVER_POINTS, FORTBO_ROVER_COLLISION_COST, &
        FORTBO_ROVER_ENDPOINT_WEIGHT, FORTBO_ROVER_OFFSET
    implicit none

    integer :: failures

    failures = 0
    call check_dimension_and_bounds(failures)
    call check_reward_decomposition(failures)
    call check_the_straight_line_collides(failures)
    call check_the_spline_uses_its_interior(failures)
    call check_gradient_is_refused(failures)

    if (failures == 0) then
        print *, "test_rover: PASS"
    else
        print *, "test_rover: FAIL", failures
        error stop 1
    end if

contains

    !! A trajectory whose control points all sit at one location.
    subroutine constant_trajectory(where, x)
        real(dp), intent(in) :: where(2)
        real(dp), intent(out) :: x(FORTBO_ROVER_DIMENSION)
        real(dp) :: control(2, FORTBO_ROVER_POINTS)
        integer :: k

        do k = 1, FORTBO_ROVER_POINTS
            control(:, k) = where
        end do
        x = reshape(control, [FORTBO_ROVER_DIMENSION])
    end subroutine constant_trajectory

    subroutine check_dimension_and_bounds(failures)
        integer, intent(inout) :: failures
        type(fortbo_rover_t) :: rover
        type(fortnum_status_t) :: status
        real(dp) :: lower(FORTBO_ROVER_DIMENSION), upper(FORTBO_ROVER_DIMENSION)

        call expect(rover%dimension() == 60, &
            "the rover problem is sixty dimensional", failures)
        call rover%bounds(lower, upper, status)
        call expect(status%code == FORTNUM_OK .and. all(lower == 0.0_dp) .and. &
            all(upper == 1.0_dp), "the domain is the unit cube", failures)
    end subroutine check_dimension_and_bounds

    !! Against the formula, on a trajectory whose terms are known by hand. A
    !! constant trajectory parked in open ground has zero collisions, so the
    !! reward is entirely the endpoint penalty plus the offset.
    subroutine check_reward_decomposition(failures)
        integer, intent(inout) :: failures
        type(fortbo_rover_t) :: rover
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_ROVER_DIMENSION), value, expected, endpoint
        real(dp) :: parked(2)
        integer :: count

        ! A patch of open ground, away from every obstacle.
        parked = [0.05_dp, 0.40_dp]
        call constant_trajectory(parked, x)
        call rover%collisions(x, count, status)
        call expect(status%code == FORTNUM_OK .and. count == 0, &
            "a trajectory parked in the open collides with nothing", failures)

        call rover%value(x, value, status)
        endpoint = sum(abs(parked - rover%start)) + sum(abs(parked - rover%goal))
        expected = -(-FORTBO_ROVER_ENDPOINT_WEIGHT*endpoint + FORTBO_ROVER_OFFSET)
        call expect(abs(value - expected) < 1.0e-12_dp, &
            "the reward is exactly the paper's endpoint penalty plus offset", &
            failures)

        ! Parking inside an obstacle charges the collision cost at every
        ! sample, which pins the magnitude the paper states.
        call constant_trajectory([0.25_dp, 0.25_dp], x)
        call rover%collisions(x, count, status)
        call expect(count > 0, "a trajectory parked in an obstacle collides", &
            failures)
        call rover%value(x, value, status)
        call expect(value > expected, &
            "colliding is worse than not colliding", failures)

        ! And by exactly the stated amount per collision.
        block
            real(dp) :: collision_part, clear_value
            call constant_trajectory([0.25_dp, 0.25_dp], x)
            call rover%value(x, value, status)
            endpoint = sum(abs([0.25_dp, 0.25_dp] - rover%start)) &
                + sum(abs([0.25_dp, 0.25_dp] - rover%goal))
            clear_value = -(-FORTBO_ROVER_ENDPOINT_WEIGHT*endpoint &
                + FORTBO_ROVER_OFFSET)
            collision_part = value - clear_value
            call expect(abs(collision_part &
                - FORTBO_ROVER_COLLISION_COST*real(count, dp)) < 1.0e-9_dp, &
                "each collision costs exactly the stated penalty", failures)
        end block
    end subroutine check_reward_decomposition

    !! If the straight line were collision-free the obstacle map would be
    !! decorative, and the fixture would reduce to writing two endpoints in
    !! sixty numbers.
    subroutine check_the_straight_line_collides(failures)
        integer, intent(inout) :: failures
        type(fortbo_rover_t) :: rover
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_ROVER_DIMENSION)
        real(dp) :: control(2, FORTBO_ROVER_POINTS), t
        integer :: k, count

        do k = 1, FORTBO_ROVER_POINTS
            t = real(k - 1, dp)/real(FORTBO_ROVER_POINTS - 1, dp)
            control(:, k) = rover%start + t*(rover%goal - rover%start)
        end do
        x = reshape(control, [FORTBO_ROVER_DIMENSION])

        call rover%collisions(x, count, status)
        call expect(status%code == FORTNUM_OK .and. count > 0, &
            "the direct route collides, so the obstacles matter", failures)
    end subroutine check_the_straight_line_collides

    !! A fixture that read only its first and last control points would pass
    !! every check above. This one does not.
    subroutine check_the_spline_uses_its_interior(failures)
        integer, intent(inout) :: failures
        type(fortbo_rover_t) :: rover
        type(fortnum_status_t) :: status
        real(dp) :: direct(FORTBO_ROVER_DIMENSION)
        real(dp) :: detour(FORTBO_ROVER_DIMENSION)
        real(dp) :: control(2, FORTBO_ROVER_POINTS), t
        real(dp) :: direct_value, detour_value
        integer :: k, direct_hits, detour_hits

        do k = 1, FORTBO_ROVER_POINTS
            t = real(k - 1, dp)/real(FORTBO_ROVER_POINTS - 1, dp)
            control(:, k) = rover%start + t*(rover%goal - rover%start)
        end do
        direct = reshape(control, [FORTBO_ROVER_DIMENSION])

        ! Same endpoints, interior pushed toward the corridor the map leaves
        ! open. Only an implementation that actually evaluates the spline can
        ! tell these apart.
        do k = 1, FORTBO_ROVER_POINTS
            t = real(k - 1, dp)/real(FORTBO_ROVER_POINTS - 1, dp)
            control(1, k) = rover%start(1) + t*(rover%goal(1) - rover%start(1))
            control(2, k) = rover%start(2) + t*(rover%goal(2) - rover%start(2)) &
                + 0.35_dp*sin(3.14159265358979_dp*t)
        end do
        detour = reshape(control, [FORTBO_ROVER_DIMENSION])

        call rover%collisions(direct, direct_hits, status)
        call rover%collisions(detour, detour_hits, status)
        call expect(direct_hits /= detour_hits, &
            "moving the interior control points changes the collision count", &
            failures)

        call rover%value(direct, direct_value, status)
        call rover%value(detour, detour_value, status)
        call expect(abs(direct_value - detour_value) > 1.0e-9_dp, &
            "the objective responds to the interior of the trajectory", failures)
    end subroutine check_the_spline_uses_its_interior

    subroutine check_gradient_is_refused(failures)
        integer, intent(inout) :: failures
        type(fortbo_rover_t) :: rover
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_ROVER_DIMENSION)
        real(dp) :: gradient(FORTBO_ROVER_DIMENSION)
        real(dp) :: value
        real(dp) :: short(3)

        call constant_trajectory([0.5_dp, 0.5_dp], x)
        call rover%gradient(x, gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "the gradient is refused by name, not silently differenced", &
            failures)

        call rover%value(short, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a trajectory of the wrong length is refused", failures)
    end subroutine check_gradient_is_refused

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_rover
