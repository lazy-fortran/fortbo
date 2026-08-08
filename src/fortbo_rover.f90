module fortbo_rover
    !! The 60-dimensional rover trajectory fixture (ROADMAP BO6).
    !!
    !! Reward structure from the TuRBO paper (arXiv:1910.01739, appendix F.2),
    !! read rather than recalled:
    !!
    !!     f(x) = c(x) - 10 (||x_{1,2} - x_s||_1 + ||x_{59,60} - x_g||_1) + 5
    !!
    !! where the trajectory is a B-spline through 30 points in the plane and
    !! `c(x)` charges `-20` for each collision along it. FortBO minimizes, so
    !! the fixture returns `-f`.
    !!
    !! **The obstacle map is ours, and that is the honest part.** The paper
    !! states the reward and cites Wang et al. for the terrain; the layout
    !! itself is not in the paper, so reproducing the published *numbers* is not
    !! possible from it. Inventing a map and calling the result "the rover
    !! problem" would produce a fixture whose scores look comparable to the
    !! literature and are not. What is reproduced is the *structure* — 60
    !! dimensions, a spline trajectory, a collision penalty of the stated
    !! magnitude, endpoint penalties with the stated weight — which is what
    !! makes it a useful high-dimensional test. Any claim against published
    !! numbers must say which map it used.
    !!
    !! The problem is hard in a specific way worth stating: the objective is
    !! **piecewise constant in the collision term**. Moving a control point
    !! slightly usually changes nothing at all, then changes the value by 20 all
    !! at once. A gradient of the collision term is zero almost everywhere and
    !! undefined on the boundaries, so this fixture is deliberately *not*
    !! differentiable and exposes no gradient — a derivative-based policy must
    !! refuse it rather than difference through a step function.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortbo_rover_t

    !! Control points, and hence dimension: 30 points in the plane.
    integer, parameter, public :: FORTBO_ROVER_POINTS = 30
    integer, parameter, public :: FORTBO_ROVER_DIMENSION = 2*FORTBO_ROVER_POINTS

    !! Penalties, from the paper.
    real(dp), parameter, public :: FORTBO_ROVER_COLLISION_COST = 20.0_dp
    real(dp), parameter, public :: FORTBO_ROVER_ENDPOINT_WEIGHT = 10.0_dp
    real(dp), parameter, public :: FORTBO_ROVER_OFFSET = 5.0_dp

    !! Samples taken along the spline when testing for collisions. The
    !! trajectory is continuous and the test is discrete, so this is the
    !! resolution at which a thin obstacle can be missed entirely — stated
    !! because it is a property of the fixture, not of a method run on it.
    integer, parameter, public :: FORTBO_ROVER_SAMPLES = 300

    type :: fortbo_rover_t
        !! Desired start and goal, in the unit square.
        real(dp) :: start(2) = [0.05_dp, 0.05_dp]
        real(dp) :: goal(2) = [0.95_dp, 0.95_dp]
    contains
        procedure, public :: dimension => rover_dimension
        procedure, public :: bounds => rover_bounds
        procedure, public :: value => rover_value
        procedure, public :: gradient => rover_gradient
        procedure, public :: collisions => rover_collisions
    end type fortbo_rover_t

    !! Axis-aligned rectangular obstacles, as `[x_low, y_low, x_high, y_high]`.
    !! Ours, not the paper's. Chosen to leave a navigable corridor so the
    !! problem is solvable, while forcing a detour so that the straight line
    !! from start to goal is not the answer — a map whose optimum were the
    !! straight line would make the collision term decorative.
    real(dp), parameter :: OBSTACLES(4, 8) = reshape([ &
        0.20_dp, 0.00_dp, 0.30_dp, 0.55_dp, &
        0.20_dp, 0.70_dp, 0.30_dp, 1.00_dp, &
        0.45_dp, 0.45_dp, 0.55_dp, 1.00_dp, &
        0.45_dp, 0.00_dp, 0.55_dp, 0.30_dp, &
        0.70_dp, 0.10_dp, 0.80_dp, 0.65_dp, &
        0.70_dp, 0.80_dp, 0.80_dp, 1.00_dp, &
        0.10_dp, 0.85_dp, 0.15_dp, 0.95_dp, &
        0.85_dp, 0.35_dp, 0.90_dp, 0.45_dp], [4, 8])

contains

    pure integer function rover_dimension(self) result(d)
        class(fortbo_rover_t), intent(in) :: self

        d = FORTBO_ROVER_DIMENSION
    end function rover_dimension

    subroutine rover_bounds(self, lower, upper, status)
        class(fortbo_rover_t), intent(in) :: self
        real(dp), intent(out) :: lower(:)
        real(dp), intent(out) :: upper(:)
        type(fortnum_status_t), intent(out) :: status

        lower = 0.0_dp
        upper = 0.0_dp
        if (size(lower) /= FORTBO_ROVER_DIMENSION .or. &
            size(upper) /= FORTBO_ROVER_DIMENSION) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo rover: bounds must have sixty entries")
            return
        end if
        lower = 0.0_dp
        upper = 1.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine rover_bounds

    !! A point on the trajectory at parameter `t` in `[0, 1]`.
    !!
    !! Uniform cubic B-spline through the control points, which is what the
    !! paper specifies. The spline does *not* interpolate its control points,
    !! so the endpoint penalties are charged on the first and last control
    !! points as the reward states, not on the curve's own ends.
    pure subroutine spline_point(control, t, point)
        real(dp), intent(in) :: control(2, FORTBO_ROVER_POINTS)
        real(dp), intent(in) :: t
        real(dp), intent(out) :: point(2)
        real(dp) :: scaled, local, basis(4)
        integer :: segment, k, index

        scaled = t*real(FORTBO_ROVER_POINTS - 3, dp)
        segment = min(int(scaled), FORTBO_ROVER_POINTS - 4)
        local = scaled - real(segment, dp)

        ! Uniform cubic B-spline basis.
        basis(1) = ((1.0_dp - local)**3)/6.0_dp
        basis(2) = (3.0_dp*local**3 - 6.0_dp*local**2 + 4.0_dp)/6.0_dp
        basis(3) = (-3.0_dp*local**3 + 3.0_dp*local**2 + 3.0_dp*local + 1.0_dp) &
            /6.0_dp
        basis(4) = (local**3)/6.0_dp

        point = 0.0_dp
        do k = 1, 4
            index = segment + k
            point = point + basis(k)*control(:, index)
        end do
    end subroutine spline_point

    pure logical function inside_obstacle(point) result(hit)
        real(dp), intent(in) :: point(2)
        integer :: k

        hit = .false.
        do k = 1, size(OBSTACLES, 2)
            if (point(1) >= OBSTACLES(1, k) .and. point(1) <= OBSTACLES(3, k) &
                .and. point(2) >= OBSTACLES(2, k) &
                .and. point(2) <= OBSTACLES(4, k)) then
                hit = .true.
                return
            end if
        end do
    end function inside_obstacle

    !! Number of sampled trajectory points inside an obstacle.
    subroutine rover_collisions(self, x, count, status)
        class(fortbo_rover_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        integer, intent(out) :: count
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: control(2, FORTBO_ROVER_POINTS), point(2)
        integer :: k

        count = 0
        if (size(x) /= FORTBO_ROVER_DIMENSION) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo rover: a trajectory has sixty coordinates")
            return
        end if
        control = reshape(x, [2, FORTBO_ROVER_POINTS])

        do k = 0, FORTBO_ROVER_SAMPLES
            call spline_point(control, real(k, dp)/real(FORTBO_ROVER_SAMPLES, dp), &
                point)
            if (inside_obstacle(point)) count = count + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rover_collisions

    !! The objective, negated for minimization.
    subroutine rover_value(self, x, value, status)
        class(fortbo_rover_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: control(2, FORTBO_ROVER_POINTS)
        real(dp) :: reward, endpoint
        integer :: count

        value = 0.0_dp
        call self%collisions(x, count, status)
        if (status%code /= FORTNUM_OK) return
        control = reshape(x, [2, FORTBO_ROVER_POINTS])

        ! L1 distances of the first and last control points from the desired
        ! start and goal, exactly as the reward states.
        endpoint = sum(abs(control(:, 1) - self%start)) &
            + sum(abs(control(:, FORTBO_ROVER_POINTS) - self%goal))

        reward = -FORTBO_ROVER_COLLISION_COST*real(count, dp) &
            - FORTBO_ROVER_ENDPOINT_WEIGHT*endpoint &
            + FORTBO_ROVER_OFFSET
        value = -reward
        call status_set(status, FORTNUM_OK, "")
    end subroutine rover_value

    !! Refused, by construction rather than by omission.
    !!
    !! The collision term is piecewise constant: its derivative is zero almost
    !! everywhere and undefined on the obstacle boundaries. A finite difference
    !! would return zero across most of the domain and something enormous at a
    !! boundary crossing, neither of which is a gradient. A policy that wants
    !! derivative information must use a different fixture, not difference
    !! through a step.
    subroutine rover_gradient(self, x, gradient, status)
        class(fortbo_rover_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "fortbo rover: the collision term is piecewise constant, so no "// &
            "gradient exists")
    end subroutine rover_gradient

end module fortbo_rover
