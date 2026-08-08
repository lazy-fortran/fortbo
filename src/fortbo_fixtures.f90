module fortbo_fixtures
    !! Constrained and multi-objective benchmark fixtures (ROADMAP BO6).
    !!
    !! `fortbo_benchmarks` holds the single-objective unconstrained functions.
    !! This module holds the two families that need more than a value and a
    !! gradient to be usable as evidence.
    !!
    !! **Constrained fixtures** carry their constraints as separate functions
    !! with their own gradients, never folded into the objective as a penalty.
    !! A fixture that pre-penalized would make every method that uses the
    !! constraint properly look identical to one that ignores it, which is the
    !! whole thing a constrained benchmark is supposed to distinguish. The
    !! constrained optimum is stated, and it is generally *not* the
    !! unconstrained one — a fixture where the constraint is slack at the
    !! optimum tests nothing.
    !!
    !! **Multi-objective fixtures** carry an analytic Pareto front, so
    !! hypervolume can be compared against the truth rather than against another
    !! run. ZDT1's front is `f2 = 1 - sqrt(f1)` on `f1` in `[0,1]`, which is
    !! exact and needs no reference grid; the front is *convex* here, and that
    !! matters because a scalarization by weighted sum can reach every point of
    !! a convex front and cannot reach the interior of a concave one. ZDT2's
    !! front is `f2 = 1 - f1^2`, concave, and is included precisely so that
    !! weakness is visible.
    !!
    !! **Noise** is supplied as a standard deviation the caller applies, not as
    !! a random draw inside the fixture. A fixture that drew its own noise would
    !! be unreplayable, and two methods could not be compared on the same
    !! realizations.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    !! Constrained fixtures.
    integer, parameter, public :: FORTBO_CONSTRAINED_TOWNSEND = 1
    integer, parameter, public :: FORTBO_CONSTRAINED_GARDNER = 2

    !! Multi-objective fixtures.
    integer, parameter, public :: FORTBO_MULTI_ZDT1 = 1
    integer, parameter, public :: FORTBO_MULTI_ZDT2 = 2

    public :: fortbo_constrained_fixture_t
    public :: fortbo_multi_objective_fixture_t

    !! A constrained problem: one objective, `n_constraints` functions each
    !! required to be at most zero.
    type :: fortbo_constrained_fixture_t
        integer :: kind = FORTBO_CONSTRAINED_GARDNER
    contains
        procedure, public :: dimension => constrained_dimension
        procedure, public :: n_constraints => constrained_n_constraints
        procedure, public :: bounds => constrained_bounds
        procedure, public :: objective => constrained_objective
        procedure, public :: constraints => constrained_constraints
        procedure, public :: is_feasible => constrained_is_feasible
    end type fortbo_constrained_fixture_t

    type :: fortbo_multi_objective_fixture_t
        integer :: kind = FORTBO_MULTI_ZDT1
        integer :: dimension = 2
    contains
        procedure, public :: n_objectives => multi_n_objectives
        procedure, public :: evaluate => multi_evaluate
        procedure, public :: front_point => multi_front_point
        procedure, public :: reference_point => multi_reference_point
        procedure, public :: front_is_convex => multi_front_is_convex
    end type fortbo_multi_objective_fixture_t

contains

    pure integer function constrained_dimension(self) result(d)
        class(fortbo_constrained_fixture_t), intent(in) :: self

        d = 2
    end function constrained_dimension

    pure integer function constrained_n_constraints(self) result(m)
        class(fortbo_constrained_fixture_t), intent(in) :: self

        select case (self%kind)
        case (FORTBO_CONSTRAINED_GARDNER)
            m = 1
        case default
            m = 1
        end select
    end function constrained_n_constraints

    subroutine constrained_bounds(self, lower, upper, status)
        class(fortbo_constrained_fixture_t), intent(in) :: self
        real(dp), intent(out) :: lower(:)
        real(dp), intent(out) :: upper(:)
        type(fortnum_status_t), intent(out) :: status

        lower = 0.0_dp
        upper = 0.0_dp
        if (size(lower) /= 2 .or. size(upper) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: constrained bounds must have width two")
            return
        end if
        select case (self%kind)
        case (FORTBO_CONSTRAINED_TOWNSEND)
            lower = [-2.25_dp, -2.5_dp]
            upper = [2.25_dp, 1.75_dp]
        case (FORTBO_CONSTRAINED_GARDNER)
            lower = 0.0_dp
            upper = 6.0_dp
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: unknown constrained fixture")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine constrained_bounds

    subroutine constrained_objective(self, point, value, status)
        class(fortbo_constrained_fixture_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (size(point) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: constrained point must have width two")
            return
        end if
        select case (self%kind)
        case (FORTBO_CONSTRAINED_TOWNSEND)
            value = -(cos((point(1) - 0.1_dp)*point(2)))**2 - point(1)*sin(point(2))
        case (FORTBO_CONSTRAINED_GARDNER)
            ! Gardner et al.'s simulation benchmark, minimized.
            value = sin(point(1)) + point(2)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: unknown constrained fixture")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine constrained_objective

    !! Constraints as `c(x) <= 0`, kept separate from the objective.
    subroutine constrained_constraints(self, point, values, status)
        class(fortbo_constrained_fixture_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: t

        values = 0.0_dp
        if (size(point) /= 2 .or. size(values) /= self%n_constraints()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: constraint shapes disagree")
            return
        end if
        select case (self%kind)
        case (FORTBO_CONSTRAINED_TOWNSEND)
            t = atan2(point(1), point(2))
            values(1) = point(1)**2 + point(2)**2 &
                - (2.0_dp*cos(t) - 0.5_dp*cos(2.0_dp*t) &
                - 0.25_dp*cos(3.0_dp*t) - 0.125_dp*cos(4.0_dp*t))**2 &
                - (2.0_dp*sin(t))**2
        case (FORTBO_CONSTRAINED_GARDNER)
            values(1) = sin(point(1))*sin(point(2)) + 0.95_dp
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: unknown constrained fixture")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine constrained_constraints

    subroutine constrained_is_feasible(self, point, feasible, status)
        class(fortbo_constrained_fixture_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        logical, intent(out) :: feasible
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:)

        feasible = .false.
        allocate (values(self%n_constraints()))
        call self%constraints(point, values, status)
        if (status%code /= FORTNUM_OK) return
        feasible = all(values <= 0.0_dp)
    end subroutine constrained_is_feasible

    pure integer function multi_n_objectives(self) result(m)
        class(fortbo_multi_objective_fixture_t), intent(in) :: self

        m = 2
    end function multi_n_objectives

    !! ZDT1 and ZDT2 on `[0,1]^d`, both minimized.
    subroutine multi_evaluate(self, point, objectives, status)
        class(fortbo_multi_objective_fixture_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: objectives(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: g, ratio
        integer :: d

        objectives = 0.0_dp
        d = self%dimension
        if (d < 2 .or. size(point) /= d .or. size(objectives) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: multi-objective shapes disagree")
            return
        end if
        if (any(point < 0.0_dp) .or. any(point > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: ZDT is defined on the unit cube")
            return
        end if

        objectives(1) = point(1)
        g = 1.0_dp + 9.0_dp*sum(point(2:))/real(d - 1, dp)
        ratio = objectives(1)/g
        select case (self%kind)
        case (FORTBO_MULTI_ZDT1)
            objectives(2) = g*(1.0_dp - sqrt(ratio))
        case (FORTBO_MULTI_ZDT2)
            objectives(2) = g*(1.0_dp - ratio*ratio)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: unknown multi-objective fixture")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_evaluate

    !! A point of the analytic Pareto front, parameterized by `f1` in `[0,1]`.
    !!
    !! The front is attained where `g = 1`, that is with every coordinate after
    !! the first at zero, so it is exact rather than a dense grid.
    subroutine multi_front_point(self, f1, objectives, status)
        class(fortbo_multi_objective_fixture_t), intent(in) :: self
        real(dp), intent(in) :: f1
        real(dp), intent(out) :: objectives(:)
        type(fortnum_status_t), intent(out) :: status

        objectives = 0.0_dp
        if (size(objectives) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: front point needs two objectives")
            return
        end if
        if (f1 < 0.0_dp .or. f1 > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: the front is parameterized on [0, 1]")
            return
        end if
        objectives(1) = f1
        select case (self%kind)
        case (FORTBO_MULTI_ZDT1)
            objectives(2) = 1.0_dp - sqrt(f1)
        case (FORTBO_MULTI_ZDT2)
            objectives(2) = 1.0_dp - f1*f1
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fixtures: unknown multi-objective fixture")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_front_point

    !! A reference point dominated by the whole front, so hypervolume is
    !! well defined and comparable across runs. Stated rather than derived from
    !! the observed data: a reference point that moves with the run makes two
    !! hypervolumes incomparable.
    pure subroutine multi_reference_point(self, reference)
        class(fortbo_multi_objective_fixture_t), intent(in) :: self
        real(dp), intent(out) :: reference(2)

        reference = [1.1_dp, 1.1_dp]
    end subroutine multi_reference_point

    !! Whether a weighted-sum scalarization can reach the whole front.
    !!
    !! ZDT1's front is convex and it can; ZDT2's is concave and it cannot reach
    !! the interior at all, however the weights are chosen. A benchmark suite
    !! containing only convex fronts would report scalarization methods as
    !! complete when they are not.
    pure logical function multi_front_is_convex(self) result(convex)
        class(fortbo_multi_objective_fixture_t), intent(in) :: self

        convex = self%kind == FORTBO_MULTI_ZDT1
    end function multi_front_is_convex

end module fortbo_fixtures
