module fortbo_feasible
    !! Fixed choices and feasible-region handling for the candidate optimizer
    !! (ROADMAP BO3).
    !!
    !! Two ways to keep a candidate search inside a region, and the difference
    !! between them is not a matter of taste.
    !!
    !! **Fixed choices** pin a categorical variable to one level, or an integer
    !! to one value, for the whole search. This is a *reparameterization*: the
    !! coordinate leaves the search entirely, so no candidate can violate it and
    !! no penalty weight has to be chosen. Whenever a constraint can be
    !! expressed this way it should be, because a search over a smaller space is
    !! strictly easier than a search over a larger one with a penalty pushing it
    !! back out.
    !!
    !! **Penalties** handle what reparameterization cannot: constraints that
    !! couple coordinates, or that are only known through a surrogate. A penalty
    !! is added to the acquisition, and the two ways of doing that behave very
    !! differently:
    !!
    !!   * a *quadratic* penalty `rho * violation^2` is smooth, so a
    !!     gradient-based optimizer can follow it, but it never makes the
    !!     constraint binding — the optimum of the penalized problem sits
    !!     slightly outside the feasible region for any finite `rho`, and
    !!     shrinking the error means raising `rho` until the problem is
    !!     ill-conditioned;
    !!   * an *exact* penalty `rho * violation` is non-smooth at the boundary
    !!     but genuinely exact: above a finite threshold on `rho` the penalized
    !!     optimum *is* the constrained optimum. The kink is the price, and it
    !!     is why this module reports which penalty is in use rather than
    !!     picking one.
    !!
    !! Both are offered, and `fortbo_penalty_is_differentiable` says which the
    !! caller has, so a gradient-based optimizer is not handed a kink without
    !! being told.
    !!
    !! A penalty is never applied to an acquisition that can go negative. As in
    !! `fortbo_constrained`, adding a positive penalty to a negative value moves
    !! it toward zero, which for a minimization convention makes an infeasible
    !! point look *better*. Here the convention is that the score is minimized
    !! and the penalty is added, so the base score must be one where larger is
    !! worse.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortbo_space, only: fortbo_space_t, FORTBO_VAR_CATEGORICAL
    implicit none
    private

    public :: fortbo_fixed_choice_t
    public :: fortbo_apply_fixed_choices
    public :: fortbo_violation
    public :: fortbo_penalized_score
    public :: fortbo_penalty_is_differentiable
    public :: fortbo_penalty_name

    integer, parameter, public :: FORTBO_PENALTY_QUADRATIC = 1
    integer, parameter, public :: FORTBO_PENALTY_EXACT = 2

    !! A variable pinned to one decoded value for the whole search.
    type :: fortbo_fixed_choice_t
        integer :: variable = 0
        real(dp) :: value = 0.0_dp
    end type fortbo_fixed_choice_t

contains

    pure function fortbo_penalty_name(kind) result(name)
        integer, intent(in) :: kind
        character(len=:), allocatable :: name

        select case (kind)
        case (FORTBO_PENALTY_QUADRATIC)
            name = "quadratic"
        case (FORTBO_PENALTY_EXACT)
            name = "exact"
        case default
            name = "unknown"
        end select
    end function fortbo_penalty_name

    !! Whether the penalized score has a derivative everywhere.
    !!
    !! The quadratic penalty does; the exact one has a kink on the constraint
    !! boundary, which is precisely where an optimizer will spend its time. A
    !! caller that intends to use gradients has to know that before it starts,
    !! not discover it from a line search that will not converge.
    pure logical function fortbo_penalty_is_differentiable(kind) result(smooth)
        integer, intent(in) :: kind

        smooth = kind == FORTBO_PENALTY_QUADRATIC
    end function fortbo_penalty_is_differentiable

    !! Overwrite the pinned coordinates of a batch of unit-cube points.
    !!
    !! Applied *after* candidate generation rather than by masking during it, so
    !! that a generator which knows nothing about the constraint still produces
    !! usable candidates. The cost is that the pinned coordinate's variation is
    !! wasted; the benefit is that no generator needs changing, which matters
    !! because Sobol candidate generation is shared with TuRBO.
    subroutine fortbo_apply_fixed_choices(space, choices, points, status)
        type(fortbo_space_t), intent(in) :: space
        type(fortbo_fixed_choice_t), intent(in) :: choices(:)
        real(dp), intent(inout) :: points(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: decoded(:), encoded(:)
        integer :: i, c, v

        if (size(points, 2) /= space%n_coordinates()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: point width does not match the space")
            return
        end if
        do c = 1, size(choices)
            v = choices(c)%variable
            if (v < 1 .or. v > space%n_variables()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo feasible: fixed-choice variable is out of range")
                return
            end if
            if (space%variables(v)%kind == FORTBO_VAR_CATEGORICAL) then
                if (choices(c)%value < 1.0_dp .or. &
                    choices(c)%value > real(space%variables(v)%n_categories, dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "fortbo feasible: fixed category is out of range")
                    return
                end if
            else
                if (choices(c)%value < space%variables(v)%lower .or. &
                    choices(c)%value > space%variables(v)%upper) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "fortbo feasible: fixed value is outside the variable's bounds")
                    return
                end if
            end if
        end do

        allocate (decoded(space%n_variables()), encoded(space%n_coordinates()))
        do i = 1, size(points, 1)
            call space%from_unit(points(i, :), decoded, status)
            if (status%code /= FORTNUM_OK) return
            do c = 1, size(choices)
                decoded(choices(c)%variable) = choices(c)%value
            end do
            call space%to_unit(decoded, encoded, status)
            if (status%code /= FORTNUM_OK) return
            points(i, :) = encoded
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_apply_fixed_choices

    !! Total violation of `c_k(x) <= 0` constraints: `sum_k max(c_k, 0)`.
    !!
    !! Only positive parts count. Crediting a slack constraint against a
    !! violated one would let a point that badly breaks one constraint pass by
    !! comfortably satisfying another, which is not what feasibility means.
    pure subroutine fortbo_violation(constraints, total, status)
        !! `constraints(constraint, point)`.
        real(dp), intent(in) :: constraints(:, :)
        real(dp), intent(out) :: total(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, k

        total = 0.0_dp
        if (size(total) /= size(constraints, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: violation output does not match the batch")
            return
        end if
        do i = 1, size(constraints, 2)
            do k = 1, size(constraints, 1)
                total(i) = total(i) + max(constraints(k, i), 0.0_dp)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_violation

    !! `score + rho * penalty(violation)`, minimized.
    subroutine fortbo_penalized_score(base, violation, rho, kind, score, status)
        real(dp), intent(in) :: base(:)
        real(dp), intent(in) :: violation(:)
        real(dp), intent(in) :: rho
        integer, intent(in) :: kind
        real(dp), intent(out) :: score(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        score = 0.0_dp
        if (size(violation) /= size(base) .or. size(score) /= size(base)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: penalty shapes disagree")
            return
        end if
        if (rho < 0.0_dp) then
            ! A negative weight rewards violation.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: penalty weight must not be negative")
            return
        end if
        if (any(violation < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: violation must not be negative")
            return
        end if
        if (fortbo_penalty_name(kind) == "unknown") then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo feasible: unknown penalty kind")
            return
        end if

        do i = 1, size(base)
            select case (kind)
            case (FORTBO_PENALTY_QUADRATIC)
                score(i) = base(i) + rho*violation(i)**2
            case (FORTBO_PENALTY_EXACT)
                score(i) = base(i) + rho*violation(i)
            end select
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_penalized_score

end module fortbo_feasible
