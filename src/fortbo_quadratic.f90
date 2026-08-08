module fortbo_quadratic
    !! Bound-constrained quadratic subproblem (ROADMAP BO3T, DTuRBO mode 2).
    !!
    !! Minimize the local model
    !!
    !!     m(s) = g'.s + 0.5 s'.H.s
    !!
    !! over a box. This is the step computation of a classical trust-region
    !! method, and in DTuRBO the box is the trust region and `g`, `H` come from
    !! the posterior rather than from the objective itself.
    !!
    !! **The Hessian is not made positive definite.** An indefinite `H` is
    !! information, not an inconvenience: it says the model sees negative
    !! curvature, and the right response is to move along it to the boundary,
    !! which is precisely what a bound-constrained solve does. Projecting `H`
    !! onto the positive-definite cone — adding a multiple of the identity until
    !! Cholesky succeeds — would replace that direction with a fabricated one
    !! and quietly turn a trust-region method into a damped Newton method that
    !! ignores the very curvature it just measured. So indefiniteness is
    !! detected, reported through `indefinite`, and otherwise left alone.
    !!
    !! The box is what keeps a nonconvex model bounded below. Without it,
    !! minimizing an indefinite quadratic is meaningless.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    public :: fortbo_quadratic_model_t
    public :: fortbo_solve_quadratic_subproblem
    public :: fortbo_quadratic_value

    type :: fortbo_quadratic_model_t
        real(dp), allocatable :: gradient(:)
        real(dp), allocatable :: hessian(:, :)
    end type fortbo_quadratic_model_t

contains

    pure real(dp) function fortbo_quadratic_value(gradient, hessian, step) result(value)
        real(dp), intent(in) :: gradient(:)
        real(dp), intent(in) :: hessian(:, :)
        real(dp), intent(in) :: step(:)

        value = dot_product(gradient, step) &
            + 0.5_dp*dot_product(step, matmul(hessian, step))
    end function fortbo_quadratic_value

    !! Minimize the quadratic model over `[lower, upper]`, starting from
    !! `start`. The step is returned, not the point: the caller adds it to the
    !! region center.
    !!
    !! `indefinite` reports whether the Hessian failed a Cholesky test. It is a
    !! diagnostic for the caller's trust-region logic, not a switch that changes
    !! what this routine solves.
    subroutine fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            start, step, model_value, status, &
            indefinite, options)
        real(dp), intent(in) :: gradient(:)
        real(dp), intent(in) :: hessian(:, :)
        real(dp), intent(in) :: lower(:)
        real(dp), intent(in) :: upper(:)
        real(dp), intent(in) :: start(:)
        real(dp), intent(out) :: step(:)
        real(dp), intent(out) :: model_value
        type(fortnum_status_t), intent(out) :: status
        logical, intent(out), optional :: indefinite
        type(lbfgsb_options_t), intent(in), optional :: options
        type(fortbo_quadratic_model_t), target :: model
        type(cholesky_factorization_t) :: factorization
        type(fortnum_status_t) :: factor_status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: local_options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: point(:)
        real(dp) :: asymmetry
        integer :: n
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(lbfgsb_options_t) :: lbfgsb_options_t_default

        n = size(gradient)
        step = 0.0_dp
        model_value = 0.0_dp
        if (present(indefinite)) indefinite = .false.

        if (n < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: empty problem")
            return
        end if
        if (size(hessian, 1) /= n .or. size(hessian, 2) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: hessian shape does not match")
            return
        end if
        if (size(lower) /= n .or. size(upper) /= n .or. size(start) /= n &
            .or. size(step) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: bound or step width does not match")
            return
        end if
        if (any(upper < lower)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: upper bound is below lower bound")
            return
        end if

        ! A nonsymmetric Hessian is a caller error, not something to symmetrize
        ! silently: it means the two triangles came from different derivations
        ! and at least one of them is wrong.
        asymmetry = maxval(abs(hessian - transpose(hessian)))
        if (asymmetry > 1.0e-8_dp*max(1.0_dp, maxval(abs(hessian)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: hessian is not symmetric")
            return
        end if

        if (present(indefinite)) then
            call factorization%factorize(hessian, factor_status)
            indefinite = factor_status%code /= FORTNUM_OK
        end if

        allocate (model%gradient(n), model%hessian(n, n))
        model%gradient = gradient
        model%hessian = hessian

        local_options = lbfgsb_options_t_default
        if (present(options)) local_options = options

        call objective%initialize_context(n, model, quadratic_objective, status)
        if (status%code /= FORTNUM_OK) return

        allocate (point(n))
        point = min(max(start, lower), upper)
        call optimizer%minimize(objective, point, lower, upper, local_options, result, &
            status)
        ! A nonconvex model can stop on a bound before the gradient test is
        ! satisfied; that is a legitimate trust-region step, so the point is
        ! kept and the caller decides from the model decrease.
        step = point
        model_value = fortbo_quadratic_value(gradient, hessian, step)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_solve_quadratic_subproblem

    subroutine quadratic_objective(context, x, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        gradient = 0.0_dp
        select type (context)
            type is (fortbo_quadratic_model_t)
            value = fortbo_quadratic_value(context%gradient, context%hessian, x)
            gradient = context%gradient + matmul(context%hessian, x)
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo quadratic: unexpected callback context")
        end select
    end subroutine quadratic_objective

end module fortbo_quadratic
