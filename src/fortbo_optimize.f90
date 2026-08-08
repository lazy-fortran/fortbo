module fortbo_optimize
    !! Bound-constrained acquisition optimization (ROADMAP BO3).
    !!
    !! The acquisition is *maximized*; FortOpt's L-BFGS-B *minimizes*. The
    !! negation happens here, once, at the boundary — which is why every
    !! acquisition in this package can state its own sign convention plainly and
    !! never think about the optimizer again.
    !!
    !! Multistart is not optional garnish. Acquisition surfaces are routinely
    !! multimodal and nearly flat away from the data: expected improvement is
    !! numerically zero over most of the domain once a few points are known, so
    !! a single local run started in the wrong basin returns a point with no
    !! improvement at all and the whole iteration is wasted. Running from
    !! several starts and keeping the best is what makes the inner optimization
    !! reliable enough that the outer loop's regret means something.
    !!
    !! Derivative use here is independent of derivative *observations*. This
    !! optimizer needs the acquisition's gradient with respect to the query
    !! point, which comes from the posterior's moment gradients. A posterior
    !! that cannot supply them gets the derivative-free path — refused rather
    !! than differenced — and the caller chooses a sampling search instead.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_acquisition, only: fortbo_acquisition_t
    implicit none
    private

    public :: fortbo_optimize_acquisition
    public :: fortbo_search_acquisition
    public :: fortbo_acquisition_context_t

    !! Context handed to the FortOpt callback. Holds references, not copies:
    !! a surrogate can be large and the inner loop evaluates it thousands of
    !! times.
    type :: fortbo_acquisition_context_t
        class(fortbo_acquisition_t), pointer :: acquisition => null()
        class(fortbo_posterior_t), pointer :: posterior => null()
        integer :: evaluations = 0
        integer :: failures = 0
    end type fortbo_acquisition_context_t

contains

    !! Maximize `acquisition` over the box `[lower, upper]` from every row of
    !! `starts`, and return the best point found.
    !!
    !! Ties are broken by the earliest start index, so a replayed run with the
    !! same starts returns the same point even when two basins are numerically
    !! indistinguishable.
    subroutine fortbo_optimize_acquisition(acquisition, posterior, lower, upper, &
            starts, best_point, best_value, status, &
            options, converged_starts)
        class(fortbo_acquisition_t), intent(in), target :: acquisition
        class(fortbo_posterior_t), intent(in), target :: posterior
        real(dp), intent(in) :: lower(:)
        real(dp), intent(in) :: upper(:)
        real(dp), intent(in) :: starts(:, :)
        real(dp), intent(out) :: best_point(:)
        real(dp), intent(out) :: best_value
        type(fortnum_status_t), intent(out) :: status
        type(lbfgsb_options_t), intent(in), optional :: options
        integer, intent(out), optional :: converged_starts
        type(fortbo_acquisition_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: local_options
        type(lbfgsb_result_t) :: result
        type(fortnum_status_t) :: run_status
        real(dp), allocatable :: point(:), values(:), query(:, :)
        integer :: n_inputs, n_starts, s, succeeded
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(lbfgsb_options_t) :: lbfgsb_options_t_default

        n_inputs = size(lower)
        n_starts = size(starts, 1)
        best_value = -huge(1.0_dp)
        best_point = 0.0_dp
        if (present(converged_starts)) converged_starts = 0

        if (n_inputs < 1 .or. size(upper) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: bound widths do not match")
            return
        end if
        if (any(upper < lower)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: upper bound is below lower bound")
            return
        end if
        if (size(best_point) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: result width does not match the bounds")
            return
        end if
        if (n_starts < 1 .or. size(starts, 2) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: start shape does not match the bounds")
            return
        end if
        if (.not. posterior%supports(FORTBO_CAP_MOMENT_GRADIENT)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo optimize: gradient-based search needs a surrogate with "// &
                "moment_gradient; use a sampling search instead")
            return
        end if

        local_options = lbfgsb_options_t_default
        if (present(options)) local_options = options

        context%acquisition => acquisition
        context%posterior => posterior
        call objective%initialize_context(n_inputs, context, negated_acquisition, status)
        if (status%code /= FORTNUM_OK) return

        allocate (point(n_inputs), values(1), query(1, n_inputs))
        succeeded = 0
        do s = 1, n_starts
            point = min(max(starts(s, :), lower), upper)
            call optimizer%minimize(objective, point, lower, upper, local_options, &
                result, run_status)
            ! A start that fails to converge is not fatal. Acquisition surfaces
            ! are flat far from the data, so some starts legitimately stall;
            ! what matters is that at least one start produced a usable point.
            if (run_status%code /= FORTNUM_OK) cycle
            succeeded = succeeded + 1

            query(1, :) = point
            call acquisition%value(posterior, query, values, status)
            if (status%code /= FORTNUM_OK) return
            if (values(1) > best_value) then
                best_value = values(1)
                best_point = point
            end if
        end do

        if (present(converged_starts)) converged_starts = succeeded
        if (succeeded == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: no start converged; the acquisition surface may "// &
                "be flat over the whole box")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_optimize_acquisition

    !! Derivative-free acquisition search over a caller-supplied candidate set.
    !!
    !! This exists because `fortbo_optimize_acquisition` refuses a surrogate
    !! without `moment_gradient` and tells the caller to use a sampling search
    !! instead -- advice that was, until this routine, impossible to follow. A
    !! value-only GP is the common case, so the gradient-free path is not a
    !! fallback for exotic models; it is what most runs need.
    !!
    !! The candidates are the caller's, not generated here, and that is
    !! deliberate. Whoever is running a benchmark has to be able to state
    !! exactly which points were scored, and a routine that invented its own
    !! quasi-random set would make the search unreproducible from the outside
    !! for no gain.
    !!
    !! Ties go to the lowest row, so the result does not depend on the order a
    !! future vectorized implementation happens to reduce in.
    subroutine fortbo_search_acquisition(acquisition, posterior, lower, upper, &
            candidates, best_point, best_value, status)
        class(fortbo_acquisition_t), intent(in) :: acquisition
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: lower(:)
        real(dp), intent(in) :: upper(:)
        real(dp), intent(in) :: candidates(:, :)
        real(dp), intent(out) :: best_point(:)
        real(dp), intent(out) :: best_value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:), clamped(:, :)
        integer :: n_inputs, n_candidates, k, best_index

        n_inputs = size(lower)
        n_candidates = size(candidates, 1)
        best_value = -huge(1.0_dp)
        best_point = 0.0_dp

        if (n_inputs < 1 .or. size(upper) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo search: bounds do not agree")
            return
        end if
        if (any(upper < lower)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo search: upper bound below lower bound")
            return
        end if
        if (n_candidates < 1 .or. size(candidates, 2) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo search: candidate shape does not match the bounds")
            return
        end if
        if (size(best_point) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo search: the result has the wrong width")
            return
        end if

        ! Clamped rather than refused: a candidate a hair outside the box is a
        ! rounding artefact of whatever generated it, and rejecting the whole
        ! set for one would be unhelpful. The returned point is inside the box
        ! either way, which is the property the caller depends on.
        allocate (clamped(n_candidates, n_inputs))
        do k = 1, n_candidates
            clamped(k, :) = min(max(candidates(k, :), lower), upper)
        end do

        allocate (values(n_candidates))
        call acquisition%value(posterior, clamped, values, status)
        if (status%code /= FORTNUM_OK) return

        best_index = 1
        do k = 2, n_candidates
            if (values(k) > values(best_index)) best_index = k
        end do
        best_value = values(best_index)
        best_point = clamped(best_index, :)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_search_acquisition

    !! FortOpt callback. Returns the negated acquisition and its negated
    !! gradient, because the optimizer descends and the acquisition is to be
    !! maximized.
    subroutine negated_acquisition(context, x, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: query(1, size(x)), values(1), gradients(1, size(x))

        value = 0.0_dp
        gradient = 0.0_dp
        select type (context)
            type is (fortbo_acquisition_context_t)
            context%evaluations = context%evaluations + 1
            query(1, :) = x
            call context%acquisition%value(context%posterior, query, values, status)
            if (status%code /= FORTNUM_OK) then
                context%failures = context%failures + 1
                return
            end if
            call context%acquisition%value_gradient(context%posterior, query, &
                gradients, status)
            if (status%code /= FORTNUM_OK) then
                context%failures = context%failures + 1
                return
            end if
            value = -values(1)
            gradient = -gradients(1, :)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo optimize: unexpected callback context")
        end select
    end subroutine negated_acquisition

end module fortbo_optimize
