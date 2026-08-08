module placement_stubs
    !! Minimal surrogates for the placement rule.
    !!
    !! They exist in a module rather than inside the test program because
    !! type-bound procedures must be module procedures. Deliberately trivial:
    !! the rule under test reads capabilities and nothing else, so a stub that
    !! declares the right capabilities exercises it exactly as well as a real
    !! GP would, and without dragging a surrogate's numerics into a test about
    !! placement.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_MOMENT_GRADIENT, FORTBO_CAP_MOMENT_HESSIAN
    implicit none
    private

    public :: value_only_t
    public :: differentiable_t

    !! A value-only surrogate: moments and nothing else.
    type, public, extends(fortbo_posterior_t) :: value_only_t
    contains
        procedure :: n_inputs => value_only_inputs
        procedure :: capabilities => value_only_caps
        procedure :: moments => value_only_moments
    end type value_only_t

    !! A surrogate whose moments are differentiable, so a graph over it carries
    !! FortAD work.
    type, public, extends(fortbo_posterior_t) :: differentiable_t
    contains
        procedure :: n_inputs => differentiable_inputs
        procedure :: capabilities => differentiable_caps
        procedure :: moments => differentiable_moments
    end type differentiable_t

contains

    pure integer function value_only_inputs(self) result(n)
        class(value_only_t), intent(in) :: self
        n = 2
    end function value_only_inputs

    pure integer function value_only_caps(self) result(caps)
        class(value_only_t), intent(in) :: self
        caps = FORTBO_CAP_MOMENTS
    end function value_only_caps

    subroutine value_only_moments(self, points, mean, variance, status)
        class(value_only_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        mean = 0.0_dp
        variance = 1.0_dp
        status%code = FORTNUM_OK
    end subroutine value_only_moments

    pure integer function differentiable_inputs(self) result(n)
        class(differentiable_t), intent(in) :: self
        n = 2
    end function differentiable_inputs

    pure integer function differentiable_caps(self) result(caps)
        class(differentiable_t), intent(in) :: self
        caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_MOMENT_GRADIENT &
            + FORTBO_CAP_MOMENT_HESSIAN
    end function differentiable_caps

    subroutine differentiable_moments(self, points, mean, variance, status)
        class(differentiable_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        mean = 0.0_dp
        variance = 1.0_dp
        status%code = FORTNUM_OK
    end subroutine differentiable_moments

end module placement_stubs
