module fortbo_posterior
    !! Versioned, model-agnostic posterior contract (ROADMAP BO0).
    !!
    !! A surrogate reaches FortBO only through this contract. The contract is
    !! deliberately partial: a model declares which operations it can perform
    !! through `capabilities`, and every operation it does not declare returns a
    !! typed refusal naming the missing operation rather than a silent
    !! approximation. An acquisition that needs joint samples must therefore
    !! fail loudly against a mean/variance-only surrogate instead of quietly
    !! degrading to an independent-marginal fantasy.
    !!
    !! Sign and shape conventions, fixed for the whole package:
    !!   * points are `(n_points, n_inputs)`, one query per row
    !!   * `mean`/`variance` are `(n_points)`; variance is the marginal
    !!     predictive variance including observation noise only when the model
    !!     says so through `FORTBO_CAP_NOISY_MOMENTS`
    !!   * `covariance` is the `(n_points, n_points)` joint predictive
    !!     covariance, symmetric and positive semidefinite
    !!   * sample arrays are `(n_points, n_samples)`
    !!
    !! FortBO minimizes. A model that natively maximizes is adapted at its own
    !! boundary, not here.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_DOMAIN_ERROR
    implicit none
    private

    !! Contract version. Bumped whenever an existing operation changes meaning,
    !! shape, or sign convention; adding a capability bit alone does not bump it.
    integer, parameter, public :: FORTBO_POSTERIOR_CONTRACT_VERSION = 1

    ! Capability bits. A model reports the inclusive-or of what it implements.
    integer, parameter, public :: FORTBO_CAP_NONE = 0
    integer, parameter, public :: FORTBO_CAP_MOMENTS = 1
    integer, parameter, public :: FORTBO_CAP_COVARIANCE = 2
    integer, parameter, public :: FORTBO_CAP_JOINT_SAMPLE = 4
    integer, parameter, public :: FORTBO_CAP_REPARAM_SAMPLE = 8
    integer, parameter, public :: FORTBO_CAP_LOG_DENSITY = 16
    integer, parameter, public :: FORTBO_CAP_MOMENT_GRADIENT = 32
    integer, parameter, public :: FORTBO_CAP_MOMENT_HESSIAN = 64
    integer, parameter, public :: FORTBO_CAP_NOISY_MOMENTS = 128

    public :: fortbo_posterior_t
    public :: fortbo_capability_name

    type, abstract, public :: fortbo_posterior_t
        !! Contract version this implementation was written against. A consumer
        !! that finds an unexpected version refuses rather than guessing.
        integer :: contract_version = FORTBO_POSTERIOR_CONTRACT_VERSION
    contains
        procedure(fortbo_n_inputs_interface), deferred, public :: n_inputs
        procedure(fortbo_capabilities_interface), deferred, public :: capabilities
        procedure, public :: supports => posterior_supports
        procedure, public :: moments => posterior_moments_refuse
        procedure, public :: covariance => posterior_covariance_refuse
        procedure, public :: joint_sample => posterior_joint_sample_refuse
        procedure, public :: reparam_sample => posterior_reparam_sample_refuse
        procedure, public :: log_density => posterior_log_density_refuse
        procedure, public :: moment_gradient => posterior_moment_gradient_refuse
        procedure, public :: moment_hessian => posterior_moment_hessian_refuse
    end type fortbo_posterior_t

    abstract interface
        pure integer function fortbo_n_inputs_interface(self)
            import :: fortbo_posterior_t
            class(fortbo_posterior_t), intent(in) :: self
        end function fortbo_n_inputs_interface

        pure integer function fortbo_capabilities_interface(self)
            import :: fortbo_posterior_t
            class(fortbo_posterior_t), intent(in) :: self
        end function fortbo_capabilities_interface
    end interface

contains

    !! True when every bit in `capability` is declared by the model. Callers
    !! test before calling so a refusal is a programming error, not control flow.
    pure logical function posterior_supports(self, capability) result(ok)
        class(fortbo_posterior_t), intent(in) :: self
        integer, intent(in) :: capability

        ok = iand(self%capabilities(), capability) == capability
    end function posterior_supports

    !! Human-readable name for a single capability bit, used in refusal
    !! messages so a failure says what was missing, not merely that it failed.
    pure function fortbo_capability_name(capability) result(name)
        integer, intent(in) :: capability
        character(len=:), allocatable :: name

        select case (capability)
        case (FORTBO_CAP_MOMENTS)
            name = "moments"
        case (FORTBO_CAP_COVARIANCE)
            name = "covariance"
        case (FORTBO_CAP_JOINT_SAMPLE)
            name = "joint_sample"
        case (FORTBO_CAP_REPARAM_SAMPLE)
            name = "reparam_sample"
        case (FORTBO_CAP_LOG_DENSITY)
            name = "log_density"
        case (FORTBO_CAP_MOMENT_GRADIENT)
            name = "moment_gradient"
        case (FORTBO_CAP_MOMENT_HESSIAN)
            name = "moment_hessian"
        case (FORTBO_CAP_NOISY_MOMENTS)
            name = "noisy_moments"
        case default
            name = "unknown"
        end select
    end function fortbo_capability_name

    !! Marginal predictive mean and variance at each query row.
    subroutine posterior_moments_refuse(self, points, mean, variance, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        mean = 0.0_dp
        variance = 0.0_dp
        call refuse(status, "moments", size(points, 1))
    end subroutine posterior_moments_refuse

    !! Joint predictive covariance over the query rows.
    subroutine posterior_covariance_refuse(self, points, covariance, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        covariance = 0.0_dp
        call refuse(status, "covariance", size(points, 1))
    end subroutine posterior_covariance_refuse

    !! Joint posterior draws. `samples(i, s)` is draw `s` at query row `i`;
    !! draws within one call share the correlation structure of `covariance`.
    subroutine posterior_joint_sample_refuse(self, points, generator, samples, status)
        use fortnum_rng, only: rng_t
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status

        samples = 0.0_dp
        call refuse(status, "joint_sample", size(points, 1))
    end subroutine posterior_joint_sample_refuse

    !! Reparameterized draws from caller-supplied standard normal base samples.
    !! This is the differentiable sampling path: with `base` held fixed the map
    !! from `points` to `samples` is deterministic and has a derivative, which
    !! is what Monte Carlo acquisitions differentiate through. Common random
    !! numbers are obtained by reusing one `base` across candidate sets.
    subroutine posterior_reparam_sample_refuse(self, points, base, samples, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: base(:, :)
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status

        samples = 0.0_dp
        call refuse(status, "reparam_sample", size(points, 1))
    end subroutine posterior_reparam_sample_refuse

    !! Predictive log density of `values` under the joint posterior at `points`.
    subroutine posterior_log_density_refuse(self, points, values, log_density, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: values(:)
        real(dp), intent(out) :: log_density
        type(fortnum_status_t), intent(out) :: status

        log_density = 0.0_dp
        call refuse(status, "log_density", size(points, 1))
    end subroutine posterior_log_density_refuse

    !! Gradients of the marginal mean and standard deviation with respect to the
    !! query point. `mean_gradient(i, j)` is d mean(i) / d points(i, j); the
    !! marginals depend only on their own row, so no cross-row block exists.
    subroutine posterior_moment_gradient_refuse(self, points, mean_gradient, &
                                                sd_gradient, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean_gradient(:, :)
        real(dp), intent(out) :: sd_gradient(:, :)
        type(fortnum_status_t), intent(out) :: status

        mean_gradient = 0.0_dp
        sd_gradient = 0.0_dp
        call refuse(status, "moment_gradient", size(points, 1))
    end subroutine posterior_moment_gradient_refuse

    !! Hessians of the marginal mean and standard deviation at a single query
    !! point. DTuRBO's local quadratic model consumes exactly this pair.
    subroutine posterior_moment_hessian_refuse(self, point, mean_hessian, &
                                               sd_hessian, status)
        class(fortbo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: mean_hessian(:, :)
        real(dp), intent(out) :: sd_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status

        mean_hessian = 0.0_dp
        sd_hessian = 0.0_dp
        call refuse(status, "moment_hessian", size(point))
    end subroutine posterior_moment_hessian_refuse

    !! Typed refusal shared by every undeclared operation.
    pure subroutine refuse(status, operation, n_points)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in) :: operation
        integer, intent(in) :: n_points

        if (n_points < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo posterior: negative query count")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "fortbo posterior: surrogate does not implement "//operation)
    end subroutine refuse

end module fortbo_posterior
