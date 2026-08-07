module fortbo
    !! Public package boundary for differentiable Bayesian optimization.
    use fortnum_kinds, only: dp
    use fortbo_posterior, only: fortbo_posterior_t, fortbo_capability_name, &
        FORTBO_POSTERIOR_CONTRACT_VERSION, FORTBO_CAP_NONE, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_REPARAM_SAMPLE, &
        FORTBO_CAP_LOG_DENSITY, FORTBO_CAP_MOMENT_GRADIENT, &
        FORTBO_CAP_MOMENT_HESSIAN, FORTBO_CAP_NOISY_MOMENTS
    implicit none
    private

    integer, parameter, public :: FORTBO_VERSION_MAJOR = 0
    integer, parameter, public :: FORTBO_VERSION_MINOR = 1

    public :: fortbo_posterior_t
    public :: fortbo_capability_name
    public :: FORTBO_POSTERIOR_CONTRACT_VERSION
    public :: FORTBO_CAP_NONE
    public :: FORTBO_CAP_MOMENTS
    public :: FORTBO_CAP_COVARIANCE
    public :: FORTBO_CAP_JOINT_SAMPLE
    public :: FORTBO_CAP_REPARAM_SAMPLE
    public :: FORTBO_CAP_LOG_DENSITY
    public :: FORTBO_CAP_MOMENT_GRADIENT
    public :: FORTBO_CAP_MOMENT_HESSIAN
    public :: FORTBO_CAP_NOISY_MOMENTS

end module fortbo
