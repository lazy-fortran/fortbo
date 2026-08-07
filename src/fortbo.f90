module fortbo
    !! Public package boundary for differentiable Bayesian optimization.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t
    implicit none
    private

    integer, parameter, public :: FORTBO_VERSION_MAJOR = 0
    integer, parameter, public :: FORTBO_VERSION_MINOR = 1

    public :: fortbo_posterior_t

    type, abstract, public :: fortbo_posterior_t
    contains
        procedure(fortbo_sample_interface), deferred, public :: sample
        procedure(fortbo_mean_variance_interface), deferred, public :: mean_variance
    end type fortbo_posterior_t

    abstract interface
        subroutine fortbo_sample_interface(self, points, samples, status)
            import :: dp, fortbo_posterior_t
            import :: fortnum_status_t
            class(fortbo_posterior_t), intent(in) :: self
            real(dp), intent(in) :: points(:, :)
            real(dp), intent(out) :: samples(:, :, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine fortbo_sample_interface

        subroutine fortbo_mean_variance_interface(self, points, mean, variance, status)
            import :: dp, fortbo_posterior_t
            import :: fortnum_status_t
            class(fortbo_posterior_t), intent(in) :: self
            real(dp), intent(in) :: points(:, :)
            real(dp), intent(out) :: mean(:)
            real(dp), intent(out) :: variance(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine fortbo_mean_variance_interface
    end interface

end module fortbo
