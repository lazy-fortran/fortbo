module fortbo_test_posteriors
    !! Concrete posteriors used to exercise the BO0 contract.
    !!
    !! `demo_posterior_t` is a fixed Gaussian process prior with a squared
    !! exponential kernel and a linear mean. It has no training data, so every
    !! moment is available in closed form and a test can state the expected
    !! answer without re-running the implementation. `moments_only_posterior_t`
    !! declares nothing beyond marginal moments and exists to show that the
    !! undeclared operations refuse.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_normal
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_REPARAM_SAMPLE, &
        FORTBO_CAP_LOG_DENSITY
    implicit none
    private

    real(dp), parameter, public :: DEMO_SIGNAL = 1.75_dp
    real(dp), parameter, public :: DEMO_LENGTHSCALE = 0.8_dp
    real(dp), parameter, public :: DEMO_JITTER = 1.0e-10_dp
    real(dp), parameter :: LOG_TWO_PI = log(8.0_dp*atan(1.0_dp))

    public :: demo_posterior_t
    public :: moments_only_posterior_t
    public :: demo_mean
    public :: demo_kernel

    type, extends(fortbo_posterior_t), public :: demo_posterior_t
        integer :: dimension = 1
    contains
        procedure, public :: n_inputs => demo_n_inputs
        procedure, public :: capabilities => demo_capabilities
        procedure, public :: moments => demo_moments
        procedure, public :: covariance => demo_covariance
        procedure, public :: joint_sample => demo_joint_sample
        procedure, public :: reparam_sample => demo_reparam_sample
        procedure, public :: log_density => demo_log_density
    end type demo_posterior_t

    type, extends(fortbo_posterior_t), public :: moments_only_posterior_t
        integer :: dimension = 1
    contains
        procedure, public :: n_inputs => moments_only_n_inputs
        procedure, public :: capabilities => moments_only_capabilities
        procedure, public :: moments => moments_only_moments
    end type moments_only_posterior_t

contains

    !! Prior mean, stated once so tests and implementation cannot drift apart
    !! without the drift being visible at this single definition.
    pure real(dp) function demo_mean(point) result(value)
        real(dp), intent(in) :: point(:)

        value = sum(point)
    end function demo_mean

    !! Squared exponential covariance.
    pure real(dp) function demo_kernel(left, right) result(value)
        real(dp), intent(in) :: left(:)
        real(dp), intent(in) :: right(:)

        value = DEMO_SIGNAL*exp(-0.5_dp*sum((left - right)**2)/DEMO_LENGTHSCALE**2)
    end function demo_kernel

    pure integer function demo_n_inputs(self) result(n)
        class(demo_posterior_t), intent(in) :: self

        n = self%dimension
    end function demo_n_inputs

    pure integer function demo_capabilities(self) result(caps)
        class(demo_posterior_t), intent(in) :: self

        caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_COVARIANCE + FORTBO_CAP_JOINT_SAMPLE &
               + FORTBO_CAP_REPARAM_SAMPLE + FORTBO_CAP_LOG_DENSITY
        if (self%dimension < 1) caps = 0
    end function demo_capabilities

    subroutine demo_moments(self, points, mean, variance, status)
        class(demo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        call check_points(self%dimension, points, size(mean), status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            mean(i) = demo_mean(points(i, :))
            variance(i) = demo_kernel(points(i, :), points(i, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine demo_moments

    subroutine demo_covariance(self, points, covariance, status)
        class(demo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n

        n = size(points, 1)
        call check_points(self%dimension, points, size(covariance, 1), status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, n
            do i = 1, n
                covariance(i, j) = demo_kernel(points(i, :), points(j, :))
            end do
            covariance(j, j) = covariance(j, j) + DEMO_JITTER
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine demo_covariance

    subroutine demo_joint_sample(self, points, generator, samples, status)
        class(demo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: base(:, :)
        integer :: i, s

        allocate (base(size(points, 1), size(samples, 2)))
        do s = 1, size(base, 2)
            do i = 1, size(base, 1)
                call rng_normal(generator, base(i, s))
            end do
        end do
        call demo_reparam_sample(self, points, base, samples, status)
    end subroutine demo_joint_sample

    subroutine demo_reparam_sample(self, points, base, samples, status)
        class(demo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: base(:, :)
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: covariance(:, :), mean(:), variance(:)
        integer :: n, s

        n = size(points, 1)
        allocate (covariance(n, n), mean(n), variance(n))
        call demo_covariance(self, points, covariance, status)
        if (status%code /= FORTNUM_OK) return
        call demo_moments(self, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, size(samples, 2)
            samples(:, s) = mean + matmul(factorization%lower, base(:, s))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine demo_reparam_sample

    subroutine demo_log_density(self, points, values, log_density, status)
        class(demo_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: values(:)
        real(dp), intent(out) :: log_density
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: covariance(:, :), mean(:), variance(:), residual(:)
        real(dp), allocatable :: solved(:)
        real(dp) :: log_det
        integer :: n

        n = size(points, 1)
        log_density = 0.0_dp
        allocate (covariance(n, n), mean(n), variance(n), residual(n), solved(n))
        call demo_covariance(self, points, covariance, status)
        if (status%code /= FORTNUM_OK) return
        call demo_moments(self, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        residual = values - mean
        solved = residual
        call factorization%solve(solved, status)
        if (status%code /= FORTNUM_OK) return
        call factorization%log_determinant(log_det, status)
        if (status%code /= FORTNUM_OK) return
        log_density = -0.5_dp*dot_product(residual, solved) - 0.5_dp*log_det &
                      - 0.5_dp*real(n, dp)*LOG_TWO_PI
        call status_set(status, FORTNUM_OK, "")
    end subroutine demo_log_density

    pure integer function moments_only_n_inputs(self) result(n)
        class(moments_only_posterior_t), intent(in) :: self

        n = self%dimension
    end function moments_only_n_inputs

    pure integer function moments_only_capabilities(self) result(caps)
        class(moments_only_posterior_t), intent(in) :: self

        caps = FORTBO_CAP_MOMENTS
        if (self%dimension < 1) caps = 0
    end function moments_only_capabilities

    subroutine moments_only_moments(self, points, mean, variance, status)
        class(moments_only_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        call check_points(self%dimension, points, size(mean), status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            mean(i) = demo_mean(points(i, :))
            variance(i) = DEMO_SIGNAL
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine moments_only_moments

    pure subroutine check_points(dimension, points, n_out, status)
        integer, intent(in) :: dimension
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: n_out
        type(fortnum_status_t), intent(out) :: status

        if (size(points, 2) /= dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "demo posterior: query width does not match dimension")
            return
        end if
        if (n_out /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "demo posterior: output length does not match query count")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_points

end module fortbo_test_posteriors
