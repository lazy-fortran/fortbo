module fortbo_ard_posterior
    !! Exact value-only Matern-5/2 ARD posterior for the Landreman replay.
    !!
    !! FortML's general kernel adapter currently exposes the isotropic Matern
    !! leaves used by the older FortBO paths.  The Landreman control instead
    !! fixes an ARD Matern-5/2 model.  This small adapter keeps that replay
    !! contract local to FortBO: it owns only dense value-only inference and
    !! deliberately does not claim gradients, joint draws, or hyperparameter
    !! fitting.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_NOISY_MOMENTS
    implicit none
    private

    public :: fortbo_ard_posterior_t
    public :: fortbo_fit_ard_posterior

    type, extends(fortbo_posterior_t) :: fortbo_ard_posterior_t
        integer :: dimension = 0
        real(dp), allocatable :: train_x(:, :)
        real(dp), allocatable :: alpha(:)
        real(dp) :: signal_variance = 1.0_dp
        real(dp) :: noise_variance = 1.0e-6_dp
        real(dp), allocatable :: lengthscales(:)
        type(cholesky_factorization_t) :: factorization
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => ard_posterior_n_inputs
        procedure, public :: capabilities => ard_posterior_capabilities
        procedure, public :: moments => ard_posterior_moments
    end type fortbo_ard_posterior_t

contains

    pure integer function ard_posterior_n_inputs(self) result(n)
        class(fortbo_ard_posterior_t), intent(in) :: self

        n = self%dimension
    end function ard_posterior_n_inputs

    pure integer function ard_posterior_capabilities(self) result(caps)
        class(fortbo_ard_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS
    end function ard_posterior_capabilities

    !! Fit a fixed-hyperparameter, exact dense value-only ARD posterior.
    !!
    !! The caller supplies the lengthscales because this adapter is a parity
    !! lane, not a hyperparameter optimizer.  The covariance and Cholesky
    !! equations are the same equations used by the independent NumPy oracle.
    subroutine fortbo_fit_ard_posterior(history, posterior, lengthscales, status, &
            signal_variance, noise_variance)
        type(fortbo_history_t), intent(in) :: history
        class(fortbo_posterior_t), intent(out), allocatable :: posterior
        real(dp), intent(in) :: lengthscales(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: signal_variance
        real(dp), intent(in), optional :: noise_variance
        type(fortbo_ard_posterior_t), allocatable :: fitted
        real(dp), allocatable :: train_x(:, :), train_y(:), covariance(:, :)
        real(dp) :: signal, noise
        integer :: n, d, i

        d = history%n_inputs
        if (d < 1 .or. size(lengthscales) /= d .or. &
                any(lengthscales <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: dimensions and lengthscales are invalid")
            return
        end if
        signal = 1.0_dp
        noise = 1.0e-6_dp
        if (present(signal_variance)) signal = signal_variance
        if (present(noise_variance)) noise = noise_variance
        if (signal <= 0.0_dp .or. noise <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: signal and noise must be positive")
            return
        end if

        call history%training_data(train_x, train_y, status)
        if (status%code /= FORTNUM_OK) return
        n = size(train_y)
        if (n < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: no usable observations")
            return
        end if

        allocate (covariance(n, n))
        do i = 1, n
            covariance(i, :) = matern52_cross(train_x(i, :), train_x, &
                lengthscales, signal)
        end do
        do i = 1, n
            covariance(i, i) = covariance(i, i) + noise
        end do

        allocate (fitted)
        call fitted%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate (fitted%alpha(n))
        fitted%alpha = train_y
        call fitted%factorization%solve(fitted%alpha, status)
        if (status%code /= FORTNUM_OK) return
        fitted%dimension = d
        fitted%signal_variance = signal
        fitted%noise_variance = noise
        allocate (fitted%lengthscales, source=lengthscales)
        allocate (fitted%train_x, source=train_x)
        fitted%fitted = .true.
        call move_alloc(fitted, posterior)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_fit_ard_posterior

    subroutine ard_posterior_moments(self, points, mean, variance, status)
        class(fortbo_ard_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        integer :: i

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension .or. size(mean) /= size(points, 1) &
                .or. size(variance) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: query shape is invalid")
            return
        end if

        allocate (cross(size(self%train_x, 1), size(points, 1)))
        do i = 1, size(points, 1)
            cross(:, i) = matern52_cross(points(i, :), self%train_x, &
                self%lengthscales, self%signal_variance)
        end do
        mean = matmul(transpose(cross), self%alpha)
        allocate (work, source=cross)
        call self%factorization%solve_lower_matrix(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = self%signal_variance - sum(work*work, dim=1)
        do i = 1, size(variance)
            if (variance(i) < -1.0e-9_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "fortbo ARD posterior: posterior variance is negative")
                return
            end if
            variance(i) = max(variance(i), 0.0_dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ard_posterior_moments

    pure function matern52_cross(x1, x2, lengthscales, signal) result(values)
        real(dp), intent(in) :: x1(:), x2(:, :), lengthscales(:), signal
        real(dp) :: values(size(x2, 1))
        real(dp) :: scaled, squared
        integer :: i

        do i = 1, size(x2, 1)
            squared = sum(((x1 - x2(i, :))/lengthscales)**2)
            scaled = sqrt(5.0_dp*squared)
            values(i) = signal*(1.0_dp + scaled + scaled*scaled/3.0_dp) * &
                exp(-scaled)
        end do
    end function matern52_cross

end module fortbo_ard_posterior
