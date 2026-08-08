module fortbo_ard_posterior
    !! Exact fixed-hyperparameter Matern-5/2 ARD posterior.
    !!
    !! An observation is a function value (component 0) or one coordinate of
    !! its derivative (component 1..dimension).  Values and derivatives at a
    !! point are therefore ordinary rows in one dense covariance matrix.  The
    !! adapter is deliberately fixed-hyperparameter: it is a parity lane for
    !! replay data, not a hyperparameter optimizer.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_normal
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_NOISY_MOMENTS, FORTBO_CAP_JOINT_SAMPLE, &
        FORTBO_CAP_MOMENT_GRADIENT
    implicit none
    private

    public :: fortbo_ard_posterior_t
    public :: fortbo_fit_ard_posterior

    type, extends(fortbo_posterior_t) :: fortbo_ard_posterior_t
        integer :: dimension = 0
        real(dp), allocatable :: train_x(:, :)
        integer, allocatable :: components(:)
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
        procedure, public :: joint_sample => ard_posterior_joint_sample
        procedure, public :: moment_gradient => ard_posterior_moment_gradient
    end type fortbo_ard_posterior_t

contains

    pure integer function ard_posterior_n_inputs(self) result(n)
        class(fortbo_ard_posterior_t), intent(in) :: self

        n = self%dimension
    end function ard_posterior_n_inputs

    pure integer function ard_posterior_capabilities(self) result(caps)
        class(fortbo_ard_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS &
            + FORTBO_CAP_JOINT_SAMPLE + FORTBO_CAP_MOMENT_GRADIENT
    end function ard_posterior_capabilities

    !! Fit a fixed-hyperparameter exact ARD posterior.  When requested, every
    !! complete history row contributes one value and one observation per
    !! derivative coordinate at the same input.
    subroutine fortbo_fit_ard_posterior(history, posterior, lengthscales, status, &
            signal_variance, noise_variance, use_gradients)
        type(fortbo_history_t), intent(in) :: history
        class(fortbo_posterior_t), intent(out), allocatable :: posterior
        real(dp), intent(in) :: lengthscales(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: signal_variance
        real(dp), intent(in), optional :: noise_variance
        logical, intent(in), optional :: use_gradients
        type(fortbo_ard_posterior_t), allocatable :: fitted
        real(dp), allocatable :: value_x(:, :), value_y(:), gradients(:, :)
        real(dp), allocatable :: train_x(:, :), train_y(:), covariance(:, :)
        integer, allocatable :: components(:)
        real(dp) :: signal, noise
        integer :: n, nobs, d, i, j, row
        logical :: use_derivatives

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

        use_derivatives = history%gradient_count() > 0
        if (present(use_gradients)) then
            if (use_gradients .and. history%gradient_count() == 0) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "fortbo ARD posterior: gradients were requested but history has none")
                return
            end if
            use_derivatives = use_gradients
        end if

        if (use_derivatives) then
            call history%gradient_data(value_x, value_y, gradients, status)
            if (status%code /= FORTNUM_OK) return
            n = size(value_y)
            if (n < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo ARD posterior: no usable gradient observations")
                return
            end if
            nobs = n*(1 + d)
            allocate (train_x(nobs, d), train_y(nobs), components(nobs))
            row = 0
            do i = 1, n
                row = row + 1
                train_x(row, :) = value_x(i, :)
                train_y(row) = value_y(i)
                components(row) = 0
                do j = 1, d
                    row = row + 1
                    train_x(row, :) = value_x(i, :)
                    train_y(row) = gradients(i, j)
                    components(row) = j
                end do
            end do
        else
            call history%training_data(train_x, train_y, status)
            if (status%code /= FORTNUM_OK) return
            nobs = size(train_y)
            if (nobs < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo ARD posterior: no usable observations")
                return
            end if
            allocate (components(nobs))
            components = 0
        end if

        allocate (covariance(nobs, nobs))
        do i = 1, nobs
            do j = 1, nobs
                covariance(i, j) = matern52_observation_covariance( &
                    train_x(i, :), components(i), train_x(j, :), components(j), &
                    lengthscales, signal)
            end do
        end do
        do i = 1, nobs
            covariance(i, i) = covariance(i, i) + noise
        end do

        allocate (fitted)
        call fitted%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate (fitted%alpha(nobs))
        fitted%alpha = train_y
        call fitted%factorization%solve(fitted%alpha, status)
        if (status%code /= FORTNUM_OK) return
        fitted%dimension = d
        fitted%signal_variance = signal
        fitted%noise_variance = noise
        allocate (fitted%lengthscales, source=lengthscales)
        allocate (fitted%train_x, source=train_x)
        allocate (fitted%components, source=components)
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
        real(dp), allocatable :: cross(:, :), solved(:, :)
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
            call matern52_observation_cross(points(i, :), self%train_x, &
                self%components, self%lengthscales, self%signal_variance, cross(:, i))
        end do
        mean = matmul(transpose(cross), self%alpha)
        allocate (solved, source=cross)
        call self%factorization%solve_lower_matrix(solved, status)
        if (status%code /= FORTNUM_OK) return
        variance = self%signal_variance - sum(solved*solved, dim=1)
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

    !! Draw exact joint latent posterior samples at the query points.
    subroutine ard_posterior_joint_sample(self, points, generator, samples, status)
        class(fortbo_ard_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: predictive_factor
        real(dp), allocatable :: mean(:), variance(:), cross(:, :), solved(:, :)
        real(dp), allocatable :: covariance(:, :), base(:, :)
        integer :: n_query, n_samples, i, j

        samples = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: surrogate has not been fitted")
            return
        end if
        n_query = size(points, 1)
        n_samples = size(samples, 2)
        if (size(points, 2) /= self%dimension .or. size(samples, 1) /= n_query &
                .or. n_query < 1 .or. n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: joint sample shape is invalid")
            return
        end if

        allocate (mean(n_query), variance(n_query), &
            cross(size(self%train_x, 1), n_query))
        allocate (solved, source=cross)
        allocate (covariance(n_query, n_query))
        call self%moments(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n_query
            call matern52_observation_cross(points(i, :), self%train_x, &
                self%components, self%lengthscales, self%signal_variance, cross(:, i))
            do j = 1, n_query
                covariance(i, j) = matern52_observation_covariance(points(i, :), 0, &
                    points(j, :), 0, self%lengthscales, self%signal_variance)
            end do
        end do
        solved = cross
        call self%factorization%solve(solved, status)
        if (status%code /= FORTNUM_OK) return
        covariance = covariance - matmul(transpose(cross), solved)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        do i = 1, n_query
            covariance(i, i) = covariance(i, i) + 1.0e-12_dp
        end do
        call predictive_factor%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate (base(n_query, n_samples))
        do j = 1, n_samples
            do i = 1, n_query
                call rng_normal(generator, base(i, j))
            end do
        end do
        samples = spread(mean, dim=2, ncopies=n_samples) + &
            matmul(predictive_factor%lower, base)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ard_posterior_joint_sample

    !! Gradients of the latent posterior mean and standard deviation.
    subroutine ard_posterior_moment_gradient(self, points, mean_gradient, &
            sd_gradient, status)
        class(fortbo_ard_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean_gradient(:, :)
        real(dp), intent(out) :: sd_gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), cross(:, :), solved(:, :)
        real(dp), allocatable :: cross_gradient(:, :, :)
        real(dp) :: standard_deviation
        integer :: n_query, n_train, d, i, j

        mean_gradient = 0.0_dp
        sd_gradient = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: surrogate has not been fitted")
            return
        end if
        n_query = size(points, 1)
        d = self%dimension
        n_train = size(self%train_x, 1)
        if (size(points, 2) /= d .or. size(mean_gradient, 1) /= n_query .or. &
                size(mean_gradient, 2) /= d .or. size(sd_gradient, 1) /= n_query &
                .or. size(sd_gradient, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo ARD posterior: gradient shape is invalid")
            return
        end if

        allocate (mean(n_query), variance(n_query), cross(n_train, n_query))
        allocate (solved, source=cross)
        allocate (cross_gradient(n_train, n_query, d))
        do i = 1, n_query
            call matern52_observation_cross(points(i, :), self%train_x, &
                self%components, self%lengthscales, self%signal_variance, cross(:, i))
            do j = 1, n_train
                call matern52_observation_gradient(points(i, :), self%train_x(j, :), &
                    self%components(j), self%lengthscales, self%signal_variance, &
                    cross_gradient(j, i, :))
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        solved = cross
        call self%factorization%solve(solved, status)
        if (status%code /= FORTNUM_OK) return
        variance = self%signal_variance - sum(cross*solved, dim=1)
        do i = 1, n_query
            if (variance(i) < -1.0e-9_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "fortbo ARD posterior: posterior variance is negative")
                return
            end if
            variance(i) = max(variance(i), 0.0_dp)
            standard_deviation = sqrt(variance(i))
            do j = 1, d
                mean_gradient(i, j) = dot_product(cross_gradient(:, i, j), self%alpha)
                if (standard_deviation > 0.0_dp) then
                    sd_gradient(i, j) = -dot_product(cross_gradient(:, i, j), &
                        solved(:, i))/standard_deviation
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ard_posterior_moment_gradient

    pure subroutine matern52_value_gradient_hessian(x1, x2, lengthscales, signal, &
            value, gradient_x1, mixed)
        real(dp), intent(in) :: x1(:), x2(:), lengthscales(:), signal
        real(dp), intent(out) :: value, gradient_x1(:), mixed(:, :)
        real(dp) :: delta(size(x1)), scaled, squared, coefficient, second
        integer :: i, j

        delta = x1 - x2
        squared = sum((delta/lengthscales)**2)
        scaled = sqrt(5.0_dp*squared)
        value = signal*(1.0_dp + scaled + scaled*scaled/3.0_dp)*exp(-scaled)
        coefficient = -5.0_dp*signal*(1.0_dp + scaled)*exp(-scaled)/3.0_dp
        gradient_x1 = coefficient*delta/(lengthscales**2)
        second = 25.0_dp*signal*exp(-scaled)/3.0_dp
        mixed = 0.0_dp
        do i = 1, size(x1)
            do j = 1, size(x1)
                if (i == j) mixed(i, j) = -coefficient/(lengthscales(i)**2)
                mixed(i, j) = mixed(i, j) - second*delta(i)*delta(j) / &
                    (lengthscales(i)**2*lengthscales(j)**2)
            end do
        end do
    end subroutine matern52_value_gradient_hessian

    pure real(dp) function matern52_observation_covariance(x1, component1, x2, &
            component2, lengthscales, signal) result(covariance)
        real(dp), intent(in) :: x1(:), x2(:), lengthscales(:), signal
        integer, intent(in) :: component1, component2
        real(dp) :: value, gradient(size(x1)), mixed(size(x1), size(x1))

        call matern52_value_gradient_hessian(x1, x2, lengthscales, signal, value, &
            gradient, mixed)
        if (component1 == 0 .and. component2 == 0) then
            covariance = value
        else if (component1 > 0 .and. component2 == 0) then
            covariance = gradient(component1)
        else if (component1 == 0 .and. component2 > 0) then
            covariance = -gradient(component2)
        else
            covariance = mixed(component1, component2)
        end if
    end function matern52_observation_covariance

    pure subroutine matern52_observation_cross(query, train_x, components, &
            lengthscales, signal, values)
        real(dp), intent(in) :: query(:), train_x(:, :), lengthscales(:), signal
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: values(:)
        real(dp) :: value, gradient(size(query)), mixed(size(query), size(query))
        integer :: i

        do i = 1, size(train_x, 1)
            call matern52_value_gradient_hessian(query, train_x(i, :), lengthscales, &
                signal, value, gradient, mixed)
            if (components(i) == 0) then
                values(i) = value
            else
                ! The observation derivative is with respect to train_x.
                values(i) = -gradient(components(i))
            end if
        end do
    end subroutine matern52_observation_cross

    pure subroutine matern52_observation_gradient(query, train_x, component, &
            lengthscales, signal, gradient_query)
        real(dp), intent(in) :: query(:), train_x(:), lengthscales(:), signal
        integer, intent(in) :: component
        real(dp), intent(out) :: gradient_query(:)
        real(dp) :: value, gradient(size(query)), mixed(size(query), size(query))

        call matern52_value_gradient_hessian(query, train_x, lengthscales, signal, &
            value, gradient, mixed)
        if (component == 0) then
            gradient_query = gradient
        else
            gradient_query = mixed(:, component)
        end if
    end subroutine matern52_observation_gradient

end module fortbo_ard_posterior
