module fortbo_variational_derivative
    !! Fixed-hyperparameter inducing-point GP with paired value/gradient data.
    !!
    !! The variational posterior is Gaussian in the inducing values.  For
    !! Gaussian observation noise its two required factorizations are
    !!
    !!   K = K(Z,Z),
    !!   B = K + noise^-1 K(Z,X) K(X,Z).
    !!
    !! The predictive mean uses noise^-1 B^-1 K(Z,X)y and the latent
    !! covariance is K** - K* K^-1 K*^T + K* B^-1 K*^T.  A derivative row is
    !! simply another observation in K(Z,X), with the covariance differentiated
    !! with respect to its observation point.  This is the fixed-hyperparameter
    !! local model used to exercise the paired-observation path; it is not a
    !! claim of published DTuRBO stochastic-training parity.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_normal
    use fortbo_history, only: fortbo_history_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE, &
        FORTBO_CAP_MOMENT_GRADIENT
    implicit none
    private

    real(dp), parameter :: FACTORIZATION_JITTER = 1.0e-12_dp
    real(dp), parameter :: JOINT_JITTER = 1.0e-12_dp

    public :: fortbo_variational_derivative_posterior_t
    public :: fortbo_fit_variational_derivative

    type, extends(fortbo_posterior_t) :: fortbo_variational_derivative_posterior_t
        integer :: dimension = 0
        integer :: n_inducing = 0
        integer :: n_observations = 0
        integer :: n_gradient_observations = 0
        real(dp), allocatable :: inducing_points(:, :)
        real(dp), allocatable :: beta(:)
        real(dp), allocatable :: lengthscales(:)
        real(dp) :: signal_variance = 1.0_dp
        real(dp) :: noise_variance = 1.0e-6_dp
        type(cholesky_factorization_t) :: prior_factorization
        type(cholesky_factorization_t) :: variational_factorization
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => variational_n_inputs
        procedure, public :: capabilities => variational_capabilities
        procedure, public :: moments => variational_moments
        procedure, public :: covariance => variational_covariance
        procedure, public :: joint_sample => variational_joint_sample
        procedure, public :: moment_gradient => variational_moment_gradient
    end type fortbo_variational_derivative_posterior_t

contains

    pure integer function variational_n_inputs(self) result(n)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self

        n = self%dimension
    end function variational_n_inputs

    pure integer function variational_capabilities(self) result(caps)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_COVARIANCE &
            + FORTBO_CAP_JOINT_SAMPLE + FORTBO_CAP_MOMENT_GRADIENT
    end function variational_capabilities

    !! Fit the fixed-hyperparameter variational derivative model.  All usable
    !! value observations are retained.  When enabled, every complete gradient
    !! row contributes one additional observation per input coordinate.
    subroutine fortbo_fit_variational_derivative(history, inducing_points, &
            posterior, lengthscales, status, signal_variance, noise_variance, &
            use_gradients)
        type(fortbo_history_t), intent(in) :: history
        real(dp), intent(in) :: inducing_points(:, :)
        class(fortbo_posterior_t), intent(out), allocatable :: posterior
        real(dp), intent(in) :: lengthscales(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: signal_variance
        real(dp), intent(in), optional :: noise_variance
        logical, intent(in), optional :: use_gradients
        type(fortbo_variational_derivative_posterior_t), allocatable :: fitted
        real(dp), allocatable :: value_x(:, :), value_y(:)
        real(dp), allocatable :: gradient_x(:, :), gradient_y(:), gradients(:, :)
        real(dp), allocatable :: observation_x(:, :), observation_y(:)
        integer, allocatable :: components(:)
        real(dp), allocatable :: prior(:, :), cross(:, :), rhs(:)
        real(dp) :: signal, noise, jitter
        integer :: d, m, n_value, n_gradient, n_observations, i, j, row
        logical :: use_derivatives

        d = history%n_inputs
        m = size(inducing_points, 1)
        if (d < 1 .or. m < 1 .or. size(inducing_points, 2) /= d .or. &
                size(lengthscales) /= d .or. any(lengthscales <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: dimensions are invalid")
            return
        end if
        signal = 1.0_dp
        noise = 1.0e-6_dp
        if (present(signal_variance)) signal = signal_variance
        if (present(noise_variance)) noise = noise_variance
        if (signal <= 0.0_dp .or. noise <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: signal and noise must be positive")
            return
        end if

        call history%training_data(value_x, value_y, status)
        if (status%code /= FORTNUM_OK) return
        n_value = size(value_y)
        if (n_value < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: no usable value observations")
            return
        end if

        use_derivatives = history%gradient_count() > 0
        if (present(use_gradients)) then
            if (use_gradients .and. history%gradient_count() == 0) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "fortbo variational derivative: gradients were requested but history has none")
                return
            end if
            use_derivatives = use_gradients
        end if
        n_gradient = 0
        if (use_derivatives) then
            call history%gradient_data(gradient_x, gradient_y, gradients, status)
            if (status%code /= FORTNUM_OK) return
            n_gradient = size(gradient_y)
        end if

        n_observations = n_value + n_gradient*d
        allocate (observation_x(n_observations, d), observation_y(n_observations), &
            components(n_observations))
        observation_x(1:n_value, :) = value_x
        observation_y(1:n_value) = value_y
        components(1:n_value) = 0
        row = n_value
        do i = 1, n_gradient
            do j = 1, d
                row = row + 1
                observation_x(row, :) = gradient_x(i, :)
                observation_y(row) = gradients(i, j)
                components(row) = j
            end do
        end do

        allocate (prior(m, m))
        do i = 1, m
            do j = 1, m
                prior(i, j) = matern52_value(inducing_points(i, :), &
                    inducing_points(j, :), lengthscales, signal)
            end do
        end do
        jitter = FACTORIZATION_JITTER*max(signal, 1.0_dp)
        do i = 1, m
            prior(i, i) = prior(i, i) + jitter
        end do

        allocate (fitted)
        call fitted%prior_factorization%factorize(prior, status)
        if (status%code /= FORTNUM_OK) return
        allocate (cross(m, n_observations))
        do j = 1, n_observations
            do i = 1, m
                cross(i, j) = inducing_observation_covariance( &
                    inducing_points(i, :), observation_x(j, :), components(j), &
                    lengthscales, signal)
            end do
        end do

        prior = prior + (1.0_dp/noise)*matmul(cross, transpose(cross))
        call fitted%variational_factorization%factorize(prior, status)
        if (status%code /= FORTNUM_OK) return
        allocate (rhs(m), fitted%beta(m))
        rhs = (1.0_dp/noise)*matmul(cross, observation_y)
        call fitted%variational_factorization%solve(rhs, status)
        if (status%code /= FORTNUM_OK) return
        fitted%beta = rhs
        fitted%dimension = d
        fitted%n_inducing = m
        fitted%n_observations = n_observations
        fitted%n_gradient_observations = n_gradient*d
        fitted%signal_variance = signal
        fitted%noise_variance = noise
        allocate (fitted%inducing_points, source=inducing_points)
        allocate (fitted%lengthscales, source=lengthscales)
        fitted%fitted = .true.
        call move_alloc(fitted, posterior)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_fit_variational_derivative

    subroutine variational_moments(self, points, mean, variance, status)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior_solved(:, :), variational_solved(:, :)
        integer :: i

        mean = 0.0_dp
        variance = 0.0_dp
        call check_query(self, points, size(mean), status)
        if (status%code /= FORTNUM_OK) return
        allocate (cross(self%n_inducing, size(points, 1)))
        do i = 1, size(points, 1)
            call query_cross(self, points(i, :), cross(:, i))
        end do
        mean = matmul(transpose(cross), self%beta)
        allocate (prior_solved, source=cross)
        allocate (variational_solved, source=cross)
        call self%prior_factorization%solve(prior_solved, status)
        if (status%code /= FORTNUM_OK) return
        call self%variational_factorization%solve(variational_solved, status)
        if (status%code /= FORTNUM_OK) return
        variance = self%signal_variance - sum(cross*prior_solved, dim=1) &
            + sum(cross*variational_solved, dim=1)
        do i = 1, size(variance)
            if (variance(i) < -1.0e-9_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "fortbo variational derivative: posterior variance is negative")
                return
            end if
            variance(i) = max(variance(i), 0.0_dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_moments

    subroutine variational_covariance(self, points, covariance, status)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior_solved(:, :), variational_solved(:, :)
        integer :: n, i, j

        covariance = 0.0_dp
        n = size(points, 1)
        call check_query(self, points, n, status)
        if (status%code /= FORTNUM_OK) return
        if (size(covariance, 1) /= n .or. size(covariance, 2) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: covariance shape is invalid")
            return
        end if
        allocate (cross(self%n_inducing, n), prior_solved(self%n_inducing, n), &
            variational_solved(self%n_inducing, n))
        do i = 1, n
            call query_cross(self, points(i, :), cross(:, i))
            do j = 1, n
                covariance(i, j) = matern52_value(points(i, :), points(j, :), &
                    self%lengthscales, self%signal_variance)
            end do
        end do
        prior_solved = cross
        variational_solved = cross
        call self%prior_factorization%solve(prior_solved, status)
        if (status%code /= FORTNUM_OK) return
        call self%variational_factorization%solve(variational_solved, status)
        if (status%code /= FORTNUM_OK) return
        covariance = covariance - matmul(transpose(cross), prior_solved) &
            + matmul(transpose(cross), variational_solved)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_covariance

    subroutine variational_joint_sample(self, points, generator, samples, status)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: mean(:), covariance(:, :), draw(:)
        integer :: n, i, s

        samples = 0.0_dp
        n = size(points, 1)
        call check_query(self, points, n, status)
        if (status%code /= FORTNUM_OK) return
        if (size(samples, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: sample count must be positive")
            return
        end if
        allocate (mean(n), covariance(n, n), draw(n))
        call self%moments(points, mean, draw, status)
        if (status%code /= FORTNUM_OK) return
        call self%covariance(points, covariance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            covariance(i, i) = covariance(i, i) + JOINT_JITTER
        end do
        call factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, size(samples, 2)
            do i = 1, n
                call rng_normal(generator, draw(i))
            end do
            samples(:, s) = mean + matmul(factorization%lower, draw)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_joint_sample

    subroutine variational_moment_gradient(self, points, mean_gradient, &
            sd_gradient, status)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean_gradient(:, :), sd_gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior_solved(:, :), variational_solved(:, :)
        real(dp), allocatable :: cross_gradient(:, :, :)
        real(dp) :: variance, standard_deviation, variance_gradient
        integer :: n, d, i, j

        mean_gradient = 0.0_dp
        sd_gradient = 0.0_dp
        n = size(points, 1)
        d = self%dimension
        call check_query(self, points, n, status)
        if (status%code /= FORTNUM_OK) return
        if (size(mean_gradient, 1) /= n .or. size(mean_gradient, 2) /= d .or. &
                size(sd_gradient, 1) /= n .or. size(sd_gradient, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: gradient shape is invalid")
            return
        end if
        allocate (cross(self%n_inducing, n), prior_solved(self%n_inducing, n), &
            variational_solved(self%n_inducing, n), &
            cross_gradient(self%n_inducing, n, d))
        do i = 1, n
            call query_cross_and_gradient(self, points(i, :), cross(:, i), &
                cross_gradient(:, i, :))
        end do
        prior_solved = cross
        variational_solved = cross
        call self%prior_factorization%solve(prior_solved, status)
        if (status%code /= FORTNUM_OK) return
        call self%variational_factorization%solve(variational_solved, status)
        if (status%code /= FORTNUM_OK) return

        do i = 1, n
            variance = self%signal_variance - dot_product(cross(:, i), prior_solved(:, i)) &
                + dot_product(cross(:, i), variational_solved(:, i))
            if (variance < -1.0e-9_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "fortbo variational derivative: posterior variance is negative")
                return
            end if
            standard_deviation = sqrt(max(variance, 0.0_dp))
            do j = 1, d
                mean_gradient(i, j) = dot_product(cross_gradient(:, i, j), self%beta)
                variance_gradient = -2.0_dp*dot_product(cross_gradient(:, i, j), &
                    prior_solved(:, i)) + 2.0_dp*dot_product(cross_gradient(:, i, j), &
                    variational_solved(:, i))
                if (standard_deviation > 0.0_dp) then
                    sd_gradient(i, j) = 0.5_dp*variance_gradient/standard_deviation
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_moment_gradient

    subroutine query_cross(self, point, cross)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: cross(:)
        integer :: i

        do i = 1, self%n_inducing
            cross(i) = matern52_value(point, self%inducing_points(i, :), &
                self%lengthscales, self%signal_variance)
        end do
    end subroutine query_cross

    subroutine query_cross_and_gradient(self, point, cross, cross_gradient)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: cross(:), cross_gradient(:, :)
        real(dp) :: value, gradient(size(point)), mixed(size(point), size(point))
        integer :: i

        do i = 1, self%n_inducing
            call matern52_value_gradient_hessian(point, self%inducing_points(i, :), &
                self%lengthscales, self%signal_variance, value, gradient, mixed)
            cross(i) = value
            cross_gradient(i, :) = gradient
        end do
    end subroutine query_cross_and_gradient

    pure real(dp) function inducing_observation_covariance(inducing, observation, &
            component, lengthscales, signal) result(covariance)
        real(dp), intent(in) :: inducing(:), observation(:), lengthscales(:), signal
        integer, intent(in) :: component
        real(dp) :: value, gradient(size(inducing)), mixed(size(inducing), size(inducing))

        call matern52_value_gradient_hessian(inducing, observation, lengthscales, &
            signal, value, gradient, mixed)
        if (component == 0) then
            covariance = value
        else
            covariance = -gradient(component)
        end if
    end function inducing_observation_covariance

    pure real(dp) function matern52_value(x1, x2, lengthscales, signal) result(value)
        real(dp), intent(in) :: x1(:), x2(:), lengthscales(:), signal
        real(dp) :: scaled

        scaled = sqrt(5.0_dp*sum(((x1 - x2)/lengthscales)**2))
        value = signal*(1.0_dp + scaled + scaled*scaled/3.0_dp)*exp(-scaled)
    end function matern52_value

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

    subroutine check_query(self, points, n_out, status)
        class(fortbo_variational_derivative_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: n_out
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension .or. n_out /= size(points, 1) .or. &
                n_out < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo variational derivative: query shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_query

end module fortbo_variational_derivative
