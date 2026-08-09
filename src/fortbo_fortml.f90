module fortbo_fortml
    !! FortML surrogates behind the FortBO posterior contract (ROADMAP BO2).
    !!
    !! This module is the whole of the derivative-observation story on the
    !! FortBO side. Above it, every acquisition and every policy sees only
    !! `fortbo_posterior_t`; below it, FortML decides whether the surrogate was
    !! conditioned on values alone or on values and gradients. Because the two
    !! adapters present the *same* contract with the same capability bits, a
    !! run that starts measuring adjoints becomes derivative-informed
    !! everywhere at once — in EI, in Thompson sampling, in TuRBO — without a
    !! single line changing above this boundary. That is the invariant the
    !! roadmap states, made concrete.
    !!
    !! `fortbo_fit_from_history` is where the choice is made, and it is made
    !! from the data rather than from a flag: a history carrying complete
    !! gradients yields a derivative-observation GP, one without yields the
    !! value-only GP. Discarding measured gradients would be throwing away
    !! information the run already paid for.
    !!
    !! FortML's derivative GP encodes each training row with a component index:
    !! zero for a function value, `j` for the partial derivative with respect to
    !! coordinate `j`. A history row carrying a value and a full gradient
    !! therefore expands into `1 + d` observation rows at the same input.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_kernels, only: kernel_t, KERNEL_MATERN52
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_NOISY_MOMENTS, FORTBO_CAP_MOMENT_GRADIENT, &
        FORTBO_CAP_MEAN_HESSIAN, FORTBO_CAP_MOMENT_HESSIAN
    use fortbo_history, only: fortbo_history_t
    use fortbo_ard_posterior, only: fortbo_fit_ard_posterior
    use fortbo_variational_derivative, only: fortbo_fit_variational_derivative
    implicit none
    private

    public :: fortbo_gp_posterior_t
    public :: fortbo_derivative_gp_posterior_t
    public :: fortbo_fit_from_history

    !! Below this standard deviation the square root's cusp dominates and no
    !! finite curvature exists. The bound is not a tuning knob: the second term
    !! of the chain rule scales as `1/sd^3`, so at this value it already reaches
    !! about 1e18 and any Newton step built on it is meaningless.
    real(dp), parameter, public :: FORTBO_SD_HESSIAN_FLOOR = 1.0e-6_dp

    !! Value-only GP presented as a posterior.
    type, extends(fortbo_posterior_t) :: fortbo_gp_posterior_t
        type(gp_regression_t) :: model
        integer :: dimension = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => gp_posterior_n_inputs
        procedure, public :: capabilities => gp_posterior_capabilities
        procedure, public :: moments => gp_posterior_moments
    end type fortbo_gp_posterior_t

    !! Derivative-observation GP presented as the *same* posterior contract.
    !! Nothing downstream can tell the difference, which is the point.
    type, extends(fortbo_posterior_t) :: fortbo_derivative_gp_posterior_t
        type(gp_derivative_regression_t) :: model
        integer :: dimension = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => derivative_gp_n_inputs
        procedure, public :: capabilities => derivative_gp_capabilities
        procedure, public :: moments => derivative_gp_moments
        procedure, public :: moment_gradient => derivative_gp_moment_gradient
        procedure, public :: moment_hessian => derivative_gp_moment_hessian
        procedure, public :: mean_hessian => derivative_gp_mean_hessian
    end type fortbo_derivative_gp_posterior_t

contains

    pure integer function gp_posterior_n_inputs(self) result(n)
        class(fortbo_gp_posterior_t), intent(in) :: self

        n = self%dimension
    end function gp_posterior_n_inputs

    pure integer function gp_posterior_capabilities(self) result(caps)
        class(fortbo_gp_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS
    end function gp_posterior_capabilities

    subroutine gp_posterior_moments(self, points, mean, variance, status)
        class(fortbo_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: model_mean(:, :)

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: query width does not match the surrogate")
            return
        end if
        allocate (model_mean(size(points, 1), 1))
        call self%model%predict(points, model_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = model_mean(:, 1)
    end subroutine gp_posterior_moments

    pure integer function derivative_gp_n_inputs(self) result(n)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self

        n = self%dimension
    end function derivative_gp_n_inputs

    !! The derivative-observation model declares moment gradients; the
    !! value-only GP does not. That asymmetry is not a design preference:
    !! FortML's `gp_predict_jvp` differentiates with respect to the *parameters*
    !! rather than the query, so the plain GP has no input-gradient path and
    !! claiming one would be a lie. A run that measures gradients therefore
    !! unlocks gradient-based candidate search as well as a sharper posterior.
    pure integer function derivative_gp_capabilities(self) result(caps)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) then
            caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS &
                + FORTBO_CAP_MOMENT_GRADIENT + FORTBO_CAP_MEAN_HESSIAN &
                + FORTBO_CAP_MOMENT_HESSIAN
        end if
    end function derivative_gp_capabilities

    !! Gradients of the predictive mean and standard deviation with respect to
    !! the query, assembled from one query-input JVP per coordinate.
    !!
    !! The standard deviation's gradient follows from the variance's by the
    !! chain rule, which divides by the standard deviation. At a training site
    !! the posterior variance collapses and that derivative genuinely does not
    !! exist — the square root has a cusp there. Zero is reported, being the
    !! subgradient of least magnitude, rather than an infinity that would wreck
    !! a line search.
    subroutine derivative_gp_moment_gradient(self, points, mean_gradient, &
            sd_gradient, status)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean_gradient(:, :)
        real(dp), intent(out) :: sd_gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: direction(:, :), mean(:, :), mean_dot(:, :)
        real(dp), allocatable :: variance(:), variance_dot(:)
        integer, allocatable :: components(:)
        real(dp) :: standard_deviation
        integer :: n, d, i, j

        mean_gradient = 0.0_dp
        sd_gradient = 0.0_dp
        n = size(points, 1)
        d = self%dimension
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= d .or. size(mean_gradient, 2) /= d &
            .or. size(sd_gradient, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: gradient width does not match the surrogate")
            return
        end if

        allocate (direction(n, d), mean(n, 1), mean_dot(n, 1))
        allocate (variance(n), variance_dot(n), components(n))
        components = 0
        do j = 1, d
            direction = 0.0_dp
            direction(:, j) = 1.0_dp
            call self%model%predict_input_jvp(points, components, direction, mean, &
                mean_dot, variance, variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            mean_gradient(:, j) = mean_dot(:, 1)
            do i = 1, n
                standard_deviation = sqrt(max(variance(i), 0.0_dp))
                if (standard_deviation > 0.0_dp) then
                    sd_gradient(i, j) = 0.5_dp*variance_dot(i)/standard_deviation
                else
                    sd_gradient(i, j) = 0.0_dp
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_gp_moment_gradient

    !! Predict the function value, not a derivative, at each query row. The
    !! component vector is therefore all zeros: the gradients live in the
    !! *training* data, where they sharpen the posterior, and an acquisition
    !! asking for moments still wants the value.
    subroutine derivative_gp_moments(self, points, mean, variance, status)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: model_mean(:, :)
        integer, allocatable :: components(:)

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: query width does not match the surrogate")
            return
        end if
        allocate (model_mean(size(points, 1), 1))
        allocate (components(size(points, 1)))
        components = 0
        call self%model%predict(points, components, model_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = model_mean(:, 1)
    end subroutine derivative_gp_moments

    !! Hessian of the predictive mean, exactly.
    !!
    !! The trick is that a derivative-observation GP can *predict a derivative
    !! component*: asking it for component `j` returns the posterior mean of
    !! df/dx_j. Differentiating that prediction with respect to the query, which
    !! is the same query-input JVP used for the gradient, gives
    !!
    !!     H(j,k) = d/dx_k E[ df/dx_j ]
    !!
    !! so the mean's second derivatives fall out of machinery the model already
    !! has. No finite differences, no separate Hessian kernel, and no wait for
    !! FortSym's matrix milestone.
    !!
    !! The result is symmetrized. The two triangles are computed by different
    !! routes through the kernel and agree to rounding rather than to the last
    !! bit; a Newton step wants an exactly symmetric matrix, and averaging is
    !! the projection onto the symmetric matrices in the Frobenius norm.
    subroutine derivative_gp_mean_hessian(self, point, hessian, status)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: query(:, :), direction(:, :)
        real(dp), allocatable :: mean(:, :), mean_dot(:, :)
        real(dp), allocatable :: variance(:), variance_dot(:)
        integer, allocatable :: components(:)
        real(dp), allocatable :: raw(:, :)
        integer :: d, j, k

        hessian = 0.0_dp
        d = self%dimension
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: surrogate has not been fitted")
            return
        end if
        if (size(point) /= d .or. size(hessian, 1) /= d .or. size(hessian, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: hessian width does not match the surrogate")
            return
        end if

        allocate (query(1, d), direction(1, d), mean(1, 1), mean_dot(1, 1))
        allocate (variance(1), variance_dot(1), components(1), raw(d, d))
        query(1, :) = point
        do j = 1, d
            components(1) = j
            do k = 1, d
                direction = 0.0_dp
                direction(1, k) = 1.0_dp
                call self%model%predict_input_jvp(query, components, direction, mean, &
                    mean_dot, variance, variance_dot, &
                    status)
                if (status%code /= FORTNUM_OK) return
                raw(j, k) = mean_dot(1, 1)
            end do
        end do
        hessian = 0.5_dp*(raw + transpose(raw))
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_gp_mean_hessian

    !! Hessians of the marginal mean and standard deviation, the pair DTuRBO's
    !! local quadratic model consumes.
    !!
    !! Both come from FortML's query-input Hessian-vector product, one call per
    !! coordinate, so the whole pair costs `d` products rather than the `d^2`
    !! JVPs the mean-only route needs. The standard deviation's curvature is
    !! then chain-ruled from the variance's:
    !!
    !!     sd = sqrt(v),  d2 sd = v_jk/(2 sd) - v_j v_k/(4 sd^3),
    !!
    !! which needs the variance's *gradient* as well as its Hessian, because the
    !! square root's own curvature contributes a term that no amount of accuracy
    !! in `v_jk` can supply.
    !!
    !! At a training site the standard deviation has a cusp: `sd` goes to zero
    !! and the second term diverges. `moment_gradient` reports the
    !! least-magnitude subgradient there, but no finite Hessian exists, so this
    !! routine refuses by name rather than returning a large number that a
    !! Newton step would silently follow off a cliff.
    subroutine derivative_gp_moment_hessian(self, point, mean_hessian, sd_hessian, &
            status)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: mean_hessian(:, :)
        real(dp), intent(out) :: sd_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: query(:, :), direction(:, :)
        real(dp), allocatable :: mean(:, :), mean_dot(:, :)
        real(dp), allocatable :: variance(:), variance_dot(:)
        real(dp), allocatable :: mean_hvp(:, :), variance_hvp(:)
        real(dp), allocatable :: variance_gradient(:)
        real(dp), allocatable :: raw_mean(:, :), raw_variance(:, :)
        integer, allocatable :: components(:)
        real(dp) :: standard_deviation
        integer :: d, j, k

        mean_hessian = 0.0_dp
        sd_hessian = 0.0_dp
        d = self%dimension
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: surrogate has not been fitted")
            return
        end if
        if (size(point) /= d .or. size(mean_hessian, 1) /= d .or. &
            size(mean_hessian, 2) /= d .or. size(sd_hessian, 1) /= d .or. &
            size(sd_hessian, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: hessian width does not match the surrogate")
            return
        end if

        allocate (query(1, d), direction(1, d), mean(1, 1), mean_dot(1, 1))
        allocate (variance(1), variance_dot(1), components(1))
        allocate (mean_hvp(d, 1), variance_hvp(d), variance_gradient(d))
        allocate (raw_mean(d, d), raw_variance(d, d))
        query(1, :) = point
        components = 0

        ! The variance's gradient, needed for the square root's own curvature.
        do j = 1, d
            direction = 0.0_dp
            direction(1, j) = 1.0_dp
            call self%model%predict_input_jvp(query, components, direction, mean, &
                mean_dot, variance, variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            variance_gradient(j) = variance_dot(1)
        end do

        standard_deviation = sqrt(max(variance(1), 0.0_dp))
        if (standard_deviation <= FORTBO_SD_HESSIAN_FLOOR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: moment_hessian undefined at a cusp")
            return
        end if

        do k = 1, d
            direction(1, :) = 0.0_dp
            direction(1, k) = 1.0_dp
            call self%model%predict_input_hvp(point, direction(1, :), mean_hvp, &
                variance_hvp, status)
            if (status%code /= FORTNUM_OK) return
            raw_mean(:, k) = mean_hvp(:, 1)
            raw_variance(:, k) = variance_hvp
        end do

        ! Symmetrize: the two triangles travel different routes through the
        ! kernel and agree to rounding rather than to the last bit, and a
        ! Newton step wants an exactly symmetric matrix. Averaging is the
        ! projection onto the symmetric matrices in the Frobenius norm.
        raw_mean = 0.5_dp*(raw_mean + transpose(raw_mean))
        raw_variance = 0.5_dp*(raw_variance + transpose(raw_variance))
        mean_hessian = raw_mean
        do j = 1, d
            do k = 1, d
                sd_hessian(j, k) = raw_variance(j, k)/(2.0_dp*standard_deviation) &
                    - variance_gradient(j)*variance_gradient(k) &
                    /(4.0_dp*standard_deviation**3)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_gp_moment_hessian

    !! Fit whichever surrogate the history's contents justify, and return it as
    !! an allocatable posterior. The caller receives `fortbo_posterior_t` and
    !! cannot tell which branch was taken without asking.
    subroutine fortbo_fit_from_history(history, posterior, status, lengthscale, &
            signal_variance, noise_variance, use_gradients, lengthscales, &
            inducing_points)
        type(fortbo_history_t), intent(in) :: history
        class(fortbo_posterior_t), intent(out), allocatable :: posterior
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: lengthscale
        real(dp), intent(in), optional :: signal_variance
        real(dp), intent(in), optional :: noise_variance
        !! Force the branch. Absent means "use gradients when the history has
        !! them", which is the behavior the roadmap requires.
        logical, intent(in), optional :: use_gradients
        !! When present, select the exact fixed-hyperparameter ARD parity adapter. This
        !! is intentionally explicit: a caller must opt in rather than
        !! receiving a different model merely because a vector happens to be
        !! available.
        real(dp), intent(in), optional :: lengthscales(:)
        !! When present with `lengthscales`, select the fixed-hyperparameter
        !! inducing variational derivative adapter instead of the dense ARD
        !! adapter. The points are in the same unit/input coordinates as the
        !! history and are caller-owned replay state.
        real(dp), intent(in), optional :: inducing_points(:, :)
        type(fortbo_gp_posterior_t), allocatable :: value_only
        type(fortbo_derivative_gp_posterior_t), allocatable :: with_gradients
        type(kernel_t) :: kernel
        real(dp), allocatable :: inputs(:, :), objectives(:), gradients(:, :)
        real(dp), allocatable :: gradient_inputs(:, :), gradient_objectives(:)
        real(dp), allocatable :: rows(:, :), targets(:, :)
        integer, allocatable :: components(:)
        real(dp) :: noise, signal
        integer :: d, n, i, j, row, expanded
        logical :: wanted, wanted_ard

        d = history%n_inputs
        if (d < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: history is not initialized")
            return
        end if

        if (present(inducing_points)) then
            if (.not. present(lengthscales)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo fortml: inducing points require ARD lengthscales")
                return
            end if
            if (present(lengthscale)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo fortml: scalar and ARD lengthscales cannot be combined")
                return
            end if
            noise = 1.0e-6_dp
            if (present(noise_variance)) noise = noise_variance
            signal = 1.0_dp
            if (present(signal_variance)) signal = signal_variance
            call fortbo_fit_variational_derivative(history, inducing_points, &
                posterior, lengthscales, status, signal_variance=signal, &
                noise_variance=noise, use_gradients=use_gradients)
            return
        end if

        if (present(lengthscales)) then
            if (present(use_gradients)) then
                if (use_gradients) then
                    if (history%gradient_count() == 0) then
                        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                            "fortbo fortml: gradients were requested but the history has none")
                        return
                    end if
                end if
            end if
            wanted_ard = history%gradient_count() > 0
            if (present(use_gradients)) wanted_ard = use_gradients
            noise = 1.0e-6_dp
            if (present(noise_variance)) noise = noise_variance
            signal = 1.0_dp
            if (present(signal_variance)) signal = signal_variance
            call fortbo_fit_ard_posterior(history, posterior, lengthscales, status, &
                signal_variance=signal, noise_variance=noise, &
                use_gradients=wanted_ard)
            return
        end if

        wanted = history%gradient_count() > 0
        if (present(use_gradients)) then
            if (use_gradients .and. history%gradient_count() == 0) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "fortbo fortml: gradients were requested but the history has none")
                return
            end if
            wanted = use_gradients
        end if

        noise = 1.0e-6_dp
        if (present(noise_variance)) noise = noise_variance
        call build_kernel(kernel, d, lengthscale, signal_variance)

        if (wanted) then
            call history%training_data(inputs, objectives, status)
            if (status%code /= FORTNUM_OK) return
            n = size(objectives)
            if (n < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo fortml: no usable observations")
                return
            end if
            call history%gradient_data(gradient_inputs, gradient_objectives, gradients, &
                status)
            if (status%code /= FORTNUM_OK) return
            n = size(gradient_objectives)
            if (n < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo fortml: no usable gradient observations")
                return
            end if
            ! Retain every value row, then append one row per derivative
            ! coordinate for each complete gradient-bearing input.
            expanded = size(objectives) + n*d
            allocate (rows(expanded, d), targets(expanded, 1), components(expanded))
            rows(1:size(objectives), :) = inputs
            targets(1:size(objectives), 1) = objectives
            components(1:size(objectives)) = 0
            row = size(objectives)
            do i = 1, n
                do j = 1, d
                    row = row + 1
                    rows(row, :) = gradient_inputs(i, :)
                    targets(row, 1) = gradients(i, j)
                    components(row) = j
                end do
            end do
            allocate (with_gradients)
            with_gradients%dimension = d
            call with_gradients%model%fit(rows, components, targets, kernel, noise, &
                status)
            if (status%code /= FORTNUM_OK) return
            with_gradients%fitted = .true.
            call move_alloc(with_gradients, posterior)
            return
        end if

        call history%training_data(inputs, objectives, status)
        if (status%code /= FORTNUM_OK) return
        n = size(objectives)
        if (n < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: no usable observations")
            return
        end if
        allocate (targets(n, 1))
        targets(:, 1) = objectives
        allocate (value_only)
        value_only%dimension = d
        call value_only%model%fit(inputs, targets, kernel, noise, status)
        if (status%code /= FORTNUM_OK) return
        value_only%fitted = .true.
        call move_alloc(value_only, posterior)
    end subroutine fortbo_fit_from_history

    !! Matern 5/2 by default: twice differentiable, which is the minimum a
    !! derivative-observation model and a Hessian-based local model both need,
    !! and far less brittle than the squared exponential's assumption of
    !! infinite smoothness.
    subroutine build_kernel(kernel, dimension, lengthscale, signal_variance)
        type(kernel_t), intent(out) :: kernel
        integer, intent(in) :: dimension
        real(dp), intent(in), optional :: lengthscale
        real(dp), intent(in), optional :: signal_variance
        real(dp) :: length, signal

        length = 0.3_dp
        signal = 1.0_dp
        if (present(lengthscale)) length = lengthscale
        if (present(signal_variance)) signal = signal_variance
        kernel%kind = KERNEL_MATERN52
        kernel%input_dim = dimension
        allocate (kernel%log_parameters(2))
        kernel%log_parameters = [log(signal), log(length)]
    end subroutine build_kernel

end module fortbo_fortml
