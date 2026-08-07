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
    use fortml_kernels, only: kernel_t, KERNEL_MATERN52, KERNEL_RBF
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_NOISY_MOMENTS
    use fortbo_history, only: fortbo_history_t
    implicit none
    private

    public :: fortbo_gp_posterior_t
    public :: fortbo_derivative_gp_posterior_t
    public :: fortbo_fit_from_history

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

    pure integer function derivative_gp_capabilities(self) result(caps)
        class(fortbo_derivative_gp_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS
    end function derivative_gp_capabilities

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

    !! Fit whichever surrogate the history's contents justify, and return it as
    !! an allocatable posterior. The caller receives `fortbo_posterior_t` and
    !! cannot tell which branch was taken without asking.
    subroutine fortbo_fit_from_history(history, posterior, status, lengthscale, &
            signal_variance, noise_variance, use_gradients)
        type(fortbo_history_t), intent(in) :: history
        class(fortbo_posterior_t), intent(out), allocatable :: posterior
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: lengthscale
        real(dp), intent(in), optional :: signal_variance
        real(dp), intent(in), optional :: noise_variance
        !! Force the branch. Absent means "use gradients when the history has
        !! them", which is the behavior the roadmap requires.
        logical, intent(in), optional :: use_gradients
        type(fortbo_gp_posterior_t), allocatable :: value_only
        type(fortbo_derivative_gp_posterior_t), allocatable :: with_gradients
        type(kernel_t) :: kernel
        real(dp), allocatable :: inputs(:, :), objectives(:), gradients(:, :)
        real(dp), allocatable :: rows(:, :), targets(:, :)
        integer, allocatable :: components(:)
        real(dp) :: noise
        integer :: d, n, i, j, row, expanded
        logical :: wanted

        d = history%n_inputs
        if (d < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo fortml: history is not initialized")
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
            call history%gradient_data(inputs, objectives, gradients, status)
            if (status%code /= FORTNUM_OK) return
            n = size(objectives)
            if (n < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo fortml: no usable gradient observations")
                return
            end if
            ! One value row plus one row per coordinate, all at the same input.
            expanded = n*(1 + d)
            allocate (rows(expanded, d), targets(expanded, 1), components(expanded))
            row = 0
            do i = 1, n
                row = row + 1
                rows(row, :) = inputs(i, :)
                targets(row, 1) = objectives(i)
                components(row) = 0
                do j = 1, d
                    row = row + 1
                    rows(row, :) = inputs(i, :)
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
