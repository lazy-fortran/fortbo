module fortbo_structured
    !! Multi-task and deep-kernel FortML surrogates behind the FortBO posterior
    !! contract (ROADMAP BO2).
    !!
    !! Both models existed in FortML with no route into a policy: they could be
    !! fitted and predicted from, but nothing above `fortbo_posterior_t` could
    !! see them, so no acquisition and no policy could use one. These adapters
    !! are that route, and they are deliberately thin -- the modelling lives in
    !! FortML, where it is generic, and only the presentation lives here.
    !!
    !! **The multi-task adapter has to answer a question the model does not
    !! ask.** A multi-output GP is a posterior over a vector; an acquisition is
    !! a function of a scalar. Something must say which output is being
    !! optimized, and the honest place for that is the caller, so
    !! `target_output` is required rather than defaulting to the first. A
    !! default would silently optimize output one on a model built for three,
    !! and the run would look entirely normal.
    !!
    !! The variance reported is the *marginal* for that output, taken from the
    !! diagonal of the joint predictive covariance. That is the right quantity
    !! for a single-objective acquisition on one task: the correlations with
    !! the other tasks have already done their work by sharpening this
    !! output's posterior through the shared data, and reporting anything else
    !! would be answering a different question. What multi-task buys is a
    !! better posterior on the target from the other tasks' observations, not
    !! a different notion of uncertainty about it.
    !!
    !! Neither adapter offers moment gradients. FortML exposes derivative
    !! products for these models with respect to *inputs of the prediction*,
    !! but a gradient-based acquisition search needs the derivative of the
    !! posterior mean and standard deviation with respect to the query point,
    !! and for the deep-kernel model that means differentiating through the
    !! feature map and the Gram solve together. Claiming the capability and
    !! differencing underneath would be worse than declining it: a policy that
    !! asks for gradients gets a refusal by name and can choose a sampling
    !! search, which `fortbo_search_acquisition` provides.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_deep_kernel_gp, only: deep_kernel_gp_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS
    implicit none
    private

    public :: fortbo_multi_task_posterior_t
    public :: fortbo_deep_kernel_posterior_t

    !! A multi-output FortML GP presented as a scalar posterior on one task.
    type, extends(fortbo_posterior_t) :: fortbo_multi_task_posterior_t
        type(multi_output_gp_t) :: model
        integer :: dimension = 0
        !! Which output the acquisition is optimizing. Required, never guessed.
        integer :: target_output = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => multi_task_n_inputs
        procedure, public :: capabilities => multi_task_capabilities
        procedure, public :: moments => multi_task_moments
        procedure, public :: adopt => multi_task_adopt
    end type fortbo_multi_task_posterior_t

    !! A deep-kernel FortML GP presented as a posterior.
    type, extends(fortbo_posterior_t) :: fortbo_deep_kernel_posterior_t
        type(deep_kernel_gp_t) :: model
        integer :: dimension = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => deep_kernel_n_inputs
        procedure, public :: capabilities => deep_kernel_capabilities
        procedure, public :: moments => deep_kernel_moments
        procedure, public :: adopt => deep_kernel_adopt
    end type fortbo_deep_kernel_posterior_t

contains

    pure integer function multi_task_n_inputs(self) result(n)
        class(fortbo_multi_task_posterior_t), intent(in) :: self

        n = self%dimension
    end function multi_task_n_inputs

    pure integer function multi_task_capabilities(self) result(caps)
        class(fortbo_multi_task_posterior_t), intent(in) :: self

        caps = FORTBO_CAP_MOMENTS
    end function multi_task_capabilities

    !! Adopt an already-fitted multi-output model and name the target output.
    !!
    !! The model is fitted by the caller through FortML's own interface rather
    !! than here, because a multi-output GP is configured with a coregionalization
    !! structure that has nothing to do with Bayesian optimization and that this
    !! module has no business inventing.
    subroutine multi_task_adopt(self, model, dimension, target_output, status)
        class(fortbo_multi_task_posterior_t), intent(out) :: self
        type(multi_output_gp_t), intent(in) :: model
        integer, intent(in) :: dimension
        integer, intent(in) :: target_output
        type(fortnum_status_t), intent(out) :: status

        if (dimension < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: the input width must be positive")
            return
        end if
        if (.not. model%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: the model must be fitted before it is adopted")
            return
        end if
        if (target_output < 1 .or. target_output > model%n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: the target output is out of range")
            return
        end if

        self%model = model
        self%dimension = dimension
        self%target_output = target_output
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_task_adopt

    !! Posterior mean and variance of the target output.
    subroutine multi_task_moments(self, points, mean, variance, status)
        class(fortbo_multi_task_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: all_means(:, :), covariance(:, :)
        integer :: m, p, k, index

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: moments before the model was adopted")
            return
        end if
        m = size(points, 1)
        if (size(points, 2) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: query points are the wrong width")
            return
        end if
        if (size(mean) /= m .or. size(variance) /= m) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-task: the moment buffers do not match the query")
            return
        end if

        p = self%model%n_outputs
        allocate (all_means(m, p), covariance(m*p, m*p))
        call self%model%predict(points, all_means, status)
        if (status%code /= FORTNUM_OK) return
        call self%model%predict_covariance(points, covariance, status)
        if (status%code /= FORTNUM_OK) return

        ! The joint covariance is ordered with outputs fastest, so the entry
        ! for query k and output j sits at (k - 1)*p + j. Getting this stride
        ! backwards produces a plausible-looking variance belonging to another
        ! task, which is why FortML's own suite checks the diagonal against the
        ! reported marginals rather than only checking symmetry.
        do k = 1, m
            index = (k - 1)*p + self%target_output
            mean(k) = all_means(k, self%target_output)
            variance(k) = max(covariance(index, index), 0.0_dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_task_moments

    pure integer function deep_kernel_n_inputs(self) result(n)
        class(fortbo_deep_kernel_posterior_t), intent(in) :: self

        n = self%dimension
    end function deep_kernel_n_inputs

    pure integer function deep_kernel_capabilities(self) result(caps)
        class(fortbo_deep_kernel_posterior_t), intent(in) :: self

        caps = FORTBO_CAP_MOMENTS
    end function deep_kernel_capabilities

    !! Adopt an already-fitted deep-kernel model.
    !!
    !! Fitted by the caller for the same reason as the multi-task case, and
    !! more strongly: training a deep kernel means optimizing network weights
    !! jointly with the base kernel's hyperparameters through the marginal
    !! likelihood, which is a training loop with its own optimizer, schedule
    !! and stopping rule. Burying that behind an adapter would hide the part
    !! most in need of the caller's attention.
    subroutine deep_kernel_adopt(self, model, status)
        class(fortbo_deep_kernel_posterior_t), intent(out) :: self
        type(deep_kernel_gp_t), intent(in) :: model
        type(fortnum_status_t), intent(out) :: status

        if (.not. model%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo deep kernel: the model must be fitted before it is adopted")
            return
        end if

        self%model = model
        self%dimension = model%input_dimension
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine deep_kernel_adopt

    subroutine deep_kernel_moments(self, points, mean, variance, status)
        class(fortbo_deep_kernel_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: model_mean(:, :)
        integer :: m

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo deep kernel: moments before the model was adopted")
            return
        end if
        m = size(points, 1)
        if (size(points, 2) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo deep kernel: query points are the wrong width")
            return
        end if
        if (size(mean) /= m .or. size(variance) /= m) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo deep kernel: the moment buffers do not match the query")
            return
        end if

        allocate (model_mean(m, 1))
        call self%model%predict(points, model_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = model_mean(:, 1)
        variance = max(variance, 0.0_dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine deep_kernel_moments

end module fortbo_structured
