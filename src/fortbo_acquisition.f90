module fortbo_acquisition
    !! Analytic acquisition functions (ROADMAP BO1).
    !!
    !! Convention. FortBO minimizes the objective and **maximizes** the
    !! acquisition. A candidate optimizer that minimizes therefore works on the
    !! negated acquisition, and it is that optimizer's job to negate, not this
    !! module's. Every acquisition here is nonnegative except the confidence
    !! bound, which inherits the objective's scale.
    !!
    !! Provenance. The closed-form values and their first-order products are not
    !! written here. They come from `src/generated`, derived and emitted by
    !! FortSym from the definition of the Gaussian CDF and PDF; this module owns
    !! only the boundaries the symbolic form cannot express — the deterministic
    !! limit at zero posterior variance, and the numerically hostile tail where
    !! the naive expression cancels.
    !!
    !! Derivative observations. Nothing in this module knows or cares whether
    !! the posterior it was handed was conditioned on values alone or on values
    !! and gradients. Every acquisition here becomes derivative-informed the
    !! moment the surrogate is, with no change on this side of the boundary.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_MOMENT_GRADIENT
    implicit none
    private

    !! Below this posterior standard deviation the point is treated as
    !! deterministic. The generated leaf divides by sigma, so the limit has to
    !! be taken here rather than evaluated there.
    real(dp), parameter, public :: FORTBO_SIGMA_FLOOR = 1.0e-300_dp

    !! Standard-normal z below which log expected improvement switches to its
    !! asymptotic branch. At z = -6 the naive difference has already lost most
    !! of its significant digits in double precision.
    real(dp), parameter :: LOG_EI_ASYMPTOTIC_Z = -6.0_dp

    public :: fortbo_acquisition_t
    public :: fortbo_ei_t
    public :: fortbo_log_ei_t
    public :: fortbo_pi_t
    public :: fortbo_ucb_t
    public :: fortbo_expected_improvement
    public :: fortbo_log_expected_improvement
    public :: fortbo_probability_of_improvement

    type, abstract :: fortbo_acquisition_t
        !! Incumbent objective value the acquisition improves upon.
        real(dp) :: best = 0.0_dp
        !! Exploration offset. The improvement threshold is `best - xi`.
        real(dp) :: xi = 0.0_dp
    contains
        procedure(fortbo_acquisition_value_interface), deferred, public :: value
        procedure, public :: value_gradient => acquisition_gradient_refuse
        procedure(fortbo_acquisition_name_interface), deferred, public :: name
    end type fortbo_acquisition_t

    abstract interface
        !! Acquisition value at each query row of `points`.
        subroutine fortbo_acquisition_value_interface(self, posterior, points, &
                values, status)
            import :: dp, fortbo_acquisition_t, fortbo_posterior_t, fortnum_status_t
            class(fortbo_acquisition_t), intent(in) :: self
            class(fortbo_posterior_t), intent(in) :: posterior
            real(dp), intent(in) :: points(:, :)
            real(dp), intent(out) :: values(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine fortbo_acquisition_value_interface

        pure function fortbo_acquisition_name_interface(self) result(name)
            import :: fortbo_acquisition_t
            class(fortbo_acquisition_t), intent(in) :: self
            character(len=:), allocatable :: name
        end function fortbo_acquisition_name_interface
    end interface

    type, extends(fortbo_acquisition_t) :: fortbo_ei_t
    contains
        procedure, public :: value => ei_value
        procedure, public :: value_gradient => ei_gradient
        procedure, public :: name => ei_name
    end type fortbo_ei_t

    type, extends(fortbo_acquisition_t) :: fortbo_log_ei_t
    contains
        procedure, public :: value => log_ei_value
        procedure, public :: name => log_ei_name
    end type fortbo_log_ei_t

    type, extends(fortbo_acquisition_t) :: fortbo_pi_t
    contains
        procedure, public :: value => pi_value_at
        procedure, public :: value_gradient => pi_gradient
        procedure, public :: name => pi_name
    end type fortbo_pi_t

    type, extends(fortbo_acquisition_t) :: fortbo_ucb_t
        !! Exploration weight. The value returned is `-(mu - beta*sigma)`, so
        !! maximizing it minimizes the lower confidence bound of the objective.
        real(dp) :: beta = 2.0_dp
    contains
        procedure, public :: value => ucb_value
        procedure, public :: value_gradient => ucb_gradient
        procedure, public :: name => ucb_name
    end type fortbo_ucb_t

    interface
        pure subroutine fortbo_generated_acquisition_leaf(mu, sigma, best, xi, ei, &
                ei_d_mu, ei_d_sigma, pi_value, &
                pi_d_mu, pi_d_sigma)
            import :: dp
            real(dp), intent(in) :: mu, sigma, best, xi
            real(dp), intent(out) :: ei, ei_d_mu, ei_d_sigma
            real(dp), intent(out) :: pi_value, pi_d_mu, pi_d_sigma
        end subroutine fortbo_generated_acquisition_leaf
    end interface

contains

    !! Scalar expected improvement for a minimized objective. At zero variance
    !! the improvement is deterministic and equals the shortfall, which is the
    !! limit of the Gaussian expression rather than a special case bolted on.
    pure subroutine fortbo_expected_improvement(mean, standard_deviation, best, xi, &
            value, d_mean, d_sd)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: standard_deviation
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: value
        real(dp), intent(out), optional :: d_mean
        real(dp), intent(out), optional :: d_sd
        real(dp) :: ei, ei_d_mu, ei_d_sigma, pi_v, pi_d_mu, pi_d_sigma
        real(dp) :: threshold

        threshold = best - xi
        if (standard_deviation <= FORTBO_SIGMA_FLOOR) then
            value = max(threshold - mean, 0.0_dp)
            if (present(d_mean)) then
                d_mean = 0.0_dp
                if (threshold > mean) d_mean = -1.0_dp
            end if
            ! Widening a collapsed posterior can only add improvement mass, but
            ! the one-sided limit is not a derivative; report zero and let the
            ! caller's line search discover the kink rather than trusting it.
            if (present(d_sd)) d_sd = 0.0_dp
            return
        end if
        call fortbo_generated_acquisition_leaf(mean, standard_deviation, best, xi, &
            ei, ei_d_mu, ei_d_sigma, pi_v, &
            pi_d_mu, pi_d_sigma)
        value = max(ei, 0.0_dp)
        if (present(d_mean)) d_mean = ei_d_mu
        if (present(d_sd)) d_sd = ei_d_sigma
    end subroutine fortbo_expected_improvement

    !! Logarithm of expected improvement, evaluated so that it stays finite and
    !! accurate far into the tail where expected improvement itself underflows.
    !! For z well below zero the naive form is the difference of two nearly
    !! equal quantities; the asymptotic branch uses the Mills-ratio expansion
    !! instead, which is accurate exactly where the naive form is not.
    pure subroutine fortbo_log_expected_improvement(mean, standard_deviation, best, &
            xi, value)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: standard_deviation
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: value
        real(dp) :: threshold, z, ei, series, z_squared

        threshold = best - xi
        if (standard_deviation <= FORTBO_SIGMA_FLOOR) then
            if (threshold > mean) then
                value = log(threshold - mean)
            else
                value = -huge(1.0_dp)
            end if
            return
        end if

        z = (threshold - mean)/standard_deviation
        if (z > LOG_EI_ASYMPTOTIC_Z) then
            call fortbo_expected_improvement(mean, standard_deviation, best, xi, ei)
            if (ei <= 0.0_dp) then
                value = -huge(1.0_dp)
                return
            end if
            value = log(ei)
            return
        end if

        ! Asymptotic branch. With phi the standard normal density,
        !     EI = sigma [ z Phi(z) + phi(z) ]  and  Phi(z) ~ phi(z) (1/(-z))
        !     (1 - 1/z^2 + 3/z^4 - 15/z^6 + ...) as z -> -infinity,
        ! so EI ~ sigma phi(z) (1/z^2)(1 - 3/z^2 + 15/z^4 - ...). Taking the
        ! logarithm keeps log phi(z) = -z^2/2 - log(2 pi)/2 exact and leaves
        ! only the well-conditioned series to evaluate.
        z_squared = z*z
        series = 1.0_dp - 3.0_dp/z_squared + 15.0_dp/z_squared**2 &
            - 105.0_dp/z_squared**3
        value = log(standard_deviation) - 0.5_dp*z_squared &
            - 0.5_dp*log(8.0_dp*atan(1.0_dp)) - log(z_squared) + log(series)
    end subroutine fortbo_log_expected_improvement

    !! Probability that the objective improves on the threshold.
    pure subroutine fortbo_probability_of_improvement(mean, standard_deviation, best, &
            xi, value, d_mean, d_sd)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: standard_deviation
        real(dp), intent(in) :: best
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: value
        real(dp), intent(out), optional :: d_mean
        real(dp), intent(out), optional :: d_sd
        real(dp) :: ei, ei_d_mu, ei_d_sigma, pi_v, pi_d_mu, pi_d_sigma
        real(dp) :: threshold

        threshold = best - xi
        if (standard_deviation <= FORTBO_SIGMA_FLOOR) then
            value = 0.0_dp
            if (threshold > mean) value = 1.0_dp
            if (present(d_mean)) d_mean = 0.0_dp
            if (present(d_sd)) d_sd = 0.0_dp
            return
        end if
        call fortbo_generated_acquisition_leaf(mean, standard_deviation, best, xi, &
            ei, ei_d_mu, ei_d_sigma, pi_v, &
            pi_d_mu, pi_d_sigma)
        value = min(max(pi_v, 0.0_dp), 1.0_dp)
        if (present(d_mean)) d_mean = pi_d_mu
        if (present(d_sd)) d_sd = pi_d_sigma
    end subroutine fortbo_probability_of_improvement

    subroutine ei_value(self, posterior, points, values, status)
        class(fortbo_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        call moments_or_refuse(posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            call fortbo_expected_improvement(mean(i), sqrt(variance(i)), self%best, &
                self%xi, values(i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ei_value

    !! Gradient of expected improvement with respect to the query coordinates,
    !! by the chain rule through the posterior moments. The posterior must
    !! declare moment gradients; otherwise this refuses rather than differencing.
    subroutine ei_gradient(self, posterior, points, gradient, status)
        class(fortbo_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        real(dp) :: value, d_mean, d_sd
        integer :: i

        call chain_rule_setup(posterior, points, mean, variance, mean_gradient, &
            sd_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            call fortbo_expected_improvement(mean(i), sqrt(variance(i)), self%best, &
                self%xi, value, d_mean, d_sd)
            gradient(i, :) = d_mean*mean_gradient(i, :) + d_sd*sd_gradient(i, :)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ei_gradient

    pure function ei_name(self) result(name)
        class(fortbo_ei_t), intent(in) :: self
        character(len=:), allocatable :: name

        name = "expected_improvement"
    end function ei_name

    subroutine log_ei_value(self, posterior, points, values, status)
        class(fortbo_log_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        call moments_or_refuse(posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            call fortbo_log_expected_improvement(mean(i), sqrt(variance(i)), &
                self%best, self%xi, values(i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine log_ei_value

    pure function log_ei_name(self) result(name)
        class(fortbo_log_ei_t), intent(in) :: self
        character(len=:), allocatable :: name

        name = "log_expected_improvement"
    end function log_ei_name

    subroutine pi_value_at(self, posterior, points, values, status)
        class(fortbo_pi_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        call moments_or_refuse(posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            call fortbo_probability_of_improvement(mean(i), sqrt(variance(i)), &
                self%best, self%xi, values(i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pi_value_at

    subroutine pi_gradient(self, posterior, points, gradient, status)
        class(fortbo_pi_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        real(dp) :: value, d_mean, d_sd
        integer :: i

        call chain_rule_setup(posterior, points, mean, variance, mean_gradient, &
            sd_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            call fortbo_probability_of_improvement(mean(i), sqrt(variance(i)), &
                self%best, self%xi, value, &
                d_mean, d_sd)
            gradient(i, :) = d_mean*mean_gradient(i, :) + d_sd*sd_gradient(i, :)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pi_gradient

    pure function pi_name(self) result(name)
        class(fortbo_pi_t), intent(in) :: self
        character(len=:), allocatable :: name

        name = "probability_of_improvement"
    end function pi_name

    subroutine ucb_value(self, posterior, points, values, status)
        class(fortbo_ucb_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)

        if (self%beta < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo acquisition: confidence weight must not be negative")
            return
        end if
        call moments_or_refuse(posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        values = -(mean - self%beta*sqrt(variance))
        call status_set(status, FORTNUM_OK, "")
    end subroutine ucb_value

    subroutine ucb_gradient(self, posterior, points, gradient, status)
        class(fortbo_ucb_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        integer :: i

        call chain_rule_setup(posterior, points, mean, variance, mean_gradient, &
            sd_gradient, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(points, 1)
            gradient(i, :) = -mean_gradient(i, :) + self%beta*sd_gradient(i, :)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ucb_gradient

    pure function ucb_name(self) result(name)
        class(fortbo_ucb_t), intent(in) :: self
        character(len=:), allocatable :: name

        name = "lower_confidence_bound"
    end function ucb_name

    !! Default gradient behavior: refuse by name. An acquisition without a
    !! closed-form derivative must say so rather than let a caller assume one.
    subroutine acquisition_gradient_refuse(self, posterior, points, gradient, status)
        class(fortbo_acquisition_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "fortbo acquisition: "//self%name()// &
            " does not provide an input gradient")
    end subroutine acquisition_gradient_refuse

    subroutine moments_or_refuse(posterior, points, mean, variance, status)
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out), allocatable :: mean(:)
        real(dp), intent(out), allocatable :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        allocate (mean(size(points, 1)), variance(size(points, 1)))
        if (.not. posterior%supports(FORTBO_CAP_MOMENTS)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo acquisition: surrogate does not implement moments")
            return
        end if
        call posterior%moments(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(variance < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo acquisition: negative posterior variance")
            return
        end if
    end subroutine moments_or_refuse

    subroutine chain_rule_setup(posterior, points, mean, variance, mean_gradient, &
            sd_gradient, status)
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out), allocatable :: mean(:), variance(:)
        real(dp), intent(out), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        type(fortnum_status_t), intent(out) :: status

        allocate (mean_gradient(size(points, 1), size(points, 2)))
        allocate (sd_gradient(size(points, 1), size(points, 2)))
        call moments_or_refuse(posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. posterior%supports(FORTBO_CAP_MOMENT_GRADIENT)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo acquisition: surrogate does not implement moment_gradient")
            return
        end if
        call posterior%moment_gradient(points, mean_gradient, sd_gradient, status)
    end subroutine chain_rule_setup

end module fortbo_acquisition
