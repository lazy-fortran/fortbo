module fortbo_monte_carlo
    !! Monte Carlo acquisition evaluation (ROADMAP BO1).
    !!
    !! Every acquisition here is an expectation over posterior draws, estimated
    !! by averaging a utility over reparameterized samples
    !!
    !!     y_s(x) = mu(x) + sigma(x) * b_s
    !!
    !! where the standard normal base draws `b_s` are generated once and then
    !! **held fixed**. That single decision buys three things at once:
    !!
    !!   * common random numbers. Two candidate sets evaluated against the same
    !!     base are compared on the same noise, so their difference is estimated
    !!     far more precisely than either value is. An optimizer that resamples
    !!     every evaluation is optimizing a different function each step and
    !!     will chatter;
    !!   * a pathwise derivative. With `b_s` fixed the map from `x` to `y_s` is
    !!     deterministic and differentiable, so the gradient of the estimate is
    !!     the estimate of the gradient. No score-function estimator, no
    !!     differencing;
    !!   * exact replay. The same seed reproduces the same estimate bit for bit.
    !!
    !! Antithetic pairing draws half the samples and negates them for the other
    !! half. For any utility with a monotone component — and improvement
    !! utilities are monotone in `y` — the paired estimate has strictly lower
    !! variance at the same cost.
    !!
    !! Scope. These are the marginal, per-point estimators: each query row is
    !! its own expectation. The batch estimators, where the utility couples the
    !! rows through the joint covariance, are BO1's batch item and live beside
    !! these rather than inside them.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_normal
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_acquisition, only: fortbo_acquisition_t
    implicit none
    private

    public :: fortbo_mc_base_t
    public :: fortbo_mc_ei_t
    public :: fortbo_mc_pi_t
    public :: fortbo_mc_noisy_ei_t

    type :: fortbo_mc_base_t
        !! Fixed standard normal base draws, shaped (n_points, n_samples).
        real(dp), allocatable :: draws(:, :)
        logical :: antithetic = .false.
    contains
        procedure, public :: generate => base_generate
        procedure, public :: n_points => base_n_points
        procedure, public :: n_samples => base_n_samples
    end type fortbo_mc_base_t

    type, extends(fortbo_acquisition_t) :: fortbo_mc_ei_t
        type(fortbo_mc_base_t) :: base
    contains
        procedure, public :: value => mc_ei_value
        procedure, public :: value_gradient => mc_ei_gradient
        procedure, public :: name => mc_ei_name
    end type fortbo_mc_ei_t

    type, extends(fortbo_acquisition_t) :: fortbo_mc_pi_t
        type(fortbo_mc_base_t) :: base
    contains
        procedure, public :: value => mc_pi_value
        procedure, public :: name => mc_pi_name
    end type fortbo_mc_pi_t

    !! Noisy expected improvement.
    !!
    !! Plain EI compares against `best`, the smallest value ever *observed*.
    !! Under noise that number is not a property of the objective at all: it is
    !! the minimum of a sample, so it is biased low by roughly the noise scale
    !! and gets worse the more points are evaluated. EI measured against it
    !! shrinks toward zero everywhere as a run proceeds, which is exactly the
    !! failure people report as "EI stops exploring".
    !!
    !! Noisy EI removes the observed value from the comparison entirely. The
    !! incumbent is the posterior's own belief about the best *latent* value at
    !! the points already evaluated, and it is re-drawn inside each sample:
    !!
    !!     NEI(x) = E[ max(min_j f_j - f(x), 0) ]
    !!
    !! where `f_j` are the latent values at the observed inputs and `f(x)` the
    !! latent value at the candidate, all under one joint posterior draw. Using
    !! a joint draw is what makes the comparison honest — a sample in which the
    !! observed points happen to be good must be the same sample in which the
    !! candidate is judged, or the estimator credits the candidate for
    !! randomness the incumbent also enjoyed.
    !!
    !! `observed_mean` and `observed_sd` are supplied by the caller rather than
    !! recomputed here, because the caller already holds them from the fit and
    !! because re-querying would let the two drift apart across a refit.
    type, extends(fortbo_acquisition_t) :: fortbo_mc_noisy_ei_t
        type(fortbo_mc_base_t) :: base
        !! Posterior moments at the already-evaluated inputs.
        real(dp), allocatable :: observed_mean(:)
        real(dp), allocatable :: observed_sd(:)
        !! Independent base draws for the observed points, shaped
        !! (n_observed, n_samples). Frozen alongside `base` so the whole
        !! acquisition replays from one seed.
        real(dp), allocatable :: observed_draws(:, :)
    contains
        procedure, public :: value => mc_noisy_ei_value
        procedure, public :: name => mc_noisy_ei_name
    end type fortbo_mc_noisy_ei_t

contains

    !! Draw and freeze the base samples. With `antithetic` the sample count must
    !! be even, because an unpaired draw would break the symmetry the estimator
    !! relies on and silently reintroduce the variance it was meant to remove.
    subroutine base_generate(self, n_points, n_samples, generator, status, antithetic)
        class(fortbo_mc_base_t), intent(out) :: self
        integer, intent(in) :: n_points
        integer, intent(in) :: n_samples
        type(rng_t), intent(inout) :: generator
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: antithetic
        integer :: i, s, half

        if (n_points < 1 .or. n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo monte carlo: sample shape must be positive")
            return
        end if
        self%antithetic = .false.
        if (present(antithetic)) self%antithetic = antithetic
        if (self%antithetic .and. mod(n_samples, 2) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo monte carlo: antithetic sampling needs an even count")
            return
        end if

        allocate (self%draws(n_points, n_samples))
        if (self%antithetic) then
            half = n_samples/2
            do s = 1, half
                do i = 1, n_points
                    call rng_normal(generator, self%draws(i, s))
                end do
                self%draws(:, half + s) = -self%draws(:, s)
            end do
        else
            do s = 1, n_samples
                do i = 1, n_points
                    call rng_normal(generator, self%draws(i, s))
                end do
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine base_generate

    pure integer function base_n_points(self) result(n)
        class(fortbo_mc_base_t), intent(in) :: self

        n = 0
        if (allocated(self%draws)) n = size(self%draws, 1)
    end function base_n_points

    pure integer function base_n_samples(self) result(n)
        class(fortbo_mc_base_t), intent(in) :: self

        n = 0
        if (allocated(self%draws)) n = size(self%draws, 2)
    end function base_n_samples

    subroutine mc_ei_value(self, posterior, points, values, status)
        class(fortbo_mc_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp) :: threshold, sample, total
        integer :: i, s, n_samples

        call marginal_setup(self%base, posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        threshold = self%best - self%xi
        n_samples = self%base%n_samples()
        do i = 1, size(points, 1)
            total = 0.0_dp
            do s = 1, n_samples
                sample = mean(i) + sqrt(variance(i))*self%base%draws(i, s)
                total = total + max(threshold - sample, 0.0_dp)
            end do
            values(i) = total/real(n_samples, dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mc_ei_value

    !! Pathwise gradient. The improvement utility is piecewise linear with a
    !! kink at the threshold; the kink is hit with probability zero for a
    !! continuous posterior, so differentiating inside the expectation is valid
    !! and the estimator is unbiased.
    subroutine mc_ei_gradient(self, posterior, points, gradient, status)
        class(fortbo_mc_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        real(dp), allocatable :: accumulator(:)
        real(dp) :: threshold, sample, standard_deviation
        integer :: i, s, n_samples

        call marginal_setup(self%base, posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. posterior%supports(FORTBO_CAP_MOMENT_GRADIENT)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo monte carlo: surrogate does not implement moment_gradient")
            return
        end if
        allocate (mean_gradient(size(points, 1), size(points, 2)))
        allocate (sd_gradient(size(points, 1), size(points, 2)))
        allocate (accumulator(size(points, 2)))
        call posterior%moment_gradient(points, mean_gradient, sd_gradient, status)
        if (status%code /= FORTNUM_OK) return

        threshold = self%best - self%xi
        n_samples = self%base%n_samples()
        do i = 1, size(points, 1)
            standard_deviation = sqrt(variance(i))
            accumulator = 0.0_dp
            do s = 1, n_samples
                sample = mean(i) + standard_deviation*self%base%draws(i, s)
                if (sample >= threshold) cycle
                accumulator = accumulator - (mean_gradient(i, :) &
                    + self%base%draws(i, s)*sd_gradient(i, :))
            end do
            gradient(i, :) = accumulator/real(n_samples, dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mc_ei_gradient

    pure function mc_ei_name(self) result(name)
        class(fortbo_mc_ei_t), intent(in) :: self
        character(len=:), allocatable :: name

        if (self%base%antithetic) then
            name = "mc_expected_improvement_antithetic"
        else
            name = "mc_expected_improvement"
        end if
    end function mc_ei_name

    subroutine mc_pi_value(self, posterior, points, values, status)
        class(fortbo_mc_pi_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp) :: threshold, sample
        integer :: i, s, n_samples, hits

        call marginal_setup(self%base, posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        threshold = self%best - self%xi
        n_samples = self%base%n_samples()
        do i = 1, size(points, 1)
            hits = 0
            do s = 1, n_samples
                sample = mean(i) + sqrt(variance(i))*self%base%draws(i, s)
                if (sample < threshold) hits = hits + 1
            end do
            values(i) = real(hits, dp)/real(n_samples, dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mc_pi_value

    pure function mc_pi_name(self) result(name)
        class(fortbo_mc_pi_t), intent(in) :: self
        character(len=:), allocatable :: name

        if (self%base%antithetic) then
            name = "mc_probability_of_improvement_antithetic"
        else
            name = "mc_probability_of_improvement"
        end if
    end function mc_pi_name

    !! Shared preconditions: base samples exist, their row count matches the
    !! query, and the surrogate provides moments. A mismatched base is refused
    !! rather than broadcast, because reusing one row's noise across every query
    !! would correlate candidates that ought to be independent.
    subroutine marginal_setup(base, posterior, points, mean, variance, status)
        type(fortbo_mc_base_t), intent(in) :: base
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out), allocatable :: mean(:)
        real(dp), intent(out), allocatable :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        allocate (mean(size(points, 1)), variance(size(points, 1)))
        if (.not. allocated(base%draws)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo monte carlo: base samples were never generated")
            return
        end if
        if (base%n_points() /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo monte carlo: base sample rows do not match the query")
            return
        end if
        if (.not. posterior%supports(FORTBO_CAP_MOMENTS)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo monte carlo: surrogate does not implement moments")
            return
        end if
        call posterior%moments(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(variance < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo monte carlo: negative posterior variance")
        end if
    end subroutine marginal_setup


    !! One joint draw per sample: the observed latent values and the candidate's
    !! latent value share the sample index, so the incumbent varies with the
    !! draw exactly as the candidate does.
    subroutine mc_noisy_ei_value(self, posterior, points, values, status)
        class(fortbo_mc_noisy_ei_t), intent(in) :: self
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp) :: sample, incumbent, total
        integer :: i, s, j, n_samples, n_observed

        values = 0.0_dp
        if (.not. allocated(self%observed_mean) .or. &
            .not. allocated(self%observed_sd) .or. &
            .not. allocated(self%observed_draws)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mc noisy ei: observed moments have not been supplied")
            return
        end if
        n_observed = size(self%observed_mean)
        if (n_observed < 1 .or. size(self%observed_sd) /= n_observed .or. &
            size(self%observed_draws, 1) /= n_observed) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mc noisy ei: observed arrays disagree")
            return
        end if
        if (any(self%observed_sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mc noisy ei: observed standard deviations must not be negative")
            return
        end if

        call marginal_setup(self%base, posterior, points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        n_samples = self%base%n_samples()
        if (size(self%observed_draws, 2) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mc noisy ei: observed draws must match the sample count")
            return
        end if

        do i = 1, size(points, 1)
            total = 0.0_dp
            do s = 1, n_samples
                incumbent = huge(1.0_dp)
                do j = 1, n_observed
                    incumbent = min(incumbent, self%observed_mean(j) &
                        + self%observed_sd(j)*self%observed_draws(j, s))
                end do
                sample = mean(i) + sqrt(variance(i))*self%base%draws(i, s)
                total = total + max(incumbent - sample, 0.0_dp)
            end do
            values(i) = total/real(n_samples, dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mc_noisy_ei_value

    pure function mc_noisy_ei_name(self) result(name)
        class(fortbo_mc_noisy_ei_t), intent(in) :: self
        character(len=:), allocatable :: name

        name = "mc_noisy_ei"
    end function mc_noisy_ei_name

end module fortbo_monte_carlo
