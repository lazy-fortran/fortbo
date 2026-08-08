module fortbo_linear_posterior
    !! Bayesian linear regression as a posterior provider (ROADMAP BO2).
    !!
    !! This module exists to demonstrate — and to keep testable — the claim that
    !! FortBO's posterior contract does not require a Gaussian process. Nothing
    !! above `fortbo_posterior_t` knows what produced the moments it consumes,
    !! and the way to keep that true is to have a second, structurally different
    !! provider in the tree that every acquisition is exercised against.
    !!
    !! A Bayesian linear model on a fixed feature map is the right second
    !! provider precisely because it is *not* a GP in the way that matters here:
    !! its posterior has a finite-dimensional parameter, its cost is independent
    !! of the number of observations once fitted, and its covariance has rank at
    !! most the feature count regardless of how many query points are asked
    !! about. An acquisition that quietly assumed full-rank joint covariance
    !! would work against a GP and fail here, which is exactly the assumption
    !! worth catching.
    !!
    !! With prior `w ~ N(0, alpha^-1 I)` and noise variance `beta^-1`, the
    !! posterior over weights after observing `Phi` and `y` is Gaussian with
    !!
    !!     S^-1 = alpha I + beta Phi^T Phi,   m = beta S Phi^T y,
    !!
    !! and the predictive moments at a feature row `phi` are `phi.m` and
    !! `phi^T S phi`. Both are exact — no approximation, no sampling — which
    !! makes this provider a useful oracle in its own right: on a genuinely
    !! linear objective the posterior mean is the least-squares fit and an
    !! acquisition's behaviour can be reasoned about by hand.
    !!
    !! The predictive variance here is the variance of the *latent* function,
    !! excluding observation noise, matching what the GP adapters report.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_normal
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_MOMENT_GRADIENT
    implicit none
    private

    public :: fortbo_linear_posterior_t

    !! The feature map. A caller supplies one; the polynomial default below is
    !! only what the tests use.
    abstract interface
        pure subroutine fortbo_feature_map_interface(point, features)
            import :: dp
            real(dp), intent(in) :: point(:)
            real(dp), intent(out) :: features(:)
        end subroutine fortbo_feature_map_interface
    end interface

    type, extends(fortbo_posterior_t) :: fortbo_linear_posterior_t
        integer :: dimension = 0
        integer :: n_features = 0
        logical :: fitted = .false.
        real(dp) :: noise_precision = 1.0e6_dp
        !! Posterior mean and covariance over the weights.
        real(dp), allocatable :: weight_mean(:)
        real(dp), allocatable :: weight_covariance(:, :)
        procedure(fortbo_feature_map_interface), pointer, nopass :: features => null()
    contains
        procedure, public :: n_inputs => linear_n_inputs
        procedure, public :: capabilities => linear_capabilities
        procedure, public :: moments => linear_moments
        procedure, public :: covariance => linear_covariance
        procedure, public :: joint_sample => linear_joint_sample
        procedure, public :: fit => linear_fit
    end type fortbo_linear_posterior_t

contains

    pure integer function linear_n_inputs(self) result(n)
        class(fortbo_linear_posterior_t), intent(in) :: self

        n = self%dimension
    end function linear_n_inputs

    !! Moments, covariance, and joint samples — the same bits a GP declares for
    !! these operations, which is the point. Moment *gradients* are not declared:
    !! they would need the feature map's Jacobian, and a caller that supplies
    !! only a value map must not have one invented for it.
    pure integer function linear_capabilities(self) result(caps)
        class(fortbo_linear_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) then
            caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_COVARIANCE &
                + FORTBO_CAP_JOINT_SAMPLE
        end if
    end function linear_capabilities

    !! Exact conjugate update. No iteration and no approximation: the posterior
    !! over weights is Gaussian and its two parameters are closed forms.
    subroutine linear_fit(self, inputs, targets, feature_map, n_features, &
            prior_precision, noise_precision, status)
        class(fortbo_linear_posterior_t), intent(inout) :: self
        real(dp), intent(in) :: inputs(:, :)
        real(dp), intent(in) :: targets(:)
        procedure(fortbo_feature_map_interface) :: feature_map
        integer, intent(in) :: n_features
        real(dp), intent(in) :: prior_precision
        real(dp), intent(in) :: noise_precision
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: design(:, :), precision(:, :), identity(:, :)
        real(dp), allocatable :: row(:), rhs(:)
        integer :: n, i, j

        self%fitted = .false.
        n = size(inputs, 1)
        if (n < 1 .or. size(targets) /= n .or. size(inputs, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: input and target shapes disagree")
            return
        end if
        if (n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: at least one feature is required")
            return
        end if
        if (prior_precision <= 0.0_dp .or. noise_precision <= 0.0_dp) then
            ! A zero prior precision is an improper prior, whose posterior is
            ! singular whenever the design is rank deficient. Refusing keeps
            ! the failure at the point where the choice was made.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: precisions must be positive")
            return
        end if

        self%dimension = size(inputs, 2)
        self%n_features = n_features
        self%noise_precision = noise_precision
        self%features => feature_map

        allocate (design(n, n_features), row(n_features))
        do i = 1, n
            call feature_map(inputs(i, :), row)
            design(i, :) = row
        end do

        allocate (precision(n_features, n_features))
        allocate (identity(n_features, n_features))
        identity = 0.0_dp
        do j = 1, n_features
            identity(j, j) = 1.0_dp
        end do
        precision = prior_precision*identity &
            + noise_precision*matmul(transpose(design), design)

        ! Invert by solving against the identity: the posterior covariance is
        ! needed in full, since every predictive variance is a quadratic form
        ! in it.
        call factorization%factorize(precision, status)
        if (status%code /= FORTNUM_OK) return
        allocate (self%weight_covariance(n_features, n_features))
        self%weight_covariance = identity
        call factorization%solve(self%weight_covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate (rhs(n_features), self%weight_mean(n_features))
        rhs = noise_precision*matmul(transpose(design), targets)
        self%weight_mean = matmul(self%weight_covariance, rhs)

        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_fit

    subroutine linear_moments(self, points, mean, variance, status)
        class(fortbo_linear_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: row(:)
        integer :: i

        mean = 0.0_dp
        variance = 0.0_dp
        call check_query(self, points, size(mean), status)
        if (status%code /= FORTNUM_OK) return
        if (size(variance) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: variance width does not match the query")
            return
        end if

        allocate (row(self%n_features))
        do i = 1, size(points, 1)
            call self%features(points(i, :), row)
            mean(i) = dot_product(row, self%weight_mean)
            variance(i) = dot_product(row, matmul(self%weight_covariance, row))
            if (variance(i) < 0.0_dp) variance(i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_moments

    !! `Phi S Phi^T`, which has rank at most the feature count however many
    !! query points are asked about. That deficiency is real and is why joint
    !! sampling below adds a jitter before factorizing.
    subroutine linear_covariance(self, points, covariance, status)
        class(fortbo_linear_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), row(:)
        integer :: n, i

        covariance = 0.0_dp
        n = size(points, 1)
        call check_query(self, points, size(covariance, 1), status)
        if (status%code /= FORTNUM_OK) return
        if (size(covariance, 2) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: covariance must be square in the query")
            return
        end if

        allocate (design(n, self%n_features), row(self%n_features))
        do i = 1, n
            call self%features(points(i, :), row)
            design(i, :) = row
        end do
        covariance = matmul(design, matmul(self%weight_covariance, &
            transpose(design)))
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_covariance

    !! Joint samples drawn in *weight* space rather than in function space.
    !!
    !! Drawing `w` once and evaluating `phi.w` at every query point gives an
    !! exactly consistent joint realization at the cost of the feature count,
    !! not the query count, and it sidesteps the rank deficiency entirely — the
    !! function-space covariance is singular by construction, so factorizing it
    !! would need a jitter whose size would then contaminate the samples.
    subroutine linear_joint_sample(self, points, generator, samples, status)
        class(fortbo_linear_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: factor(:, :), draw(:), weights(:), row(:)
        integer :: n, s, i, j

        samples = 0.0_dp
        n = size(points, 1)
        call check_query(self, points, size(samples, 1), status)
        if (status%code /= FORTNUM_OK) return

        call factorization%factorize(self%weight_covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate (factor, source=factorization%lower)

        allocate (draw(self%n_features), weights(self%n_features))
        allocate (row(self%n_features))
        do s = 1, size(samples, 2)
            do j = 1, self%n_features
                call rng_normal(generator, draw(j))
            end do
            weights = self%weight_mean + matmul(factor, draw)
            do i = 1, n
                call self%features(points(i, :), row)
                samples(i, s) = dot_product(row, weights)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_joint_sample

    pure subroutine check_query(self, points, n_out, status)
        class(fortbo_linear_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: n_out
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: model has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: query width does not match the model")
            return
        end if
        if (n_out /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo linear: output length does not match the query")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_query

end module fortbo_linear_posterior
