program test_linear_posterior
    !! BO2: a non-GP posterior provider.
    !!
    !! The claim under test is the contract's, not the model's: an acquisition
    !! must not care what produced its moments. So the same acquisition code
    !! that runs against the GP adapters is run against a Bayesian linear model
    !! here, unchanged, and is required to behave.
    !!
    !! Oracles:
    !!
    !!   * on a genuinely linear objective with tight noise, the posterior mean
    !!     must recover the true coefficients, which the test states rather than
    !!     fits;
    !!   * the predictive moments are checked against the quadratic forms they
    !!     are defined by, recomputed here from the weight posterior;
    !!   * joint samples are checked against the covariance they must reproduce,
    !!     which is the property that makes batch acquisitions valid;
    !!   * the rank deficiency is checked directly. This provider's function
    !!     space covariance is singular once there are more query points than
    !!     features, and an acquisition that assumed otherwise would pass
    !!     against a GP and fail here.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_posterior, only: FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_linear_posterior, only: fortbo_linear_posterior_t
    use fortbo_acquisition, only: fortbo_ei_t, fortbo_ucb_t
    use fortbo_batch, only: fortbo_batch_samples_t, fortbo_qei
    use fortbo_knowledge_gradient, only: fortbo_knowledge_gradient_value
    implicit none

    integer :: failures

    failures = 0
    call check_recovers_a_linear_objective(failures)
    call check_moments_match_their_definition(failures)
    call check_joint_samples_match_the_covariance(failures)
    call check_rank_deficiency_is_real(failures)
    call check_acquisitions_run_unchanged(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_linear_posterior: PASS"
    else
        print *, "test_linear_posterior: FAIL", failures
        error stop 1
    end if

contains

    !! Quadratic features in one input: [1, x, x^2].
    pure subroutine quadratic_features(point, features)
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: features(:)

        features(1) = 1.0_dp
        features(2) = point(1)
        features(3) = point(1)**2
    end subroutine quadratic_features

    !! The objective the features can represent exactly.
    pure real(dp) function truth(x) result(value)
        real(dp), intent(in) :: x

        value = 0.5_dp - 1.25_dp*x + 2.0_dp*x*x
    end function truth

    subroutine build_model(model, status)
        type(fortbo_linear_posterior_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: inputs(9, 1), targets(9)
        integer :: k

        do k = 1, 9
            inputs(k, 1) = -1.0_dp + 0.25_dp*real(k - 1, dp)
            targets(k) = truth(inputs(k, 1))
        end do
        call model%fit(inputs, targets, quadratic_features, 3, 1.0e-6_dp, &
            1.0e8_dp, status)
    end subroutine build_model

    !! The features span the objective, so with tight noise and a weak prior the
    !! posterior mean must be the true coefficients.
    subroutine check_recovers_a_linear_objective(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: expected(3)

        call build_model(model, status)
        call expect(status%code == FORTNUM_OK, "the linear model fits", failures)
        call expect(model%supports(FORTBO_CAP_MOMENTS) .and. &
            model%supports(FORTBO_CAP_COVARIANCE) .and. &
            model%supports(FORTBO_CAP_JOINT_SAMPLE), &
            "a non-GP provider declares the same capability bits", failures)
        call expect(.not. model%supports(FORTBO_CAP_MOMENT_GRADIENT), &
            "it declines the gradient it has no feature Jacobian for", failures)

        expected = [0.5_dp, -1.25_dp, 2.0_dp]
        call expect(maxval(abs(model%weight_mean - expected)) < 1.0e-6_dp, &
            "the posterior mean recovers the true coefficients", failures)
        call expect(model%n_inputs() == 1, "the provider reports its width", &
            failures)
    end subroutine check_recovers_a_linear_objective

    !! Against the quadratic forms the moments are defined by, recomputed here.
    subroutine check_moments_match_their_definition(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: points(5, 1), mean(5), variance(5), row(3)
        real(dp) :: expected_mean, expected_variance
        integer :: k
        logical :: matches

        call build_model(model, status)
        do k = 1, 5
            points(k, 1) = -0.8_dp + 0.4_dp*real(k - 1, dp)
        end do
        call model%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "moments evaluate", failures)

        matches = .true.
        do k = 1, 5
            call quadratic_features(points(k, :), row)
            expected_mean = dot_product(row, model%weight_mean)
            expected_variance = dot_product(row, &
                matmul(model%weight_covariance, row))
            if (abs(mean(k) - expected_mean) > 1.0e-12_dp) matches = .false.
            if (abs(variance(k) - expected_variance) > 1.0e-12_dp) matches = .false.
        end do
        call expect(matches, "the moments are the quadratic forms they claim", &
            failures)

        ! Predicting the objective it was fitted on must be accurate.
        matches = .true.
        do k = 1, 5
            if (abs(mean(k) - truth(points(k, 1))) > 1.0e-6_dp) matches = .false.
        end do
        call expect(matches, "the model predicts the objective it can represent", &
            failures)
        call expect(all(variance >= 0.0_dp), "variances are non-negative", failures)
    end subroutine check_moments_match_their_definition

    !! Joint samples must reproduce the covariance, since that is what makes a
    !! batch acquisition built on them valid.
    subroutine check_joint_samples_match_the_covariance(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_points = 4, n_samples = 200000
        real(dp) :: points(n_points, 1)
        real(dp) :: covariance(n_points, n_points)
        real(dp) :: samples(n_points, 1)
        real(dp) :: totals(n_points), cross(n_points, n_points)
        real(dp) :: empirical, standard_error
        integer :: k, s, i, j
        logical :: matches

        call build_model(model, status)
        do k = 1, n_points
            points(k, 1) = -0.6_dp + 0.4_dp*real(k - 1, dp)
        end do
        call model%covariance(points, covariance, status)
        call expect(status%code == FORTNUM_OK, "the covariance evaluates", failures)

        call rng_seed(generator, int(202020, int64), status)
        totals = 0.0_dp
        cross = 0.0_dp
        do s = 1, n_samples
            call model%joint_sample(points, generator, samples, status)
            if (status%code /= FORTNUM_OK) exit
            totals = totals + samples(:, 1)
            do i = 1, n_points
                do j = 1, n_points
                    cross(i, j) = cross(i, j) + samples(i, 1)*samples(j, 1)
                end do
            end do
        end do
        totals = totals/real(n_samples, dp)

        matches = .true.
        do i = 1, n_points
            do j = 1, n_points
                empirical = cross(i, j)/real(n_samples, dp) - totals(i)*totals(j)
                ! The sampling error of a covariance entry scales with the
                ! geometric mean of the two variances.
                standard_error = sqrt(abs(covariance(i, i)*covariance(j, j))) &
                    /sqrt(real(n_samples, dp))
                if (abs(empirical - covariance(i, j)) > 6.0_dp*standard_error) &
                    matches = .false.
            end do
        end do
        call expect(matches, &
            "joint samples reproduce the covariance they are drawn from", failures)
    end subroutine check_joint_samples_match_the_covariance

    !! The property that makes this a useful second provider: with more query
    !! points than features the joint covariance is singular. Any acquisition
    !! that factorized it without care would pass against a GP and fail here.
    subroutine check_rank_deficiency_is_real(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(6, 1), covariance(6, 6)
        type(fortbo_batch_samples_t) :: batch
        real(dp) :: value
        integer :: k

        call build_model(model, status)
        do k = 1, 6
            points(k, 1) = -0.5_dp + 0.2_dp*real(k - 1, dp)
        end do
        call model%covariance(points, covariance, status)
        call expect(status%code == FORTNUM_OK, &
            "a rank-deficient covariance is still produced", failures)

        ! Six points, three features: the sixth row is a combination of the
        ! others, so the matrix cannot be full rank.
        call expect(abs(determinant_is_tiny(covariance)) < 1.0e-20_dp, &
            "the covariance is singular with more points than features", failures)

        ! Batch sampling must nonetheless work, because it draws in weight
        ! space rather than factorizing the function-space covariance.
        call rng_seed(generator, int(4242, int64), status)
        call batch%generate(model, points, 500, generator, status)
        call expect(status%code == FORTNUM_OK, &
            "batch sampling works despite the rank deficiency", failures)
        call fortbo_qei(batch, 1.0_dp, 0.0_dp, value, status)
        call expect(status%code == FORTNUM_OK .and. value >= 0.0_dp, &
            "qEI runs against a non-GP provider", failures)
    end subroutine check_rank_deficiency_is_real

    !! A crude but sufficient singularity witness: the product of the pivots
    !! from Gaussian elimination without pivoting, which underflows to zero for
    !! a matrix this deficient.
    real(dp) function determinant_is_tiny(matrix) result(value)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), allocatable :: work(:, :)
        integer :: n, i, j
        real(dp) :: factor

        n = size(matrix, 1)
        allocate (work, source=matrix)
        value = 1.0_dp
        do i = 1, n
            if (abs(work(i, i)) < 1.0e-300_dp) then
                value = 0.0_dp
                return
            end if
            do j = i + 1, n
                factor = work(j, i)/work(i, i)
                work(j, :) = work(j, :) - factor*work(i, :)
            end do
            value = value*work(i, i)
        end do
    end function determinant_is_tiny

    !! The contract's actual claim: unchanged acquisition code against a
    !! different provider.
    subroutine check_acquisitions_run_unchanged(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model
        type(fortnum_status_t) :: status
        type(fortbo_ei_t) :: ei
        type(fortbo_ucb_t) :: ucb
        real(dp) :: points(5, 1), values(5), kg_value
        real(dp) :: reference(3, 1)
        integer :: k

        call build_model(model, status)
        do k = 1, 5
            points(k, 1) = -0.8_dp + 0.4_dp*real(k - 1, dp)
        end do

        ei%best = 1.0_dp
        ei%xi = 0.0_dp
        call ei%value(model, points, values, status)
        call expect(status%code == FORTNUM_OK .and. all(values >= 0.0_dp), &
            "expected improvement runs against a non-GP provider", failures)

        ucb%beta = 2.0_dp
        call ucb%value(model, points, values, status)
        call expect(status%code == FORTNUM_OK, &
            "UCB runs against a non-GP provider", failures)

        ! Knowledge gradient needs the joint covariance, which this provider has.
        reference(:, 1) = [-0.5_dp, 0.0_dp, 0.5_dp]
        call fortbo_knowledge_gradient_value(model, [0.25_dp], reference, 0.01_dp, &
            kg_value, status)
        call expect(status%code == FORTNUM_OK .and. kg_value >= 0.0_dp, &
            "knowledge gradient runs against a non-GP provider", failures)
    end subroutine check_acquisitions_run_unchanged

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_linear_posterior_t) :: model, unfitted
        type(fortnum_status_t) :: status
        real(dp) :: inputs(4, 1), targets(4), points(2, 1), mean(2), variance(2)
        integer :: k

        do k = 1, 4
            inputs(k, 1) = real(k, dp)
            targets(k) = truth(inputs(k, 1))
        end do

        call model%fit(inputs, targets, quadratic_features, 3, 0.0_dp, 1.0_dp, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an improper prior is refused rather than left to fail later", failures)

        call model%fit(inputs, targets(1:3), quadratic_features, 3, 1.0_dp, &
            1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched inputs and targets are refused", failures)

        call unfitted%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "querying an unfitted provider is refused", failures)
        call expect(unfitted%capabilities() == 0, &
            "an unfitted provider declares nothing", failures)
    end subroutine check_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_linear_posterior
