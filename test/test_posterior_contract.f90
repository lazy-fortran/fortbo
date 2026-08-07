program test_posterior_contract
    !! BO0: the versioned posterior contract.
    !!
    !! Oracles are independent of the implementation under test:
    !!   * the joint sampler is checked against its own empirical mean and
    !!     covariance over many draws, which is a Monte Carlo oracle and not a
    !!     restatement of the Cholesky code that produced the draws;
    !!   * the predictive log density is checked against the closed-form scalar
    !!     normal density and against the explicit two-by-two inverse and
    !!     determinant, neither of which touches the factorization path;
    !!   * refusals are checked for both the status code and the operation name
    !!     in the message, so a refusal cannot degrade into a generic failure.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_posterior, only: fortbo_posterior_t, fortbo_capability_name, &
        FORTBO_POSTERIOR_CONTRACT_VERSION, FORTBO_CAP_MOMENTS, FORTBO_CAP_COVARIANCE, &
        FORTBO_CAP_JOINT_SAMPLE, FORTBO_CAP_REPARAM_SAMPLE, FORTBO_CAP_LOG_DENSITY, &
        FORTBO_CAP_MOMENT_GRADIENT
    use fortbo_test_posteriors, only: demo_posterior_t, moments_only_posterior_t, &
        demo_mean, demo_kernel, DEMO_SIGNAL
    implicit none

    integer :: failures

    failures = 0
    call check_version(failures)
    call check_capability_bits(failures)
    call check_refusals(failures)
    call check_moments_against_closed_form(failures)
    call check_covariance_structure(failures)
    call check_reparam_is_deterministic(failures)
    call check_sample_moments(failures)
    call check_log_density_scalar(failures)
    call check_log_density_pair(failures)

    if (failures == 0) then
        print *, "test_posterior_contract: PASS"
    else
        print *, "test_posterior_contract: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_version(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior

        posterior%dimension = 2
        call expect(FORTBO_POSTERIOR_CONTRACT_VERSION == 1, &
                    "contract version is 1", failures)
        call expect(posterior%contract_version == 1, &
                    "implementations default to the current contract version", failures)
        call expect(posterior%n_inputs() == 2, "n_inputs reports the input width", &
                    failures)
    end subroutine check_version

    subroutine check_capability_bits(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: full
        type(moments_only_posterior_t) :: partial

        full%dimension = 1
        partial%dimension = 1
        call expect(full%supports(FORTBO_CAP_MOMENTS), "declared bit is supported", &
                    failures)
        call expect(full%supports(FORTBO_CAP_MOMENTS + FORTBO_CAP_JOINT_SAMPLE), &
                    "a conjunction of declared bits is supported", failures)
        call expect(.not. full%supports(FORTBO_CAP_MOMENT_GRADIENT), &
                    "an undeclared bit is not supported", failures)
        call expect(.not. partial%supports(FORTBO_CAP_MOMENTS + FORTBO_CAP_COVARIANCE), &
                    "a conjunction with one missing bit is not supported", failures)
        call expect(fortbo_capability_name(FORTBO_CAP_JOINT_SAMPLE) == "joint_sample", &
                    "capability names are reported", failures)
    end subroutine check_capability_bits

    !! Every operation a model does not declare must refuse by name.
    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(moments_only_posterior_t) :: partial
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), covariance(2, 2), samples(2, 3), base(2, 3)
        real(dp) :: gradient(2, 1), values(2), density
        real(dp) :: hessian(1, 1), point(1)

        partial%dimension = 1
        points = reshape([0.1_dp, 0.6_dp], [2, 1])
        base = 0.0_dp
        values = 0.0_dp
        point = 0.0_dp
        call rng_seed(generator, int(7, int64), status)

        call partial%covariance(points, covariance, status)
        call expect_refusal(status, "covariance", failures)
        call partial%joint_sample(points, generator, samples, status)
        call expect_refusal(status, "joint_sample", failures)
        call partial%reparam_sample(points, base, samples, status)
        call expect_refusal(status, "reparam_sample", failures)
        call partial%log_density(points, values, density, status)
        call expect_refusal(status, "log_density", failures)
        call partial%moment_gradient(points, gradient, gradient, status)
        call expect_refusal(status, "moment_gradient", failures)
        call partial%moment_hessian(point, hessian, hessian, status)
        call expect_refusal(status, "moment_hessian", failures)
    end subroutine check_refusals

    subroutine check_moments_against_closed_form(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: points(3, 2), mean(3), variance(3)

        posterior%dimension = 2
        points = reshape([0.0_dp, 0.5_dp, -1.25_dp, 1.0_dp, -0.5_dp, 2.0_dp], [3, 2])
        call posterior%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "moments succeed", failures)
        call expect_close(mean(1), 1.0_dp, 1.0e-14_dp, "mean at first query", failures)
        call expect_close(mean(2), 0.0_dp, 1.0e-14_dp, "mean at second query", failures)
        call expect_close(mean(3), 0.75_dp, 1.0e-14_dp, "mean at third query", failures)
        call expect_close(variance(1), DEMO_SIGNAL, 1.0e-14_dp, &
                          "prior variance is the signal variance", failures)
    end subroutine check_moments_against_closed_form

    !! The joint covariance must be symmetric and its diagonal must agree with
    !! the marginal variances the same model reports through `moments`.
    subroutine check_covariance_structure(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: points(4, 1), covariance(4, 4), mean(4), variance(4)
        real(dp) :: asymmetry
        integer :: i, j

        posterior%dimension = 1
        points = reshape([-1.0_dp, -0.25_dp, 0.4_dp, 1.3_dp], [4, 1])
        call posterior%covariance(points, covariance, status)
        call expect(status%code == FORTNUM_OK, "covariance succeeds", failures)
        call posterior%moments(points, mean, variance, status)

        asymmetry = 0.0_dp
        do j = 1, 4
            do i = 1, 4
                asymmetry = max(asymmetry, abs(covariance(i, j) - covariance(j, i)))
            end do
        end do
        call expect(asymmetry == 0.0_dp, "covariance is exactly symmetric", failures)
        do i = 1, 4
            call expect_close(covariance(i, i), variance(i), 1.0e-8_dp, &
                              "covariance diagonal matches marginal variance", failures)
        end do
        call expect_close(covariance(1, 2), demo_kernel([-1.0_dp], [-0.25_dp]), &
                          1.0e-14_dp, "off-diagonal matches the kernel", failures)
    end subroutine check_covariance_structure

    !! With the base samples held fixed the reparameterized map is a
    !! deterministic function of the query, which is what makes it
    !! differentiable and what common random numbers rely on. A zero base must
    !! land exactly on the mean.
    subroutine check_reparam_is_deterministic(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: points(3, 1), base(3, 2), first(3, 2), second(3, 2)
        real(dp) :: zero_base(3, 1), at_mean(3, 1), mean(3), variance(3)

        posterior%dimension = 1
        points = reshape([0.0_dp, 0.7_dp, 1.4_dp], [3, 1])
        base = reshape([0.3_dp, -1.1_dp, 0.8_dp, 2.0_dp, 0.25_dp, -0.6_dp], [3, 2])
        call posterior%reparam_sample(points, base, first, status)
        call expect(status%code == FORTNUM_OK, "reparam sampling succeeds", failures)
        call posterior%reparam_sample(points, base, second, status)
        call expect(maxval(abs(first - second)) == 0.0_dp, &
                    "reparam sampling is bitwise reproducible", failures)

        zero_base = 0.0_dp
        call posterior%reparam_sample(points, zero_base, at_mean, status)
        call posterior%moments(points, mean, variance, status)
        call expect(maxval(abs(at_mean(:, 1) - mean)) < 1.0e-14_dp, &
                    "a zero base reproduces the posterior mean", failures)
    end subroutine check_reparam_is_deterministic

    !! Monte Carlo oracle: the empirical first and second moments of the joint
    !! sampler must converge to the analytic mean and covariance. The tolerance
    !! is a five-standard-error band, so a correct sampler passes essentially
    !! always and a sampler with the wrong correlation structure cannot.
    subroutine check_sample_moments(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        type(rng_t) :: generator
        integer, parameter :: n_points = 3
        integer, parameter :: n_draws = 40000
        real(dp) :: points(n_points, 1), mean(n_points), variance(n_points)
        real(dp) :: covariance(n_points, n_points)
        real(dp), allocatable :: samples(:, :)
        real(dp) :: empirical_mean(n_points), empirical_cov(n_points, n_points)
        real(dp) :: standard_error, deviation
        integer :: i, j

        posterior%dimension = 1
        points = reshape([-0.6_dp, 0.0_dp, 0.45_dp], [n_points, 1])
        call posterior%moments(points, mean, variance, status)
        call posterior%covariance(points, covariance, status)

        allocate (samples(n_points, n_draws))
        call rng_seed(generator, int(20260808, int64), status)
        call posterior%joint_sample(points, generator, samples, status)
        call expect(status%code == FORTNUM_OK, "joint sampling succeeds", failures)

        do i = 1, n_points
            empirical_mean(i) = sum(samples(i, :))/real(n_draws, dp)
        end do
        do j = 1, n_points
            do i = 1, n_points
                empirical_cov(i, j) = sum((samples(i, :) - empirical_mean(i))* &
                                          (samples(j, :) - empirical_mean(j))) &
                                      /real(n_draws - 1, dp)
            end do
        end do

        do i = 1, n_points
            standard_error = sqrt(variance(i)/real(n_draws, dp))
            deviation = abs(empirical_mean(i) - mean(i))
            call expect(deviation < 5.0_dp*standard_error, &
                        "empirical mean matches the analytic mean", failures)
        end do
        do j = 1, n_points
            do i = 1, n_points
                standard_error = sqrt((variance(i)*variance(j) + covariance(i, j)**2) &
                                      /real(n_draws, dp))
                deviation = abs(empirical_cov(i, j) - covariance(i, j))
                call expect(deviation < 5.0_dp*standard_error, &
                            "empirical covariance matches the analytic covariance", &
                            failures)
            end do
        end do
    end subroutine check_sample_moments

    !! One query point reduces the predictive density to the scalar normal, for
    !! which the closed form involves no factorization at all.
    subroutine check_log_density_scalar(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: points(1, 1), values(1), density, expected
        real(dp) :: mu, sigma_squared

        posterior%dimension = 1
        points = reshape([0.35_dp], [1, 1])
        values = [1.2_dp]
        call posterior%log_density(points, values, density, status)
        call expect(status%code == FORTNUM_OK, "log density succeeds", failures)

        mu = demo_mean([0.35_dp])
        sigma_squared = demo_kernel([0.35_dp], [0.35_dp])
        expected = -0.5_dp*log(2.0_dp*acos(-1.0_dp)*sigma_squared) &
                   - 0.5_dp*(values(1) - mu)**2/sigma_squared
        call expect_close(density, expected, 1.0e-8_dp, &
                          "log density matches the scalar normal", failures)
    end subroutine check_log_density_scalar

    !! Two query points, checked against the explicit two-by-two inverse and
    !! determinant rather than against a Cholesky solve.
    subroutine check_log_density_pair(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), values(2), density, expected
        real(dp) :: a, b, c, determinant, r1, r2, quadratic

        posterior%dimension = 1
        points = reshape([-0.2_dp, 0.9_dp], [2, 1])
        values = [0.4_dp, -0.75_dp]
        call posterior%log_density(points, values, density, status)
        call expect(status%code == FORTNUM_OK, "paired log density succeeds", failures)

        a = demo_kernel([-0.2_dp], [-0.2_dp])
        b = demo_kernel([-0.2_dp], [0.9_dp])
        c = demo_kernel([0.9_dp], [0.9_dp])
        determinant = a*c - b*b
        r1 = values(1) - demo_mean([-0.2_dp])
        r2 = values(2) - demo_mean([0.9_dp])
        quadratic = (c*r1*r1 - 2.0_dp*b*r1*r2 + a*r2*r2)/determinant
        expected = -0.5_dp*quadratic - 0.5_dp*log(determinant) &
                   - log(2.0_dp*acos(-1.0_dp))
        call expect_close(density, expected, 1.0e-7_dp, &
                          "log density matches the explicit two-by-two form", failures)
    end subroutine check_log_density_pair

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

    subroutine expect_close(actual, expected, tolerance, description, failures)
        real(dp), intent(in) :: actual
        real(dp), intent(in) :: expected
        real(dp), intent(in) :: tolerance
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (abs(actual - expected) > tolerance) then
            failures = failures + 1
            print *, "  FAIL: ", description, actual, expected
        end if
    end subroutine expect_close

    subroutine expect_refusal(status, operation, failures)
        type(fortnum_status_t), intent(in) :: status
        character(len=*), intent(in) :: operation
        integer, intent(inout) :: failures

        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            failures = failures + 1
            print *, "  FAIL: refusal code for ", operation
            return
        end if
        if (index(status%msg, operation) == 0) then
            failures = failures + 1
            print *, "  FAIL: refusal for ", operation, " does not name the operation"
        end if
    end subroutine expect_refusal

end program test_posterior_contract
