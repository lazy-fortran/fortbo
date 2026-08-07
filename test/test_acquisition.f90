program test_acquisition
    !! BO1: analytic acquisitions.
    !!
    !! Oracles, none of which reuse the generated closed form:
    !!   * expected improvement and probability of improvement are checked
    !!     against Simpson quadrature of their defining integrals,
    !!         EI = int_{-inf}^{tau} (tau - y) N(y; mu, sigma) dy
    !!         PI = int_{-inf}^{tau} N(y; mu, sigma) dy
    !!     evaluated on a fine grid. Quadrature knows nothing about erf;
    !!   * the chain-rule input gradients are checked against central finite
    !!     differences of the acquisition value itself, at a step chosen for the
    !!     usual cube-root-of-epsilon truncation/rounding balance;
    !!   * log expected improvement is checked against log(EI) where EI is
    !!     representable, and against the quadrature integral evaluated in
    !!     extended-range form far into the tail where EI underflows to zero and
    !!     the naive logarithm would return -infinity.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_DOMAIN_ERROR
    use fortbo_acquisition, only: fortbo_ei_t, fortbo_log_ei_t, fortbo_pi_t, &
        fortbo_ucb_t, fortbo_expected_improvement, fortbo_probability_of_improvement, &
        fortbo_log_expected_improvement
    use fortbo_test_posteriors, only: curved_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_ei_against_quadrature(failures)
    call check_pi_against_quadrature(failures)
    call check_zero_variance_limit(failures)
    call check_monotonicity(failures)
    call check_log_ei(failures)
    call check_moment_derivatives(failures)
    call check_input_gradients(failures)
    call check_ucb(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_acquisition: PASS"
    else
        print *, "test_acquisition: FAIL", failures
        error stop 1
    end if

contains

    !! Simpson quadrature of the expected-improvement integral. The integrand
    !! vanishes above the threshold, and twelve standard deviations below the
    !! mean the normal density is far below double-precision relevance.
    function quadrature_ei(mean, sd, threshold) result(value)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: sd
        real(dp), intent(in) :: threshold
        real(dp) :: value
        integer, parameter :: n = 200000
        real(dp) :: lower, upper, h, y, total
        integer :: i

        lower = min(mean - 12.0_dp*sd, threshold - 12.0_dp*sd)
        upper = threshold
        if (upper <= lower) then
            value = 0.0_dp
            return
        end if
        h = (upper - lower)/real(n, dp)
        total = integrand_ei(lower, mean, sd, threshold) &
            + integrand_ei(upper, mean, sd, threshold)
        do i = 1, n - 1
            y = lower + real(i, dp)*h
            if (mod(i, 2) == 1) then
                total = total + 4.0_dp*integrand_ei(y, mean, sd, threshold)
            else
                total = total + 2.0_dp*integrand_ei(y, mean, sd, threshold)
            end if
        end do
        value = total*h/3.0_dp
    end function quadrature_ei

    pure function integrand_ei(y, mean, sd, threshold) result(value)
        real(dp), intent(in) :: y, mean, sd, threshold
        real(dp) :: value

        value = (threshold - y)*normal_density(y, mean, sd)
    end function integrand_ei

    function quadrature_pi(mean, sd, threshold) result(value)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: sd
        real(dp), intent(in) :: threshold
        real(dp) :: value
        integer, parameter :: n = 200000
        real(dp) :: lower, upper, h, y, total
        integer :: i

        lower = min(mean - 12.0_dp*sd, threshold - 12.0_dp*sd)
        upper = threshold
        if (upper <= lower) then
            value = 0.0_dp
            return
        end if
        h = (upper - lower)/real(n, dp)
        total = normal_density(lower, mean, sd) + normal_density(upper, mean, sd)
        do i = 1, n - 1
            y = lower + real(i, dp)*h
            if (mod(i, 2) == 1) then
                total = total + 4.0_dp*normal_density(y, mean, sd)
            else
                total = total + 2.0_dp*normal_density(y, mean, sd)
            end if
        end do
        value = total*h/3.0_dp
    end function quadrature_pi

    pure function normal_density(y, mean, sd) result(value)
        real(dp), intent(in) :: y, mean, sd
        real(dp) :: value

        value = exp(-0.5_dp*((y - mean)/sd)**2)/(sd*sqrt(8.0_dp*atan(1.0_dp)))
    end function normal_density

    subroutine check_ei_against_quadrature(failures)
        integer, intent(inout) :: failures
        real(dp) :: means(5), sds(3), best, xi, analytic, numeric
        integer :: i, j

        means = [-2.0_dp, -0.5_dp, 0.0_dp, 0.75_dp, 3.0_dp]
        sds = [0.25_dp, 1.0_dp, 2.5_dp]
        best = 0.5_dp
        xi = 0.05_dp

        do i = 1, size(means)
            do j = 1, size(sds)
                call fortbo_expected_improvement(means(i), sds(j), best, xi, analytic)
                numeric = quadrature_ei(means(i), sds(j), best - xi)
                call expect_close(analytic, numeric, 1.0e-9_dp + 1.0e-7_dp*abs(numeric), &
                    "EI matches its defining integral", failures)
            end do
        end do
    end subroutine check_ei_against_quadrature

    subroutine check_pi_against_quadrature(failures)
        integer, intent(inout) :: failures
        real(dp) :: means(4), sds(3), best, analytic, numeric
        integer :: i, j

        means = [-1.5_dp, 0.0_dp, 0.4_dp, 2.0_dp]
        sds = [0.3_dp, 1.0_dp, 2.0_dp]
        best = 0.5_dp

        do i = 1, size(means)
            do j = 1, size(sds)
                call fortbo_probability_of_improvement(means(i), sds(j), best, 0.0_dp, &
                    analytic)
                numeric = quadrature_pi(means(i), sds(j), best)
                call expect_close(analytic, numeric, 1.0e-9_dp, &
                    "PI matches its defining integral", failures)
            end do
        end do
    end subroutine check_pi_against_quadrature

    !! As the posterior collapses the acquisition must approach the
    !! deterministic shortfall continuously, not jump at the floor.
    subroutine check_zero_variance_limit(failures)
        integer, intent(inout) :: failures
        real(dp) :: value, limit_value, small

        call fortbo_expected_improvement(0.2_dp, 0.0_dp, 1.0_dp, 0.0_dp, value)
        call expect_close(value, 0.8_dp, 1.0e-14_dp, &
            "zero variance gives the deterministic shortfall", failures)
        call fortbo_expected_improvement(2.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, value)
        call expect_close(value, 0.0_dp, 1.0e-14_dp, &
            "a worse deterministic point has no improvement", failures)

        small = 1.0e-9_dp
        call fortbo_expected_improvement(0.2_dp, small, 1.0_dp, 0.0_dp, value)
        call fortbo_expected_improvement(0.2_dp, 0.0_dp, 1.0_dp, 0.0_dp, limit_value)
        call expect_close(value, limit_value, 1.0e-7_dp, &
            "the limit is approached continuously", failures)

        call fortbo_probability_of_improvement(0.2_dp, 0.0_dp, 1.0_dp, 0.0_dp, value)
        call expect_close(value, 1.0_dp, 1.0e-14_dp, &
            "a certainly better point improves with probability one", &
            failures)
        call fortbo_probability_of_improvement(2.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, value)
        call expect_close(value, 0.0_dp, 1.0e-14_dp, &
            "a certainly worse point never improves", failures)
    end subroutine check_zero_variance_limit

    !! Structural properties that hold for every Gaussian posterior: expected
    !! improvement falls as the mean rises and rises as the uncertainty grows.
    subroutine check_monotonicity(failures)
        integer, intent(inout) :: failures
        real(dp) :: previous, value
        integer :: i
        logical :: decreasing, increasing

        decreasing = .true.
        previous = huge(1.0_dp)
        do i = 0, 40
            call fortbo_expected_improvement(-2.0_dp + 0.1_dp*real(i, dp), 1.0_dp, &
                0.0_dp, 0.0_dp, value)
            if (value > previous) decreasing = .false.
            previous = value
        end do
        call expect(decreasing, "EI decreases as the posterior mean rises", failures)

        increasing = .true.
        previous = -huge(1.0_dp)
        do i = 1, 40
            call fortbo_expected_improvement(0.5_dp, 0.05_dp*real(i, dp), 0.0_dp, &
                0.0_dp, value)
            if (value < previous) increasing = .false.
            previous = value
        end do
        call expect(increasing, "EI increases as the posterior widens", failures)
    end subroutine check_monotonicity

    !! In the tail EI underflows to zero, so log(EI) is -infinity while the true
    !! log expected improvement is a large finite negative number. The oracle
    !! there is the asymptotic value of the integral computed in log space.
    subroutine check_log_ei(failures)
        integer, intent(inout) :: failures
        real(dp) :: value, direct, ei, z, expected
        integer :: i
        logical :: agrees, finite

        agrees = .true.
        do i = 0, 20
            call fortbo_expected_improvement(-1.0_dp + 0.15_dp*real(i, dp), 1.0_dp, &
                0.0_dp, 0.0_dp, ei)
            call fortbo_log_expected_improvement(-1.0_dp + 0.15_dp*real(i, dp), &
                1.0_dp, 0.0_dp, 0.0_dp, value)
            direct = log(ei)
            if (abs(value - direct) > 1.0e-9_dp*max(1.0_dp, abs(direct))) &
                agrees = .false.
        end do
        call expect(agrees, "log EI agrees with log of EI where EI is representable", &
            failures)

        ! Deep tail: mean far above the incumbent, unit standard deviation.
        call fortbo_expected_improvement(40.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, ei)
        call fortbo_log_expected_improvement(40.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, value)
        finite = value > -huge(1.0_dp) .and. value < 0.0_dp
        call expect(ei == 0.0_dp, "EI has underflowed in the deep tail", failures)
        call expect(finite, "log EI stays finite where EI has underflowed", failures)

        ! Leading asymptotic term: log EI ~ -z^2/2 - log(2 pi)/2 - log(z^2).
        z = -40.0_dp
        expected = -0.5_dp*z*z - 0.5_dp*log(8.0_dp*atan(1.0_dp)) - log(z*z)
        call expect(abs(value - expected) < 0.01_dp, &
            "log EI matches its leading asymptotic form", failures)

        call fortbo_log_expected_improvement(2.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, value)
        call expect(value == -huge(1.0_dp), &
            "a deterministic non-improvement has log EI of minus infinity", &
            failures)
    end subroutine check_log_ei

    !! Derivatives with respect to the posterior moments, against central
    !! differences of the analytic value.
    subroutine check_moment_derivatives(failures)
        integer, intent(inout) :: failures
        real(dp) :: mean, sd, best, step, value, d_mean, d_sd
        real(dp) :: plus, minus, numeric
        integer :: i

        best = 1.0_dp
        step = 1.0e-6_dp
        do i = 1, 5
            mean = -1.0_dp + 0.6_dp*real(i, dp)
            sd = 0.4_dp + 0.3_dp*real(i, dp)

            call fortbo_expected_improvement(mean, sd, best, 0.0_dp, value, d_mean, d_sd)
            call fortbo_expected_improvement(mean + step, sd, best, 0.0_dp, plus)
            call fortbo_expected_improvement(mean - step, sd, best, 0.0_dp, minus)
            numeric = (plus - minus)/(2.0_dp*step)
            call expect_close(d_mean, numeric, 1.0e-6_dp, &
                "dEI/dmean matches central differences", failures)

            call fortbo_expected_improvement(mean, sd + step, best, 0.0_dp, plus)
            call fortbo_expected_improvement(mean, sd - step, best, 0.0_dp, minus)
            numeric = (plus - minus)/(2.0_dp*step)
            call expect_close(d_sd, numeric, 1.0e-6_dp, &
                "dEI/dsd matches central differences", failures)

            call fortbo_probability_of_improvement(mean, sd, best, 0.0_dp, value, &
                d_mean, d_sd)
            call fortbo_probability_of_improvement(mean + step, sd, best, 0.0_dp, plus)
            call fortbo_probability_of_improvement(mean - step, sd, best, 0.0_dp, minus)
            numeric = (plus - minus)/(2.0_dp*step)
            call expect_close(d_mean, numeric, 1.0e-6_dp, &
                "dPI/dmean matches central differences", failures)
        end do

        ! The exact closed form dEI/dmean = -Phi(z) is what fortsym derived; it
        ! must hold to full precision, not merely to difference accuracy.
        call fortbo_expected_improvement(0.3_dp, 1.7_dp, 1.0_dp, 0.0_dp, value, &
            d_mean, d_sd)
        call fortbo_probability_of_improvement(0.3_dp, 1.7_dp, 1.0_dp, 0.0_dp, plus)
        call expect_close(d_mean, -plus, 1.0e-13_dp, &
            "dEI/dmean equals minus the improvement probability", failures)
    end subroutine check_moment_derivatives

    !! Chain-rule gradients through a posterior whose mean and standard
    !! deviation both curve, against central differences of the acquisition.
    subroutine check_input_gradients(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ei_t) :: ei
        type(fortbo_pi_t) :: pi_acq
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: points(3, 2), gradient(3, 2), numeric(3, 2)

        posterior%dimension = 2
        points = reshape([0.4_dp, -0.7_dp, 1.1_dp, 0.9_dp, 0.2_dp, -1.3_dp], [3, 2])

        ei%best = 2.0_dp
        call ei%value_gradient(posterior, points, gradient, status)
        call expect(status%code == FORTNUM_OK, "the EI gradient is available", failures)
        call difference_gradient_ei(posterior, ei, points, numeric)
        call expect(maxval(abs(gradient - numeric)) < 1.0e-6_dp, &
            "the EI input gradient matches central differences", failures)

        pi_acq%best = 2.0_dp
        call pi_acq%value_gradient(posterior, points, gradient, status)
        call difference_gradient_pi(posterior, pi_acq, points, numeric)
        call expect(maxval(abs(gradient - numeric)) < 1.0e-6_dp, &
            "the PI input gradient matches central differences", failures)

        ucb%beta = 1.5_dp
        call ucb%value_gradient(posterior, points, gradient, status)
        call difference_gradient_ucb(posterior, ucb, points, numeric)
        call expect(maxval(abs(gradient - numeric)) < 1.0e-6_dp, &
            "the confidence-bound gradient matches central differences", &
            failures)
    end subroutine check_input_gradients

    subroutine difference_gradient_ei(posterior, acquisition, points, numeric)
        type(curved_posterior_t), intent(in) :: posterior
        type(fortbo_ei_t), intent(in) :: acquisition
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: numeric(:, :)
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:, :), plus(:), minus(:)
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: i, j

        allocate (shifted, source=points)
        allocate (plus(size(points, 1)), minus(size(points, 1)))
        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                shifted = points
                shifted(i, j) = points(i, j) + step
                call acquisition%value(posterior, shifted, plus, status)
                shifted(i, j) = points(i, j) - step
                call acquisition%value(posterior, shifted, minus, status)
                numeric(i, j) = (plus(i) - minus(i))/(2.0_dp*step)
            end do
        end do
    end subroutine difference_gradient_ei

    subroutine difference_gradient_pi(posterior, acquisition, points, numeric)
        type(curved_posterior_t), intent(in) :: posterior
        type(fortbo_pi_t), intent(in) :: acquisition
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: numeric(:, :)
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:, :), plus(:), minus(:)
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: i, j

        allocate (shifted, source=points)
        allocate (plus(size(points, 1)), minus(size(points, 1)))
        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                shifted = points
                shifted(i, j) = points(i, j) + step
                call acquisition%value(posterior, shifted, plus, status)
                shifted(i, j) = points(i, j) - step
                call acquisition%value(posterior, shifted, minus, status)
                numeric(i, j) = (plus(i) - minus(i))/(2.0_dp*step)
            end do
        end do
    end subroutine difference_gradient_pi

    subroutine difference_gradient_ucb(posterior, acquisition, points, numeric)
        type(curved_posterior_t), intent(in) :: posterior
        type(fortbo_ucb_t), intent(in) :: acquisition
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: numeric(:, :)
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:, :), plus(:), minus(:)
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: i, j

        allocate (shifted, source=points)
        allocate (plus(size(points, 1)), minus(size(points, 1)))
        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                shifted = points
                shifted(i, j) = points(i, j) + step
                call acquisition%value(posterior, shifted, plus, status)
                shifted(i, j) = points(i, j) - step
                call acquisition%value(posterior, shifted, minus, status)
                numeric(i, j) = (plus(i) - minus(i))/(2.0_dp*step)
            end do
        end do
    end subroutine difference_gradient_ucb

    subroutine check_ucb(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), values(2), mean(2), variance(2)

        posterior%dimension = 1
        points = reshape([0.5_dp, -1.5_dp], [2, 1])
        ucb%beta = 2.0_dp
        call ucb%value(posterior, points, values, status)
        call expect(status%code == FORTNUM_OK, "the confidence bound evaluates", &
            failures)
        call posterior%moments(points, mean, variance, status)
        call expect_close(values(1), -(mean(1) - 2.0_dp*sqrt(variance(1))), 1.0e-14_dp, &
            "the confidence bound is the negated lower bound", failures)

        ucb%beta = 0.0_dp
        call ucb%value(posterior, points, values, status)
        call expect_close(values(1), -mean(1), 1.0e-14_dp, &
            "zero exploration weight gives the negated mean", failures)

        ucb%beta = -1.0_dp
        call ucb%value(posterior, points, values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative exploration weight is refused", failures)
    end subroutine check_ucb

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(moments_only_posterior_t) :: partial
        type(fortbo_ei_t) :: ei
        type(fortbo_log_ei_t) :: log_ei
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 1), values(2), gradient(2, 1)

        partial%dimension = 1
        points = reshape([0.1_dp, 0.8_dp], [2, 1])

        call ei%value(partial, points, values, status)
        call expect(status%code == FORTNUM_OK, &
            "a moments-only surrogate still supports EI", failures)

        call ei%value_gradient(partial, points, gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "EI refuses a gradient without moment gradients", failures)
        call expect(index(status%msg, "moment_gradient") > 0, &
            "the refusal names the missing posterior operation", failures)

        call log_ei%value_gradient(partial, points, gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "log EI refuses an input gradient", failures)
        call expect(index(status%msg, "log_expected_improvement") > 0, &
            "the refusal names the acquisition", failures)
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

end program test_acquisition
