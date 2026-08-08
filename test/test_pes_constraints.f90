program test_pes_constraints
    !! BO3: PES constraints C1.2 and C2 by expectation propagation.
    !!
    !! **The oracle is exact, not another approximation.** The paper's prior on
    !! the latent vector is diagonal and each constraint touches exactly one
    !! coordinate, so the true posterior factorizes and each marginal has a
    !! closed form -- a truncated Gaussian for the Hessian constraints, and the
    !! standard probit-likelihood moments for the soft maximum. EP applied to a
    !! single factor per coordinate must reproduce those exactly, because the
    !! tilted distribution it moment-matches *is* the true marginal. So this is
    !! not a tolerance chosen to let an approximation through: any disagreement
    !! beyond rounding is a defect.
    !!
    !! That is a stronger check than it looks. It catches a cavity computed
    !! with the wrong sign, a `kappa` applied in the wrong direction (the
    !! paper's two factor types differ by exactly that: `m + 1/kappa` for the
    !! soft maximum against `m - 1/kappa` for the truncation), and a site
    !! precision that forgot to subtract the cavity -- none of which a check
    !! for "the constraint moved the mean the right way" would notice.
    !!
    !! The closed forms below are written from the definitions of the truncated
    !! normal and the probit posterior, not lifted from the module.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_pes, only: fortbo_pes_latent_constraints, &
        FORTBO_PES_EP_MAX_PASSES
    implicit none

    integer :: failures

    failures = 0
    call check_truncation_matches_the_closed_form(failures)
    call check_soft_maximum_matches_the_closed_form(failures)
    call check_the_constraints_bind(failures)
    call check_deep_tails_stay_finite(failures)
    call check_against_quadrature(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_pes_constraints: PASS"
    else
        print *, "test_pes_constraints: FAIL", failures
        error stop 1
    end if

contains

    pure real(dp) function standard_pdf(a) result(value)
        real(dp), intent(in) :: a

        value = exp(-0.5_dp*a*a)/sqrt(2.0_dp*3.141592653589793_dp)
    end function standard_pdf

    pure real(dp) function standard_cdf(a) result(value)
        real(dp), intent(in) :: a

        value = 0.5_dp*erfc(-a/sqrt(2.0_dp))
    end function standard_cdf

    !! Exact mean and variance of `N(mu, v)` truncated to the negative half
    !! line, from the definition of the truncated normal.
    subroutine exact_truncated(mu, v, mean, variance)
        real(dp), intent(in) :: mu, v
        real(dp), intent(out) :: mean, variance
        real(dp) :: sd, cut, lambda

        sd = sqrt(v)
        cut = -mu/sd
        lambda = standard_pdf(cut)/standard_cdf(cut)
        mean = mu - sd*lambda
        variance = v*(1.0_dp - cut*lambda - lambda**2)
    end subroutine exact_truncated

    !! Exact moments of `N(mu, v)` reweighted by `Phi((z - threshold)/sigma)`,
    !! the standard probit-likelihood posterior.
    subroutine exact_soft_maximum(mu, v, threshold, sigma, mean, variance)
        real(dp), intent(in) :: mu, v, threshold, sigma
        real(dp), intent(out) :: mean, variance
        real(dp) :: denominator, alpha, lambda

        denominator = sqrt(v + sigma**2)
        alpha = (mu - threshold)/denominator
        lambda = standard_pdf(alpha)/standard_cdf(alpha)
        mean = mu + v*lambda/denominator
        variance = v - (v**2/(v + sigma**2))*lambda*(alpha + lambda)
    end subroutine exact_soft_maximum

    !! Every Hessian coordinate against the exact truncated normal.
    subroutine check_truncation_matches_the_closed_form(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 6
        real(dp) :: prior_mean(n), prior_variance(n)
        real(dp) :: mean(n), variance(n)
        real(dp) :: expected_mean, expected_variance
        real(dp) :: worst_mean, worst_variance
        integer :: passes, k
        logical :: converged

        ! Entry one is `f(x*)` and carries the soft maximum; entries two
        ! onward are the Hessian diagonal. A spread of prior means, so both
        ! the constraint-is-nearly-satisfied and the constraint-is-violated
        ! regimes are covered.
        prior_mean(1) = 0.4_dp
        prior_variance(1) = 1.0_dp
        prior_mean(2:) = [-2.0_dp, -0.5_dp, 0.0_dp, 0.8_dp, 2.5_dp]
        prior_variance(2:) = [0.5_dp, 1.0_dp, 0.25_dp, 2.0_dp, 1.5_dp]

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, &
            0.0_dp, 0.3_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_OK, "the EP pass runs", failures)
        call expect(converged, "EP converges on a well-posed problem", failures)

        worst_mean = 0.0_dp
        worst_variance = 0.0_dp
        do k = 2, n
            call exact_truncated(prior_mean(k), prior_variance(k), &
                expected_mean, expected_variance)
            worst_mean = max(worst_mean, abs(mean(k) - expected_mean))
            worst_variance = max(worst_variance, &
                abs(variance(k) - expected_variance))
        end do

        ! Exact, not approximate: one factor per coordinate on a diagonal
        ! prior means the tilted distribution EP matches is the true marginal.
        call expect(worst_mean < 1.0e-9_dp, &
            "the Hessian marginals match the exact truncated normal mean", &
            failures)
        call expect(worst_variance < 1.0e-9_dp, &
            "and its exact variance", failures)
        if (worst_mean >= 1.0e-9_dp .or. worst_variance >= 1.0e-9_dp) &
            print *, "    worst mean/variance error:", worst_mean, worst_variance
    end subroutine check_truncation_matches_the_closed_form

    !! The `f(x*)` coordinate against the exact probit posterior.
    subroutine check_soft_maximum_matches_the_closed_form(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 3
        real(dp) :: prior_mean(n), prior_variance(n), mean(n), variance(n)
        real(dp) :: expected_mean, expected_variance
        real(dp) :: worst_mean, worst_variance
        real(dp) :: best_values(4), sigmas(3)
        integer :: passes, a, b
        logical :: converged

        best_values = [-2.0_dp, 0.0_dp, 1.0_dp, 3.0_dp]
        sigmas = [0.1_dp, 0.5_dp, 2.0_dp]

        worst_mean = 0.0_dp
        worst_variance = 0.0_dp
        do a = 1, size(best_values)
            do b = 1, size(sigmas)
                prior_mean = [0.7_dp, -1.0_dp, -0.4_dp]
                prior_variance = [1.3_dp, 0.8_dp, 0.6_dp]

                call fortbo_pes_latent_constraints(prior_mean, prior_variance, &
                    best_values(a), sigmas(b), mean, variance, passes, &
                    converged, status)
                if (status%code /= FORTNUM_OK) cycle

                call exact_soft_maximum(prior_mean(1), prior_variance(1), &
                    best_values(a), sigmas(b), expected_mean, expected_variance)
                worst_mean = max(worst_mean, abs(mean(1) - expected_mean))
                worst_variance = max(worst_variance, &
                    abs(variance(1) - expected_variance))
            end do
        end do

        call expect(worst_mean < 1.0e-9_dp, &
            "the f(x*) marginal matches the exact probit posterior mean", &
            failures)
        call expect(worst_variance < 1.0e-9_dp, &
            "and its exact variance", failures)
        if (worst_mean >= 1.0e-9_dp .or. worst_variance >= 1.0e-9_dp) &
            print *, "    worst mean/variance error:", worst_mean, worst_variance
    end subroutine check_soft_maximum_matches_the_closed_form

    !! The constraints must actually do something, and in the paper's
    !! direction. The paper maximizes: `x*` is a local maximum, so the Hessian
    !! diagonal is pushed negative and `f(x*)` is pushed above the incumbent.
    subroutine check_the_constraints_bind(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 4
        real(dp) :: prior_mean(n), prior_variance(n), mean(n), variance(n)
        integer :: passes, k
        logical :: converged, pushed_negative, sharpened

        ! A prior that violates every constraint: a positive Hessian diagonal
        ! and an `f(x*)` below the incumbent.
        prior_mean = [-1.0_dp, 1.5_dp, 2.0_dp, 0.5_dp]
        prior_variance = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, &
            2.0_dp, 0.5_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_OK, "the EP pass runs", failures)

        pushed_negative = .true.
        sharpened = .true.
        do k = 2, n
            if (mean(k) >= prior_mean(k)) pushed_negative = .false.
            ! Conditioning on a constraint cannot increase the variance.
            if (variance(k) > prior_variance(k) + 1.0e-12_dp) sharpened = .false.
        end do
        call expect(pushed_negative, &
            "the Hessian diagonal is pushed toward the negative half line", &
            failures)
        call expect(sharpened, &
            "conditioning never inflates a Hessian variance", failures)

        call expect(mean(1) > prior_mean(1), &
            "f(x*) is pushed above its prior mean by the incumbent", failures)
        call expect(variance(1) <= prior_variance(1) + 1.0e-12_dp, &
            "and its variance does not inflate", failures)
    end subroutine check_the_constraints_bind

    !! Deep in the tail both the normal density and its integral underflow, so
    !! their quotient is `NaN` while the ratio itself is well behaved. Every EP
    !! update divides by that ratio, so a `NaN` would not stay local -- it
    !! would poison every coordinate through the shared iteration.
    subroutine check_deep_tails_stay_finite(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 4
        real(dp) :: prior_mean(n), prior_variance(n), mean(n), variance(n)
        integer :: passes, k
        logical :: converged, finite, ordered

        ! Hessian entries enormously far into the forbidden half line, and an
        ! `f(x*)` hopelessly below the incumbent.
        prior_mean = [-60.0_dp, 40.0_dp, 80.0_dp, 25.0_dp]
        prior_variance = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, &
            50.0_dp, 0.5_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_OK, "the extreme case runs", failures)

        finite = .true.
        ordered = .true.
        do k = 1, n
            ! The only portable NaN test: a NaN is not equal to itself.
            if (mean(k) /= mean(k) .or. variance(k) /= variance(k)) finite = .false.
            if (abs(mean(k)) > 1.0e100_dp) finite = .false.
            if (variance(k) <= 0.0_dp) ordered = .false.
        end do
        call expect(finite, &
            "no marginal becomes NaN or unbounded deep in the tail", failures)
        call expect(ordered, "every variance stays positive", failures)

        ! And the answer is still the right shape: the truncated coordinates
        ! must sit below zero however far the prior started above it.
        call expect(all(mean(2:) < 0.0_dp), &
            "even hopeless Hessian priors end up negative", failures)
    end subroutine check_deep_tails_stay_finite

    !! The closed forms above and the module now share their algebra, so on
    !! their own they would check the EP bookkeeping -- cavity removal, site
    !! combination, convergence -- but not the moments themselves. This checks
    !! the moments against numerical integration of the actual definition,
    !! which shares nothing with either.
    subroutine check_against_quadrature(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 3
        real(dp) :: prior_mean(n), prior_variance(n), mean(n), variance(n)
        real(dp) :: numeric_mean, numeric_variance
        integer :: passes, k
        logical :: converged, agrees

        prior_mean = [0.6_dp, 0.9_dp, -1.1_dp]
        prior_variance = [1.4_dp, 0.7_dp, 1.9_dp]

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, &
            0.2_dp, 0.4_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_OK, "the EP pass runs", failures)

        agrees = .true.
        ! Coordinate one carries the soft maximum, the rest the truncation.
        call integrate_marginal(prior_mean(1), prior_variance(1), .true., &
            0.2_dp, 0.4_dp, numeric_mean, numeric_variance)
        if (abs(mean(1) - numeric_mean) > 1.0e-7_dp) agrees = .false.
        if (abs(variance(1) - numeric_variance) > 1.0e-7_dp) agrees = .false.
        do k = 2, n
            call integrate_marginal(prior_mean(k), prior_variance(k), .false., &
                0.2_dp, 0.4_dp, numeric_mean, numeric_variance)
            if (abs(mean(k) - numeric_mean) > 1.0e-7_dp) agrees = .false.
            if (abs(variance(k) - numeric_variance) > 1.0e-7_dp) agrees = .false.
        end do

        call expect(agrees, &
            "the EP marginals match numerical integration of the constraints", &
            failures)
    end subroutine check_against_quadrature

    !! Mean and variance of `N(z|mu,v) * factor(z)`, by direct integration.
    subroutine integrate_marginal(mu, v, soft_maximum, threshold, sigma, &
            mean, variance)
        real(dp), intent(in) :: mu, v, threshold, sigma
        logical, intent(in) :: soft_maximum
        real(dp), intent(out) :: mean, variance
        integer, parameter :: steps = 400000
        real(dp) :: sd, low, high, step, z, weight, density
        real(dp) :: mass, first, second
        integer :: k

        sd = sqrt(v)
        low = mu - 12.0_dp*sd
        high = mu + 12.0_dp*sd
        ! The truncation factor is a step, and integrating a trapezoid rule
        ! straight across a jump discontinuity converges only as O(h) -- which
        ! at this resolution is around 1e-4 and swamps the quantity being
        ! checked. Integrating up to the jump instead of across it restores
        ! the smooth-integrand convergence rate.
        if (.not. soft_maximum) then
            high = 0.0_dp
            if (low >= high) low = high - 24.0_dp*sd
        end if
        step = (high - low)/real(steps, dp)

        mass = 0.0_dp
        first = 0.0_dp
        second = 0.0_dp
        do k = 0, steps
            z = low + real(k, dp)*step
            density = standard_pdf((z - mu)/sd)/sd
            if (soft_maximum) then
                weight = standard_cdf((z - threshold)/sigma)
            else
                ! The domain already stops at the jump, so the factor is one
                ! throughout it.
                weight = 1.0_dp
            end if
            ! Trapezoid: the endpoints carry half weight.
            if (k == 0 .or. k == steps) then
                density = 0.5_dp*density
            end if
            mass = mass + density*weight*step
            first = first + z*density*weight*step
            second = second + z*z*density*weight*step
        end do

        mean = first/mass
        variance = second/mass - mean*mean
    end subroutine integrate_marginal

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prior_mean(3), prior_variance(3), mean(3), variance(3)
        real(dp) :: short(2)
        integer :: passes
        logical :: converged

        prior_mean = [0.5_dp, -1.0_dp, -0.5_dp]
        prior_variance = [1.0_dp, 1.0_dp, 1.0_dp]

        call fortbo_pes_latent_constraints(prior_mean, [1.0_dp, 0.0_dp, 1.0_dp], &
            0.0_dp, 0.5_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero prior variance is refused", failures)

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, 0.0_dp, &
            0.0_dp, mean, variance, passes, converged, status)
        ! Zero noise is the *hard* constraint, which is a different algorithm
        ! rather than a limiting case of this one.
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero soft-constraint scale is refused, not silently hardened", &
            failures)

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, 0.0_dp, &
            0.5_dp, short, variance, passes, converged, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched output buffers are refused", failures)

        call fortbo_pes_latent_constraints(prior_mean, prior_variance, 0.0_dp, &
            0.5_dp, mean, variance, passes, converged, status)
        call expect(status%code == FORTNUM_OK .and. &
            passes >= 1 .and. passes <= FORTBO_PES_EP_MAX_PASSES, &
            "the pass count is reported and within its cap", failures)
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

end program test_pes_constraints
