module fortbo_risk
    !! Risk-sensitive and multi-fidelity acquisitions (ROADMAP BO1).
    !!
    !! **Risk-sensitive.** The ordinary acquisition ranks by the posterior mean
    !! and rewards uncertainty, which is right while exploring and wrong when
    !! the answer will be *used*. A design that is excellent in expectation and
    !! occasionally catastrophic is not a good design. Three utilities are
    !! offered, all under a minimization convention where a lower score is
    !! better:
    !!
    !!   * mean-variance, `mu + lambda sigma^2`. Simple and differentiable, but
    !!     it penalizes upside and downside alike, since variance does not know
    !!     which tail it is in;
    !!   * value at risk, `mu + sigma Phi^-1(alpha)`, the `alpha` quantile. It
    !!     asks how bad the outcome is at a given confidence, and it ignores
    !!     everything beyond that point entirely;
    !!   * conditional value at risk, `mu + sigma phi(z_alpha)/(1 - alpha)`, the
    !!     mean of the worst `1 - alpha` of outcomes. It is the one that sees
    !!     the tail it is protecting against, and it is coherent where VaR is
    !!     not — VaR can *rise* when two risks are combined, which is not a
    !!     property anyone wants in an objective.
    !!
    !! CVaR is at least VaR by construction, and the test asserts that ordering
    !! rather than trusting the formulas to agree.
    !!
    !! **Multi-fidelity.** A cheap approximation is worth evaluating only if it
    !! says something about the expensive target. Two numbers decide that: the
    !! correlation `rho` between the fidelity and the target, and the cost of
    !! evaluating there. The acquisition is
    !!
    !!     mf(x, m) = base(x) * rho_m^2 / cost_m^alpha.
    !!
    !! `rho` is squared because information about a Gaussian scales with the
    !! square of correlation — the variance explained, not the correlation
    !! itself. Using `rho` unsquared would over-value weakly correlated
    !! fidelities in exactly the regime where they mislead, and this is the
    !! error the test is written to catch.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortbo_normal, only: fortbo_inverse_normal
    implicit none
    private

    public :: fortbo_mean_variance
    public :: fortbo_value_at_risk
    public :: fortbo_conditional_value_at_risk
    public :: fortbo_multi_fidelity_acquisition

contains

    !! `mu + lambda sigma^2`, lower is better.
    subroutine fortbo_mean_variance(mean, variance, lambda, value, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: variance(:)
        real(dp), intent(in) :: lambda
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (size(variance) /= size(mean) .or. size(value) /= size(mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: mean and variance shapes disagree")
            return
        end if
        if (any(variance < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: variance must not be negative")
            return
        end if
        if (lambda < 0.0_dp) then
            ! A negative lambda rewards variance, which is risk *seeking*. It is
            ! a coherent thing to want and a very unlikely thing to have meant,
            ! so it is refused rather than accepted silently.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: lambda must not be negative")
            return
        end if

        value = mean + lambda*variance
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_mean_variance

    !! The `alpha` quantile of the posterior: the outcome that is not exceeded
    !! with probability `alpha`.
    subroutine fortbo_value_at_risk(mean, sd, alpha, value, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: alpha
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: quantile

        value = 0.0_dp
        call check_tail_inputs(mean, sd, alpha, size(value), status)
        if (status%code /= FORTNUM_OK) return

        quantile = fortbo_inverse_normal(alpha)
        value = mean + sd*quantile
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_value_at_risk

    !! The mean of the worst `1 - alpha` of outcomes.
    !!
    !! For a normal this is `mu + sigma phi(z)/(1 - alpha)` with `z` the
    !! `alpha` quantile — a closed form, so no sampling and no quadrature.
    subroutine fortbo_conditional_value_at_risk(mean, sd, alpha, value, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: alpha
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: quantile, density

        value = 0.0_dp
        call check_tail_inputs(mean, sd, alpha, size(value), status)
        if (status%code /= FORTNUM_OK) return

        quantile = fortbo_inverse_normal(alpha)
        density = exp(-0.5_dp*quantile*quantile)/sqrt(8.0_dp*atan(1.0_dp))
        value = mean + sd*density/(1.0_dp - alpha)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_conditional_value_at_risk

    pure subroutine check_tail_inputs(mean, sd, alpha, n_out, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: alpha
        integer, intent(in) :: n_out
        type(fortnum_status_t), intent(out) :: status

        if (size(sd) /= size(mean) .or. n_out /= size(mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: mean and standard deviation shapes disagree")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: standard deviation must not be negative")
            return
        end if
        if (alpha <= 0.0_dp .or. alpha >= 1.0_dp) then
            ! The endpoints are not merely awkward: at one the tail is empty and
            ! the conditional mean is undefined, at zero the "worst outcomes"
            ! are all of them.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo risk: alpha must lie strictly inside (0, 1)")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_tail_inputs

    !! `base * rho^2 / cost^alpha` for one fidelity.
    subroutine fortbo_multi_fidelity_acquisition(base, correlation, cost, alpha, &
            value, status)
        real(dp), intent(in) :: base(:)
        !! Correlation of this fidelity with the target fidelity.
        real(dp), intent(in) :: correlation(:)
        real(dp), intent(in) :: cost(:)
        real(dp), intent(in) :: alpha
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        value = 0.0_dp
        if (size(correlation) /= size(base) .or. size(cost) /= size(base) .or. &
            size(value) /= size(base)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-fidelity: array shapes disagree")
            return
        end if
        if (any(abs(correlation) > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-fidelity: correlation must lie in [-1, 1]")
            return
        end if
        if (any(cost <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-fidelity: costs must be positive")
            return
        end if
        if (alpha < 0.0_dp .or. alpha > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-fidelity: alpha must lie in [0, 1]")
            return
        end if
        if (any(base < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-fidelity: base acquisition must be non-negative")
            return
        end if

        do i = 1, size(base)
            ! Squared: information about a Gaussian scales with variance
            ! explained, not with correlation. A negative correlation is just as
            ! informative as a positive one of the same magnitude, which the
            ! square also handles.
            value(i) = base(i)*correlation(i)**2/cost(i)**alpha
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_multi_fidelity_acquisition

end module fortbo_risk
