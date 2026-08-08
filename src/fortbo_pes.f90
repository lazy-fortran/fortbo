module fortbo_pes
    !! Predictive entropy search (ROADMAP BO1).
    !!
    !! Max-value entropy search asks how much an observation tells us about the
    !! optimum's *value*. Predictive entropy search asks about its *location*,
    !! which is the quantity a run actually wants: knowing the optimum is near
    !! -3.7 is worth much less than knowing where it is.
    !!
    !!     PES(x) = H[p(f(x))] - E_{x*}[ H[p(f(x) | x*)] ]
    !!
    !! The expectation runs over sampled minimizer *locations*. Conditioning on
    !! `x*` being the global minimizer is what makes this hard: the exact
    !! condition is `f(x) >= f(x*)` for every `x` at once, plus a vanishing
    !! gradient and a positive semidefinite Hessian at `x*`. The published
    !! method approximates all of that with expectation propagation.
    !!
    !! **This module is an ingredient of PES, not PES.** It keeps only the
    !! constraint that a single query's marginal can express — `f(x) >= f(x*)` —
    !! and computes the resulting entropy *exactly for that constraint*, by
    !! quadrature over a stated integral rather than by moment matching.
    !!
    !! That reduction does not reproduce PES's location-awareness, and the
    !! measurement says so plainly. The weighting factor's dependence on `t` has
    !! slope `sqrt((1 - rho)/(1 + rho))` when the two standard deviations agree,
    !! so a query *strongly* correlated with the sampled minimizer gets a
    !! *flatter* weighting and therefore a smaller entropy reduction — the
    !! opposite of what a location-aware acquisition must do. The information
    !! PES actually extracts comes from conditioning the whole posterior on `x*`
    !! being a stationary minimum, which couples queries and needs expectation
    !! propagation; none of that is here.
    !!
    !! What is here is correct and useful on its own: the conditional density
    !! and its entropy under a minimum constraint, validated against simulation.
    !! Building the full estimator on top means adding the joint conditioning,
    !! not tuning this.
    !!
    !! With `(f(x), f(x*))` jointly normal, the conditional density of `f(x)`
    !! given `f(x) >= f(x*)` is
    !!
    !!     p(t) = phi((t - mu)/sigma)/sigma * Phi(a(t)) / Z,
    !!     a(t) = (t - mu_star - rho (sigma_star/sigma) (t - mu))
    !!            / (sigma_star sqrt(1 - rho^2)),
    !!     Z    = Phi((mu - mu_star) / sqrt(sigma^2 + sigma_star^2 - 2 c)),
    !!
    !! and its entropy is integrated directly.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_pes_conditional_entropy
    public :: fortbo_predictive_entropy_search

    !! Quadrature half-width in standard deviations, and interval count. Eight
    !! standard deviations leaves under 1e-15 of the mass outside, below the
    !! quadrature error at this resolution.
    real(dp), parameter, public :: FORTBO_PES_QUADRATURE_WIDTH = 8.0_dp
    integer, parameter, public :: FORTBO_PES_INTERVALS = 2000

    !! Below this probability the conditioning event is numerically impossible
    !! and its entropy is meaningless.
    real(dp), parameter, public :: FORTBO_PES_MASS_FLOOR = 1.0e-12_dp

contains

    !! Differential entropy of `f(x)` given `f(x) >= f(x*)`, by Simpson's rule.
    subroutine fortbo_pes_conditional_entropy(mean, sd, star_mean, star_sd, &
            covariance, entropy, status)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: sd
        real(dp), intent(in) :: star_mean
        real(dp), intent(in) :: star_sd
        real(dp), intent(in) :: covariance
        real(dp), intent(out) :: entropy
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: rho, spread, normalizer, conditional_sd
        real(dp) :: lower, upper, h, t, density, weighted, total
        real(dp) :: argument, term
        integer :: k

        entropy = 0.0_dp
        if (sd <= 0.0_dp .or. star_sd <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: standard deviations must be positive")
            return
        end if
        rho = covariance/(sd*star_sd)
        if (abs(rho) >= 1.0_dp) then
            ! Perfect correlation makes the pair degenerate: the conditional
            ! density is a point mass or the constraint is vacuous, and neither
            ! has a differential entropy.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: the query and the minimizer are perfectly correlated")
            return
        end if

        spread = sqrt(sd*sd + star_sd*star_sd - 2.0_dp*covariance)
        if (spread <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: the difference has no spread")
            return
        end if
        ! P(f(x) >= f(x*)) = P(D >= 0) with D = f(x) - f(x*), whose mean is
        ! mean - star_mean. Writing the difference the other way round gives
        ! P(D <= 0) and the density then fails to integrate to one.
        normalizer = 0.5_dp*(1.0_dp + erf((mean - star_mean)/(spread*sqrt(2.0_dp))))
        if (normalizer <= FORTBO_PES_MASS_FLOOR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: the conditioning event has no usable mass")
            return
        end if

        conditional_sd = star_sd*sqrt(1.0_dp - rho*rho)
        lower = mean - FORTBO_PES_QUADRATURE_WIDTH*sd
        upper = mean + FORTBO_PES_QUADRATURE_WIDTH*sd
        h = (upper - lower)/real(FORTBO_PES_INTERVALS, dp)

        total = 0.0_dp
        do k = 0, FORTBO_PES_INTERVALS
            t = lower + h*real(k, dp)
            density = exp(-0.5_dp*((t - mean)/sd)**2)/(sd*sqrt(8.0_dp*atan(1.0_dp)))
            argument = (t - star_mean - rho*(star_sd/sd)*(t - mean))/conditional_sd
            weighted = density*0.5_dp*(1.0_dp + erf(argument/sqrt(2.0_dp))) &
                /normalizer
            if (weighted > 0.0_dp) then
                term = -weighted*log(weighted)
            else
                term = 0.0_dp
            end if
            if (k == 0 .or. k == FORTBO_PES_INTERVALS) then
                total = total + term
            else if (mod(k, 2) == 1) then
                total = total + 4.0_dp*term
            else
                total = total + 2.0_dp*term
            end if
        end do
        entropy = total*h/3.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_pes_conditional_entropy

    !! PES at each candidate, averaged over the supplied minimizer samples.
    !!
    !! `star_mean`, `star_sd`, and `covariance` are indexed
    !! `(candidate, sample)`: the correlation between a query and a sampled
    !! minimizer depends on both, which is exactly the location information the
    !! value-based acquisition throws away.
    subroutine fortbo_predictive_entropy_search(mean, sd, star_mean, star_sd, &
            covariance, value, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: star_mean(:, :)
        real(dp), intent(in) :: star_sd(:, :)
        real(dp), intent(in) :: covariance(:, :)
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        type(fortnum_status_t) :: local
        real(dp) :: full_entropy, conditional, contribution
        integer :: n, m, i, k, used

        value = 0.0_dp
        n = size(mean)
        if (n < 1 .or. size(sd) /= n .or. size(value) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: moment shapes disagree")
            return
        end if
        m = size(star_mean, 2)
        if (m < 1 .or. size(star_mean, 1) /= n .or. &
            any(shape(star_sd) /= shape(star_mean)) .or. &
            any(shape(covariance) /= shape(star_mean))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: minimizer-sample shapes disagree")
            return
        end if
        if (any(sd < 0.0_dp) .or. any(star_sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: standard deviations must not be negative")
            return
        end if

        do i = 1, n
            if (sd(i) <= 0.0_dp) then
                value(i) = 0.0_dp
                cycle
            end if
            full_entropy = 0.5_dp*log(2.0_dp*exp(1.0_dp)*4.0_dp*atan(1.0_dp)) &
                + log(sd(i))
            used = 0
            do k = 1, m
                call fortbo_pes_conditional_entropy(mean(i), sd(i), &
                    star_mean(i, k), star_sd(i, k), &
                    covariance(i, k), conditional, &
                    local)
                ! A sample whose conditioning event is degenerate contributes
                ! nothing rather than aborting the whole batch: one impossible
                ! minimizer sample should not silently discard the others.
                if (local%code /= FORTNUM_OK) cycle
                contribution = full_entropy - conditional
                ! Mutual information is non-negative; a negative value here is
                ! quadrature error, not a finding.
                value(i) = value(i) + max(contribution, 0.0_dp)
                used = used + 1
            end do
            if (used > 0) then
                value(i) = value(i)/real(used, dp)
            else
                value(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_predictive_entropy_search

end module fortbo_pes
