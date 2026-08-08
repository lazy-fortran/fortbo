module fortbo_pes
    !! Predictive entropy search (ROADMAP BO1).
    !!
    !! Written against Hernandez-Lobato, Hoffman and Ghahramani,
    !! *Predictive Entropy Search for Efficient Global Optimization*
    !! (arXiv:1406.2541), read rather than recalled. The paper is fetched by
    !! `fortbo-bench/scripts/fetch_provenance.py`; nothing is transcribed from
    !! it, but the constraints below are named as it names them so the two can
    !! be compared.
    !!
    !! Conditioning on the optimum's *location* is intractable — it asserts
    !! `f(z) >= f(x*)` for every `z` at once — so the paper replaces it with
    !! three simplified constraints:
    !!
    !!   * **C1**: `x*` is a local optimum, via `grad f(x*) = 0` and a
    !!     definiteness condition on `diag[hess f(x*)]`;
    !!   * **C2**: `f(x*)` is better than every past observation, softened to
    !!     account for observation noise;
    !!   * **C3**: `f(x)` is worse than `f(x*)` — conditioning only on the
    !!     query at hand.
    !!
    !! **FortBO implements C3.** C1 and C2 need expectation propagation over a
    !! latent vector holding `f(x*)` and the Hessian diagonal, and that is not
    !! here. Naming them makes the gap checkable instead of leaving a reader to
    !! infer it from behaviour.
    !!
    !! Under C3 the paper approximates the truncated predictive by a Gaussian of
    !! matched variance, which makes the entropy closed form rather than a
    !! quadrature. With `f = [f(x); f(x*)]` jointly normal with mean `m` and
    !! covariance `V`,
    !!
    !!     s    = V11 + V22 - 2 V12,
    !!     alpha = (m1 - m2) / sqrt(s),        (minimization; the paper
    !!                                          maximizes and has m2 - m1)
    !!     beta  = phi(alpha) / Phi(alpha),
    !!     v     = V11 - beta (beta + alpha) (V11 - V12)^2 / s,
    !!     H     = 0.5 log(2 pi e (v + noise)).
    !!
    !! `s` collapses toward zero as the query approaches `x*`, where `V11` and
    !! `V12` coincide, and the formula is then dividing two vanishing
    !! quantities. The paper's own remedy is applied: `V12` is scaled by the
    !! largest factor in `[0, 1]` that keeps `s` above a floor, which reads as
    !! slightly loosening the dependence between `f(x)` and `f(x*)` exactly
    !! where they are nearly the same variable.
    !!
    !! `fortbo_pes_conditional_entropy` keeps the earlier quadrature over the
    !! true truncated density. It is not dead code: comparing it against the
    !! moment-matched form *measures* the approximation the paper makes, rather
    !! than leaving its size unstated.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_pes_conditional_entropy
    public :: fortbo_pes_matched_variance
    public :: fortbo_predictive_entropy_search

    !! Quadrature half-width in standard deviations, and interval count. Eight
    !! standard deviations leaves under 1e-15 of the mass outside, below the
    !! quadrature error at this resolution.
    real(dp), parameter, public :: FORTBO_PES_QUADRATURE_WIDTH = 8.0_dp
    integer, parameter, public :: FORTBO_PES_INTERVALS = 2000

    !! Below this probability the conditioning event is numerically impossible
    !! and its entropy is meaningless.
    real(dp), parameter, public :: FORTBO_PES_MASS_FLOOR = 1.0e-12_dp

    !! Floor on `s = V11 + V22 - 2 V12`, below which the matched-variance
    !! formula divides two vanishing quantities. The paper uses 1e-10 and so
    !! does this.
    real(dp), parameter, public :: FORTBO_PES_S_FLOOR = 1.0e-10_dp

contains

    !! Variance of `f(x)` under C3, by the paper's moment matching.
    !!
    !! `correlation_scale` reports the factor applied to `V12` to keep `s` above
    !! its floor: one when nothing was needed, less than one when the query sat
    !! close enough to `x*` that the two are nearly the same variable. Reporting
    !! it rather than applying it silently is what lets a caller tell an honest
    !! variance from a rescued one.
    subroutine fortbo_pes_matched_variance(mean, sd, star_mean, star_sd, &
            covariance, variance, correlation_scale, status)
        real(dp), intent(in) :: mean, sd, star_mean, star_sd, covariance
        real(dp), intent(out) :: variance
        real(dp), intent(out) :: correlation_scale
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: v11, v22, v12, s, alpha, beta, cdf, pdf, allowed

        variance = 0.0_dp
        correlation_scale = 1.0_dp
        if (sd <= 0.0_dp .or. star_sd <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: standard deviations must be positive")
            return
        end if

        v11 = sd*sd
        v22 = star_sd*star_sd
        v12 = covariance
        s = v11 + v22 - 2.0_dp*v12
        if (s < FORTBO_PES_S_FLOOR) then
            ! Shrink the dependence just enough to restore a usable `s`. The
            ! largest admissible scale solves v11 + v22 - 2 k v12 = floor.
            if (v12 == 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo pes: the pair has no usable spread")
                return
            end if
            allowed = (v11 + v22 - FORTBO_PES_S_FLOOR)/(2.0_dp*v12)
            correlation_scale = min(max(allowed, 0.0_dp), 1.0_dp)
            v12 = correlation_scale*v12
            s = v11 + v22 - 2.0_dp*v12
        end if
        if (s <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: the pair has no usable spread")
            return
        end if

        ! FortBO minimizes, so the constraint is f(x) >= f(x*) and alpha carries
        ! m1 - m2. The paper maximizes and has the opposite sign.
        alpha = (mean - star_mean)/sqrt(s)
        cdf = 0.5_dp*(1.0_dp + erf(alpha/sqrt(2.0_dp)))
        if (cdf <= FORTBO_PES_MASS_FLOOR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pes: the conditioning event has no usable mass")
            return
        end if
        pdf = exp(-0.5_dp*alpha*alpha)/sqrt(8.0_dp*atan(1.0_dp))
        beta = pdf/cdf

        variance = v11 - beta*(beta + alpha)*(v11 - v12)**2/s
        ! A truncation cannot raise the variance, and cannot make it negative;
        ! either outcome here is rounding rather than a finding.
        variance = min(max(variance, 0.0_dp), v11)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_pes_matched_variance

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
        real(dp) :: matched, scale
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
                call fortbo_pes_matched_variance(mean(i), sd(i), &
                    star_mean(i, k), star_sd(i, k), covariance(i, k), &
                    matched, scale, local)
                if (local%code == FORTNUM_OK) then
                    if (matched <= 0.0_dp) then
                        conditional = -huge(1.0_dp)
                    else
                        conditional = 0.5_dp*log(2.0_dp*exp(1.0_dp) &
                            *4.0_dp*atan(1.0_dp)) + 0.5_dp*log(matched)
                    end if
                end if
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
