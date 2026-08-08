module fortbo_preference
    !! Preference learning and noisy dominance (ROADMAP BO4).
    !!
    !! Two related problems share this module because they share one primitive.
    !!
    !! *Preference learning* fits a latent objective to pairwise judgements
    !! rather than to values. An observer who can say "I prefer a to b" but
    !! cannot put a number on either is common — human raters, expensive
    !! adjudicated comparisons, simulations scored only relative to each other.
    !! FortBO minimizes, so `a` is preferred to `b` when its latent value is
    !! lower. Under independent Gaussian judgement noise of scale `s` on each
    !! latent value the difference of the two noisy judgements is Gaussian with
    !! standard deviation sqrt(2) s, giving the Thurstone-Mosteller model
    !!
    !!     P(a preferred to b) = Phi((f_b - f_a) / (sqrt(2) s)).
    !!
    !! *Noisy dominance* asks the same question of a multi-objective posterior:
    !! given independent normal beliefs about each objective at two points, what
    !! is the probability that one actually dominates the other? Per objective
    !! that is again a Gaussian tail probability, with the two posterior
    !! standard deviations combined in quadrature. Deciding dominance by
    !! comparing posterior means instead throws away exactly the information
    !! that says whether the comparison is trustworthy, which is what makes a
    !! Pareto front built from noisy observations fill up with points that were
    !! never actually good.
    !!
    !! Neither probability nor any of its derivatives is written here: they come
    !! from `src/generated/fortbo_generated_preference_leaf.f90`, derived by
    !! FortSym from the definition above.
    !!
    !! Independence is assumed across objectives and, in
    !! `fortbo_probability_non_dominated`, across front members. The first
    !! holds for the independent-output surrogates in this package; the second
    !! is an approximation and is documented rather than hidden, because the
    !! exact quantity requires integrating over the joint posterior of the whole
    !! front.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_preference_probability
    public :: fortbo_preference_log_probability
    public :: fortbo_preference_log_likelihood
    public :: fortbo_probability_dominates
    public :: fortbo_probability_non_dominated

    !! Smallest judgement noise scale that still yields a finite gradient. At
    !! zero the model degenerates to a step function whose derivative does not
    !! exist, so it is refused rather than regularized silently.
    real(dp), parameter, public :: FORTBO_PREFERENCE_SCALE_FLOOR = 1.0e-300_dp

    !! Beyond this argument `erfc` underflows to zero and its logarithm becomes
    !! -infinity where the true value is merely large and negative. The
    !! asymptotic branch takes over there. erfc(27) is about 1e-318, so the
    !! switch happens well inside the range where the series is accurate to far
    !! better than double precision.
    real(dp), parameter :: LOG_ERFC_ASYMPTOTIC_X = 20.0_dp

    interface
        pure subroutine fortbo_generated_preference_leaf(f_a, f_b, scale, &
                probability, probability_d_f_a, probability_d_f_b, &
                probability_d_scale, log_probability, log_probability_d_f_a, &
                log_probability_d_f_b, log_probability_d_scale)
            import :: dp
            real(dp), intent(in) :: f_a, f_b, scale
            real(dp), intent(out) :: probability, probability_d_f_a
            real(dp), intent(out) :: probability_d_f_b, probability_d_scale
            real(dp), intent(out) :: log_probability, log_probability_d_f_a
            real(dp), intent(out) :: log_probability_d_f_b
            real(dp), intent(out) :: log_probability_d_scale
        end subroutine fortbo_generated_preference_leaf
    end interface

contains

    !! Probability that `a` is preferred to `b`, i.e. that the noisy judgement
    !! ranks the lower latent value first.
    pure function fortbo_preference_probability(f_a, f_b, scale) result(probability)
        real(dp), intent(in) :: f_a
        real(dp), intent(in) :: f_b
        real(dp), intent(in) :: scale
        real(dp) :: probability
        real(dp) :: d_f_a, d_f_b, d_scale
        real(dp) :: log_probability, log_d_f_a, log_d_f_b, log_d_scale

        if (scale < FORTBO_PREFERENCE_SCALE_FLOOR) then
            ! The noiseless limit: a step at f_a == f_b. The tie is the only
            ! place the limit is ambiguous, and one half is its symmetric value.
            if (f_a < f_b) then
                probability = 1.0_dp
            else if (f_a > f_b) then
                probability = 0.0_dp
            else
                probability = 0.5_dp
            end if
            return
        end if

        call fortbo_generated_preference_leaf(f_a, f_b, scale, probability, &
            d_f_a, d_f_b, d_scale, log_probability, log_d_f_a, log_d_f_b, &
            log_d_scale)
    end function fortbo_preference_probability

    !! Log of the same probability, together with its gradient with respect to
    !! the two latent values. Far into the tail the generated expression
    !! underflows, so an asymptotic branch takes over; see
    !! `LOG_ERFC_ASYMPTOTIC_X`. The branch reproduces the generated value to
    !! within rounding at the switch point, which the test checks rather than
    !! assumes.
    pure subroutine fortbo_preference_log_probability(f_a, f_b, scale, &
            log_probability, d_f_a, d_f_b)
        real(dp), intent(in) :: f_a
        real(dp), intent(in) :: f_b
        real(dp), intent(in) :: scale
        real(dp), intent(out) :: log_probability
        real(dp), intent(out) :: d_f_a
        real(dp), intent(out) :: d_f_b
        real(dp) :: probability, p_d_f_a, p_d_f_b, p_d_scale, log_d_scale
        real(dp) :: x, spread, ratio

        ! `x` is the argument the generated leaf passes to `erfc`. Phi(z) is
        ! erfc(-z/sqrt(2))/2 and z = (f_b - f_a)/(sqrt(2) scale), so the two
        ! factors of sqrt(2) combine and the divisor is 2 scale.
        spread = 2.0_dp*scale
        x = (f_a - f_b)/spread
        if (x > LOG_ERFC_ASYMPTOTIC_X) then
            ! log erfc(x) = -x^2 - log(x sqrt(pi)) + log(1 - 1/(2x^2) + ...).
            ! Two correction terms are kept; at x >= 20 the first omitted term
            ! is below 1e-6 relative and the whole correction is itself tiny.
            ratio = 1.0_dp/(2.0_dp*x*x)
            ! The probability is erfc(x)/2, so the halving carries through the
            ! logarithm as a -log(2) that the bare erfc series does not supply.
            log_probability = -x*x - log(x*sqrt(4.0_dp*atan(1.0_dp))) &
                + log(1.0_dp - ratio + 3.0_dp*ratio*ratio) - log(2.0_dp)
            ! d/dx log erfc(x) = -2 exp(-x^2) / (sqrt(pi) erfc(x)), and in this
            ! regime erfc(x) ~ exp(-x^2)/(x sqrt(pi)), so the ratio collapses to
            ! -2x. The next correction is O(1/x), retained to first order.
            d_f_a = (-2.0_dp*x - 1.0_dp/x)/spread
            d_f_b = -d_f_a
            return
        end if

        call fortbo_generated_preference_leaf(f_a, f_b, scale, probability, &
            p_d_f_a, p_d_f_b, p_d_scale, log_probability, d_f_a, d_f_b, &
            log_d_scale)
    end subroutine fortbo_preference_log_probability

    !! Total log-likelihood of a set of pairwise judgements under the latent
    !! values, with its gradient. `winners(k)` was judged better than
    !! `losers(k)`. The gradient is what a fitter needs; it is accumulated from
    !! the generated derivatives so the score and the objective can never drift
    !! apart.
    subroutine fortbo_preference_log_likelihood(latent, winners, losers, scale, &
            log_likelihood, gradient, status)
        real(dp), intent(in) :: latent(:)
        integer, intent(in) :: winners(:)
        integer, intent(in) :: losers(:)
        real(dp), intent(in) :: scale
        real(dp), intent(out) :: log_likelihood
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: term, d_win, d_lose
        integer :: k, n

        log_likelihood = 0.0_dp
        gradient = 0.0_dp
        n = size(latent)

        if (size(winners) /= size(losers)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: winners and losers must have equal length")
            return
        end if
        if (size(gradient) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: gradient must match the latent vector")
            return
        end if
        if (scale < FORTBO_PREFERENCE_SCALE_FLOOR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: the noise scale must be positive")
            return
        end if
        do k = 1, size(winners)
            if (winners(k) < 1 .or. winners(k) > n .or. &
                losers(k) < 1 .or. losers(k) > n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo preference: comparison index out of range")
                return
            end if
            if (winners(k) == losers(k)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo preference: a point cannot be compared with itself")
                return
            end if
        end do

        do k = 1, size(winners)
            call fortbo_preference_log_probability(latent(winners(k)), &
                latent(losers(k)), scale, term, d_win, d_lose)
            log_likelihood = log_likelihood + term
            gradient(winners(k)) = gradient(winners(k)) + d_win
            gradient(losers(k)) = gradient(losers(k)) + d_lose
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_preference_log_likelihood

    !! Probability that `a` dominates `b` under independent normal posteriors.
    !! With continuous marginals a tie has probability zero, so requiring strict
    !! improvement in at least one objective costs nothing and the answer is the
    !! product of the per-objective tail probabilities.
    subroutine fortbo_probability_dominates(mean_a, sd_a, mean_b, sd_b, &
            probability, status)
        real(dp), intent(in) :: mean_a(:)
        real(dp), intent(in) :: sd_a(:)
        real(dp), intent(in) :: mean_b(:)
        real(dp), intent(in) :: sd_b(:)
        real(dp), intent(out) :: probability
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: combined
        integer :: m

        probability = 0.0_dp
        if (size(mean_a) /= size(sd_a) .or. size(mean_a) /= size(mean_b) .or. &
            size(mean_a) /= size(sd_b)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: dominance arguments must agree in size")
            return
        end if
        if (size(mean_a) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: at least one objective is required")
            return
        end if
        if (any(sd_a < 0.0_dp) .or. any(sd_b < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: standard deviations must not be negative")
            return
        end if

        probability = 1.0_dp
        do m = 1, size(mean_a)
            ! The difference of the two beliefs has variance sd_a^2 + sd_b^2,
            ! which the preference model writes as (sqrt(2) scale)^2.
            combined = sqrt(0.5_dp*(sd_a(m)**2 + sd_b(m)**2))
            probability = probability*fortbo_preference_probability(mean_a(m), &
                mean_b(m), combined)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_probability_dominates

    !! Probability that a candidate is dominated by no member of a front.
    !!
    !! Exact only when the members are independent of one another; for a shared
    !! surrogate they are not, and the result is then an approximation that
    !! understates the probability, because positively correlated members
    !! dominate the candidate together more often than independence predicts.
    !! It is reported as such rather than presented as exact.
    subroutine fortbo_probability_non_dominated(mean, sd, front_mean, front_sd, &
            probability, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        !! `front_mean(objective, member)`.
        real(dp), intent(in) :: front_mean(:, :)
        real(dp), intent(in) :: front_sd(:, :)
        real(dp), intent(out) :: probability
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: dominated
        integer :: j

        probability = 0.0_dp
        if (size(front_mean, 1) /= size(mean) .or. &
            size(front_sd, 1) /= size(mean) .or. &
            size(front_sd, 2) /= size(front_mean, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo preference: front arrays must match the candidate")
            return
        end if

        probability = 1.0_dp
        do j = 1, size(front_mean, 2)
            call fortbo_probability_dominates(front_mean(:, j), front_sd(:, j), &
                mean, sd, dominated, status)
            if (status%code /= FORTNUM_OK) return
            probability = probability*(1.0_dp - dominated)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_probability_non_dominated

end module fortbo_preference
