program test_pes
    !! BO1: predictive entropy search.
    !!
    !! Oracles:
    !!
    !!   * the conditional entropy is checked against Monte Carlo: draw the
    !!     bivariate pair, keep the draws satisfying the constraint, and estimate
    !!     the entropy from the retained sample by comparing the quadrature
    !!     density against the empirical histogram's own normalization. The
    !!     cheaper and sharper version used here integrates the *density*
    !!     against simulation rather than the entropy, since a density is
    !!     estimable to sampling error while a differential entropy from samples
    !!     is not;
    !!   * the conditional density must integrate to one, which catches a
    !!     normalizer error that an entropy comparison alone would absorb;
    !!   * the measurement showing this is an *ingredient* of PES rather than
    !!     PES: a strongly correlated query gets a flatter constraint and so a
    !!     smaller entropy reduction, which is the opposite of location-aware.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_pes, only: fortbo_pes_conditional_entropy, &
        fortbo_predictive_entropy_search, fortbo_pes_matched_variance
    implicit none

    integer :: failures

    failures = 0
    call check_conditional_density_normalizes(failures)
    call check_conditional_density_against_simulation(failures)
    call check_entropy_is_reduced_by_conditioning(failures)
    call check_correlation_carries_the_information(failures)
    call check_matched_variance_against_simulation(failures)
    call check_matched_entropy_tracks_the_exact_one(failures)
    call check_near_coincidence_is_rescued(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_pes: PASS"
    else
        print *, "test_pes: FAIL", failures
        error stop 1
    end if

contains

    !! The conditional density stated in the module, written out again here.
    real(dp) function conditional_density(t, mean, sd, star_mean, star_sd, &
            covariance) result(value)
        real(dp), intent(in) :: t, mean, sd, star_mean, star_sd, covariance
        real(dp) :: rho, spread, normalizer, conditional_sd, argument, base

        rho = covariance/(sd*star_sd)
        spread = sqrt(sd*sd + star_sd*star_sd - 2.0_dp*covariance)
        normalizer = 0.5_dp*(1.0_dp + erf((mean - star_mean)/(spread*sqrt(2.0_dp))))
        conditional_sd = star_sd*sqrt(1.0_dp - rho*rho)
        base = exp(-0.5_dp*((t - mean)/sd)**2)/(sd*sqrt(8.0_dp*atan(1.0_dp)))
        argument = (t - star_mean - rho*(star_sd/sd)*(t - mean))/conditional_sd
        value = base*0.5_dp*(1.0_dp + erf(argument/sqrt(2.0_dp)))/normalizer
    end function conditional_density

    !! A density that does not integrate to one has the wrong normalizer, and an
    !! entropy comparison would partly absorb that error rather than expose it.
    subroutine check_conditional_density_normalizes(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 20000
        real(dp) :: mean, sd, star_mean, star_sd, covariance
        real(dp) :: lower, upper, h, t, total, term
        integer :: c, k
        logical :: normalized

        normalized = .true.
        do c = 1, 3
            select case (c)
            case (1)
                mean = 0.0_dp; sd = 1.0_dp
                star_mean = 0.5_dp; star_sd = 1.0_dp; covariance = 0.3_dp
            case (2)
                mean = 2.0_dp; sd = 0.5_dp
                star_mean = 1.0_dp; star_sd = 1.5_dp; covariance = -0.2_dp
            case (3)
                mean = -1.0_dp; sd = 2.0_dp
                star_mean = 0.0_dp; star_sd = 0.8_dp; covariance = 0.9_dp
            end select

            lower = mean - 12.0_dp*sd
            upper = mean + 12.0_dp*sd
            h = (upper - lower)/real(n, dp)
            total = 0.0_dp
            do k = 0, n
                t = lower + h*real(k, dp)
                term = conditional_density(t, mean, sd, star_mean, star_sd, &
                    covariance)
                if (k == 0 .or. k == n) then
                    total = total + term
                else if (mod(k, 2) == 1) then
                    total = total + 4.0_dp*term
                else
                    total = total + 2.0_dp*term
                end if
            end do
            total = total*h/3.0_dp
            if (abs(total - 1.0_dp) > 1.0e-8_dp) normalized = .false.
        end do
        call expect(normalized, "the conditional density integrates to one", &
            failures)
    end subroutine check_conditional_density_normalizes

    !! Simulate the bivariate pair, keep the draws satisfying the constraint,
    !! and compare the retained fraction in a bin against the density's integral
    !! over that bin. This checks the density itself, which is estimable to
    !! sampling error, rather than an entropy estimated from samples, which is
    !! not.
    subroutine check_conditional_density_against_simulation(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 600000
        real(dp), parameter :: mean = 0.0_dp, sd = 1.0_dp
        real(dp), parameter :: star_mean = 0.4_dp, star_sd = 1.0_dp
        real(dp), parameter :: covariance = 0.35_dp
        real(dp) :: rho, a, b, draw_one, draw_two, query, star
        real(dp) :: bin_low, bin_high, predicted, empirical, standard_error
        integer :: k, kept, in_bin
        logical :: matches

        rho = covariance/(sd*star_sd)
        ! Cholesky of the 2x2 correlation matrix.
        a = sd
        b = star_sd

        call rng_seed(generator, int(515151, int64), status)
        kept = 0
        in_bin = 0
        bin_low = -0.5_dp
        bin_high = 0.5_dp
        do k = 1, n_samples
            call rng_normal(generator, draw_one)
            call rng_normal(generator, draw_two)
            query = mean + a*draw_one
            star = star_mean + b*(rho*draw_one + sqrt(1.0_dp - rho*rho)*draw_two)
            if (query < star) cycle
            kept = kept + 1
            if (query >= bin_low .and. query < bin_high) in_bin = in_bin + 1
        end do

        call expect(kept > 1000, "the constraint retains a usable sample", failures)
        empirical = real(in_bin, dp)/real(kept, dp)

        ! Integrate the density over the same bin.
        predicted = simpson_density(bin_low, bin_high, mean, sd, star_mean, &
            star_sd, covariance)
        standard_error = sqrt(predicted*(1.0_dp - predicted)/real(kept, dp))
        matches = abs(empirical - predicted) < 5.0_dp*standard_error
        call expect(matches, &
            "the conditional density matches the simulated constrained draws", &
            failures)
    end subroutine check_conditional_density_against_simulation

    real(dp) function simpson_density(lower, upper, mean, sd, star_mean, star_sd, &
            covariance) result(total)
        real(dp), intent(in) :: lower, upper, mean, sd, star_mean, star_sd
        real(dp), intent(in) :: covariance
        integer, parameter :: n = 4000
        real(dp) :: h, t, term
        integer :: k

        h = (upper - lower)/real(n, dp)
        total = 0.0_dp
        do k = 0, n
            t = lower + h*real(k, dp)
            term = conditional_density(t, mean, sd, star_mean, star_sd, covariance)
            if (k == 0 .or. k == n) then
                total = total + term
            else if (mod(k, 2) == 1) then
                total = total + 4.0_dp*term
            else
                total = total + 2.0_dp*term
            end if
        end do
        total = total*h/3.0_dp
    end function simpson_density

    !! Conditioning on information can only reduce entropy, so the acquisition
    !! is non-negative. This is a theorem and holds at every configuration.
    subroutine check_entropy_is_reduced_by_conditioning(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: conditional, full
        real(dp) :: covariance
        integer :: k
        logical :: reduced

        full = 0.5_dp*log(2.0_dp*exp(1.0_dp)*4.0_dp*atan(1.0_dp))
        reduced = .true.
        do k = -8, 8
            covariance = 0.1_dp*real(k, dp)
            call fortbo_pes_conditional_entropy(0.0_dp, 1.0_dp, 0.3_dp, 1.0_dp, &
                covariance, conditional, status)
            if (status%code /= FORTNUM_OK) cycle
            if (conditional > full + 1.0e-9_dp) reduced = .false.
        end do
        call expect(reduced, "conditioning never increases the entropy", failures)
    end subroutine check_entropy_is_reduced_by_conditioning

    !! The measurement that shows this is *not* full PES.
    !!
    !! Two candidates with identical marginals, one strongly correlated with the
    !! sampled minimizer and one not. A location-aware acquisition would prefer
    !! the correlated one. This estimator does the opposite, and the reason is
    !! structural: the weighting factor's slope in `t` is
    !! `sqrt((1 - rho)/(1 + rho))`, so high correlation flattens the constraint
    !! and reduces the entropy reduction.
    !!
    !! The test asserts the behaviour that actually occurs rather than the one
    !! that would be convenient, because a passing test asserting the convenient
    !! direction would have hidden that the single-query reduction cannot carry
    !! location information at all.
    subroutine check_correlation_carries_the_information(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(2), sd(2)
        real(dp) :: star_mean(2, 1), star_sd(2, 1), covariance(2, 1)
        real(dp) :: value(2)

        mean = [0.0_dp, 0.0_dp]
        sd = [1.0_dp, 1.0_dp]
        star_mean(:, 1) = [0.2_dp, 0.2_dp]
        star_sd(:, 1) = [1.0_dp, 1.0_dp]
        covariance(:, 1) = [0.8_dp, 0.0_dp]

        call fortbo_predictive_entropy_search(mean, sd, star_mean, star_sd, &
            covariance, value, status)
        call expect(status%code == FORTNUM_OK, "the conditional acquisition evaluates", &
            failures)
        call expect(all(value >= 0.0_dp), "the acquisition is never negative", &
            failures)
        call expect(value(2) > value(1), &
            "a flatter constraint reduces less entropy, as the slope predicts", &
            failures)

        ! A point with no posterior uncertainty cannot learn anything.
        call fortbo_predictive_entropy_search([0.0_dp], [0.0_dp], &
            star_mean(1:1, :), star_sd(1:1, :), covariance(1:1, :), value(1:1), &
            status)
        call expect(value(1) == 0.0_dp, &
            "a certain point carries no information", failures)
    end subroutine check_correlation_carries_the_information

    !! The paper's matched variance is the variance of the truncated
    !! distribution, so simulation of the constrained draws is the oracle. This
    !! is what separates a correct moment match from a plausible formula.
    subroutine check_matched_variance_against_simulation(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 600000
        real(dp) :: mean, sd, star_mean, star_sd, covariance
        real(dp) :: matched, scale, rho
        real(dp) :: draw_one, draw_two, query, star
        real(dp) :: total, total_squares, empirical, standard_error
        integer :: c, k, kept
        logical :: matches

        matches = .true.
        do c = 1, 3
            select case (c)
            case (1)
                mean = 0.0_dp; sd = 1.0_dp
                star_mean = 0.4_dp; star_sd = 1.0_dp; covariance = 0.35_dp
            case (2)
                mean = 1.0_dp; sd = 0.7_dp
                star_mean = 0.0_dp; star_sd = 1.2_dp; covariance = -0.3_dp
            case (3)
                mean = -0.5_dp; sd = 2.0_dp
                star_mean = -1.0_dp; star_sd = 0.5_dp; covariance = 0.4_dp
            end select

            call fortbo_pes_matched_variance(mean, sd, star_mean, star_sd, &
                covariance, matched, scale, status)
            if (status%code /= FORTNUM_OK) matches = .false.

            rho = covariance/(sd*star_sd)
            call rng_seed(generator, int(90210 + c, int64), status)
            total = 0.0_dp
            total_squares = 0.0_dp
            kept = 0
            do k = 1, n_samples
                call rng_normal(generator, draw_one)
                call rng_normal(generator, draw_two)
                query = mean + sd*draw_one
                star = star_mean + star_sd*(rho*draw_one &
                    + sqrt(1.0_dp - rho*rho)*draw_two)
                if (query < star) cycle
                kept = kept + 1
                total = total + query
                total_squares = total_squares + query*query
            end do
            empirical = total_squares/real(kept, dp) - (total/real(kept, dp))**2
            ! The variance of a sample variance is dominated by the fourth
            ! moment; for a near-Gaussian sample that is about 2 v^2 / n.
            standard_error = empirical*sqrt(2.0_dp/real(kept, dp))
            if (abs(matched - empirical) > 6.0_dp*standard_error) matches = .false.
        end do
        call expect(matches, &
            "the matched variance equals the truncated variance it approximates", &
            failures)
    end subroutine check_matched_variance_against_simulation

    !! The paper replaces the true truncated entropy by that of a Gaussian with
    !! the same variance. Comparing the two *measures* that approximation rather
    !! than leaving its size unstated: they must be close, and the matched one
    !! must be the larger, since among distributions of a given variance the
    !! Gaussian has the greatest entropy.
    subroutine check_matched_entropy_tracks_the_exact_one(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: matched, scale, exact, gaussian, gap, worst
        real(dp) :: mean, sd, star_mean, star_sd, covariance
        integer :: c
        logical :: bounded, ordered

        bounded = .true.
        ordered = .true.
        worst = 0.0_dp
        do c = 1, 3
            select case (c)
            case (1)
                mean = 0.0_dp; sd = 1.0_dp
                star_mean = 0.4_dp; star_sd = 1.0_dp; covariance = 0.35_dp
            case (2)
                mean = 1.0_dp; sd = 0.7_dp
                star_mean = 0.0_dp; star_sd = 1.2_dp; covariance = -0.3_dp
            case (3)
                mean = -0.5_dp; sd = 2.0_dp
                star_mean = -1.0_dp; star_sd = 0.5_dp; covariance = 0.4_dp
            end select

            call fortbo_pes_matched_variance(mean, sd, star_mean, star_sd, &
                covariance, matched, scale, status)
            gaussian = 0.5_dp*log(2.0_dp*exp(1.0_dp)*4.0_dp*atan(1.0_dp)) &
                + 0.5_dp*log(matched)
            call fortbo_pes_conditional_entropy(mean, sd, star_mean, star_sd, &
                covariance, exact, status)
            gap = gaussian - exact
            worst = max(worst, abs(gap))
            ! Maximum entropy at fixed variance belongs to the Gaussian.
            if (gap < -1.0e-9_dp) ordered = .false.
            if (abs(gap) > 0.1_dp) bounded = .false.
        end do
        call expect(ordered, &
            "the matched entropy is at least the true truncated entropy", failures)
        call expect(bounded, &
            "the moment-matching approximation stays under a tenth of a nat", &
            failures)
    end subroutine check_matched_entropy_tracks_the_exact_one

    !! The paper warns that `s` collapses as the query approaches `x*`, and
    !! prescribes shrinking the dependence to restore it. The rescue must fire,
    !! must be reported, and must leave a usable variance.
    subroutine check_near_coincidence_is_rescued(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: matched, scale

        ! Nearly the same variable: V11, V22 and V12 all essentially one.
        call fortbo_pes_matched_variance(0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, &
            1.0_dp - 1.0e-14_dp, matched, scale, status)
        call expect(status%code == FORTNUM_OK, &
            "a near-coincident query is rescued rather than refused", failures)
        call expect(scale < 1.0_dp, &
            "the dependence scaling is reported when it is applied", failures)
        call expect(matched >= 0.0_dp .and. matched <= 1.0_dp, &
            "the rescued variance stays inside its prior bound", failures)

        ! Away from coincidence nothing is rescaled, so a caller can tell an
        ! honest variance from a rescued one.
        call fortbo_pes_matched_variance(0.0_dp, 1.0_dp, 0.3_dp, 1.0_dp, &
            0.2_dp, matched, scale, status)
        call expect(scale == 1.0_dp, &
            "no scaling is applied when none is needed", failures)
    end subroutine check_near_coincidence_is_rescued

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: entropy, value(2)
        real(dp) :: star(2, 1), star_sd(2, 1), covariance(2, 1)

        call fortbo_pes_conditional_entropy(0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, &
            0.0_dp, entropy, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero standard deviation is refused", failures)

        ! Perfect correlation makes the pair degenerate.
        call fortbo_pes_conditional_entropy(0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, &
            1.0_dp, entropy, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a perfectly correlated pair is refused", failures)

        ! A minimizer that is certainly far above the query leaves the
        ! conditioning event with no usable mass.
        ! The event is impossible when the query is almost surely *below* the
        ! minimizer, not above it.
        call fortbo_pes_conditional_entropy(-60.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, &
            0.0_dp, entropy, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a conditioning event with no mass is refused", failures)

        star = 0.0_dp
        star_sd = 1.0_dp
        covariance = 0.0_dp
        call fortbo_predictive_entropy_search([0.0_dp, 0.0_dp], [1.0_dp], star, &
            star_sd, covariance, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched moment arrays are refused", failures)

        call fortbo_predictive_entropy_search([0.0_dp, 0.0_dp], [1.0_dp, -1.0_dp], &
            star, star_sd, covariance, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative standard deviation is refused", failures)
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

end program test_pes
