program test_preference
    !! BO4: preference learning and noisy dominance.
    !!
    !! Oracles are independent of the implementation:
    !!   * the preference and dominance probabilities are checked against Monte
    !!     Carlo simulation of the very process they model — draw the noisy
    !!     judgements, count how often `a` wins — with tolerances taken from the
    !!     binomial standard error rather than tuned until the test passed;
    !!   * the log-likelihood gradient is checked against central differences of
    !!     the log-likelihood, so the score cannot silently disagree with the
    !!     objective a fitter is climbing;
    !!   * the asymptotic tail branch is checked for agreement with the
    !!     generated expression on the shared side of the switch, and against
    !!     the exact Gaussian tail further out;
    !!   * symmetry and normalization are checked as identities the model must
    !!     satisfy no matter how it is computed.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortbo_preference, only: fortbo_preference_probability, &
        fortbo_preference_log_probability, fortbo_preference_log_likelihood, &
        fortbo_probability_dominates, fortbo_probability_non_dominated
    implicit none

    integer :: failures

    failures = 0
    call check_preference_identities(failures)
    call check_preference_against_monte_carlo(failures)
    call check_log_probability_consistency(failures)
    call check_asymptotic_tail(failures)
    call check_log_likelihood_gradient(failures)
    call check_likelihood_prefers_the_truth(failures)
    call check_dominance_against_monte_carlo(failures)
    call check_dominance_limits(failures)
    call check_non_dominated_against_monte_carlo(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_preference: PASS"
    else
        print *, "test_preference: FAIL", failures
        error stop 1
    end if

contains

    !! Identities the model satisfies by construction, and which therefore
    !! detect a sign error or a misplaced sqrt(2) immediately.
    subroutine check_preference_identities(failures)
        integer, intent(inout) :: failures
        real(dp) :: p, q
        integer :: k

        call expect(abs(fortbo_preference_probability(1.0_dp, 1.0_dp, 0.7_dp) &
            - 0.5_dp) < 1.0e-14_dp, "equal latent values give an even chance", &
            failures)

        ! FortBO minimizes: the lower value is preferred.
        call expect(fortbo_preference_probability(0.0_dp, 1.0_dp, 0.5_dp) > 0.5_dp, &
            "the lower latent value is the preferred one", failures)

        do k = -3, 3
            p = fortbo_preference_probability(real(k, dp), 0.0_dp, 0.9_dp)
            q = fortbo_preference_probability(0.0_dp, real(k, dp), 0.9_dp)
            call expect(abs(p + q - 1.0_dp) < 1.0e-14_dp, &
                "the two orderings of a pair are complementary", failures)
        end do

        ! Larger noise pulls every judgement toward a coin flip.
        call expect(fortbo_preference_probability(0.0_dp, 1.0_dp, 10.0_dp) < &
            fortbo_preference_probability(0.0_dp, 1.0_dp, 0.1_dp), &
            "more judgement noise makes the preference less certain", failures)

        ! The noiseless limit is a step, not a NaN.
        call expect(fortbo_preference_probability(0.0_dp, 1.0_dp, 0.0_dp) &
            == 1.0_dp, "a noiseless judge always ranks correctly", failures)
        call expect(fortbo_preference_probability(1.0_dp, 1.0_dp, 0.0_dp) &
            == 0.5_dp, "a noiseless tie is an even chance", failures)
    end subroutine check_preference_identities

    !! Simulate the model's own generative story and count. This is the
    !! definition of the quantity, computed a completely different way.
    subroutine check_preference_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        real(dp) :: f_a, f_b, scale, noise_a, noise_b, predicted, estimate
        real(dp) :: standard_error
        integer :: k, wins, case_index

        call rng_seed(generator, int(20260808, int64), status)

        do case_index = 1, 3
            select case (case_index)
            case (1)
                f_a = 0.0_dp; f_b = 1.0_dp; scale = 1.0_dp
            case (2)
                f_a = 2.0_dp; f_b = -0.5_dp; scale = 2.0_dp
            case (3)
                f_a = -1.0_dp; f_b = -1.25_dp; scale = 0.4_dp
            end select

            wins = 0
            do k = 1, n_samples
                call rng_normal(generator, noise_a)
                call rng_normal(generator, noise_b)
                if (f_a + scale*noise_a < f_b + scale*noise_b) wins = wins + 1
            end do
            estimate = real(wins, dp)/real(n_samples, dp)
            predicted = fortbo_preference_probability(f_a, f_b, scale)
            standard_error = sqrt(predicted*(1.0_dp - predicted) &
                /real(n_samples, dp))
            call expect(abs(estimate - predicted) < 5.0_dp*standard_error, &
                "the preference probability matches simulated judgements", &
                failures)
        end do
    end subroutine check_preference_against_monte_carlo

    !! In the regime where both are computable, the log must be the log.
    subroutine check_log_probability_consistency(failures)
        integer, intent(inout) :: failures
        real(dp) :: log_p, d_a, d_b, p
        integer :: k

        do k = -40, 40
            p = fortbo_preference_probability(real(k, dp)*0.1_dp, 0.0_dp, 0.8_dp)
            call fortbo_preference_log_probability(real(k, dp)*0.1_dp, 0.0_dp, &
                0.8_dp, log_p, d_a, d_b)
            call expect(abs(log_p - log(p)) < 1.0e-12_dp*max(1.0_dp, abs(log_p)), &
                "the log probability is the log of the probability", failures)
            call expect(abs(d_a + d_b) < 1.0e-12_dp*max(1.0_dp, abs(d_a)), &
                "the two latent derivatives are equal and opposite", failures)
        end do
    end subroutine check_log_probability_consistency

    !! The asymptotic branch must join the generated one without a step, and
    !! must stay accurate where the generated one has underflowed away.
    subroutine check_asymptotic_tail(failures)
        integer, intent(inout) :: failures
        real(dp) :: scale, f_a, log_p, d_a, d_b, reference, z
        real(dp) :: below, above, numeric, plus, minus
        real(dp), parameter :: h = 1.0e-5_dp

        scale = 1.0_dp

        ! The branch switches at erfc argument 20, i.e. f_a - f_b = 40 scale.
        ! Both sides describe the same function, so the values must agree far
        ! more closely than they differ from neighbouring points.
        ! The offsets must be small enough that the true function barely moves:
        ! its slope here is about -2x/(2 scale) = -20, so 1e-7 either side
        ! changes the value by about 2e-6 and anything larger would be a seam.
        call fortbo_preference_log_probability(40.0_dp*scale - 1.0e-7_dp, 0.0_dp, &
            scale, below, d_a, d_b)
        call fortbo_preference_log_probability(40.0_dp*scale + 1.0e-7_dp, 0.0_dp, &
            scale, above, d_a, d_b)
        call expect(abs(below - above) < 1.0e-4_dp, &
            "the asymptotic branch joins the generated one continuously", &
            failures)

        ! Far out, compare against the leading Gaussian tail written here
        ! independently. The preference probability is Phi(z) with
        ! z = (f_b - f_a)/(sqrt(2) scale), and log Phi(-|z|) is asymptotically
        ! -z^2/2 - log(|z| sqrt(2 pi)).
        f_a = 80.0_dp*scale
        call fortbo_preference_log_probability(f_a, 0.0_dp, scale, log_p, d_a, d_b)
        z = f_a/(sqrt(2.0_dp)*scale)
        reference = -0.5_dp*z*z - log(z*sqrt(8.0_dp*atan(1.0_dp)))
        ! Absolute, not relative: a relative tolerance against a value near
        ! -3200 would tolerate an additive constant of several units, which is
        ! exactly the kind of error a stray log(2) is.
        call expect(abs(log_p - reference) < 1.0e-3_dp, &
            "the deep tail matches the Gaussian asymptotic", failures)
        call expect(log_p < -1000.0_dp, &
            "the deep tail is finite where the direct expression underflows", &
            failures)

        ! The tail derivative must still be the derivative.
        call fortbo_preference_log_probability(f_a + h, 0.0_dp, scale, plus, &
            below, above)
        call fortbo_preference_log_probability(f_a - h, 0.0_dp, scale, minus, &
            below, above)
        numeric = (plus - minus)/(2.0_dp*h)
        call fortbo_preference_log_probability(f_a, 0.0_dp, scale, log_p, d_a, d_b)
        call expect(abs(numeric - d_a) < 1.0e-4_dp*abs(d_a), &
            "the tail derivative matches central differences", failures)
    end subroutine check_asymptotic_tail

    !! The gradient a fitter climbs must be the gradient of the objective it
    !! reports. Central differences of the log-likelihood are the oracle.
    subroutine check_log_likelihood_gradient(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n = 5
        real(dp) :: latent(n), gradient(n), shifted(n), scratch(n)
        real(dp) :: value, plus, minus, numeric
        real(dp), parameter :: h = 1.0e-6_dp
        integer :: winners(7), losers(7), j
        logical :: matches

        latent = [0.4_dp, -1.1_dp, 0.0_dp, 2.3_dp, -0.7_dp]
        winners = [2, 5, 1, 3, 2, 5, 1]
        losers = [1, 1, 4, 4, 3, 3, 5]

        call fortbo_preference_log_likelihood(latent, winners, losers, 0.6_dp, &
            value, gradient, status)
        call expect(status%code == FORTNUM_OK, "the log-likelihood is computed", &
            failures)
        call expect(value < 0.0_dp, "a log-likelihood of probabilities is negative", &
            failures)

        matches = .true.
        do j = 1, n
            shifted = latent
            shifted(j) = latent(j) + h
            call fortbo_preference_log_likelihood(shifted, winners, losers, &
                0.6_dp, plus, scratch, status)
            shifted(j) = latent(j) - h
            call fortbo_preference_log_likelihood(shifted, winners, losers, &
                0.6_dp, minus, scratch, status)
            numeric = (plus - minus)/(2.0_dp*h)
            if (abs(numeric - gradient(j)) > 1.0e-6_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            "the log-likelihood gradient matches central differences", failures)

        ! A latent vector shifted by a constant explains the same comparisons:
        ! only differences are observed. The gradient must therefore sum to zero.
        call expect(abs(sum(gradient)) < 1.0e-12_dp, &
            "the gradient is orthogonal to a constant shift", failures)

        call fortbo_preference_log_likelihood(latent + 100.0_dp, winners, losers, &
            0.6_dp, plus, scratch, status)
        call expect(abs(plus - value) < 1.0e-12_dp*abs(value), &
            "a constant shift leaves the likelihood unchanged", failures)
    end subroutine check_log_likelihood_gradient

    !! The behavioral point of the whole model: a latent ordering consistent
    !! with the judgements must score better than one that contradicts them.
    subroutine check_likelihood_prefers_the_truth(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: consistent(4), reversed(4), flat(4)
        real(dp) :: gradient(4), good_value, bad_value, flat_value
        integer :: winners(6), losers(6)

        ! Every comparison agrees with the ordering 1 < 2 < 3 < 4.
        winners = [1, 1, 1, 2, 2, 3]
        losers = [2, 3, 4, 3, 4, 4]

        consistent = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        reversed = [3.0_dp, 2.0_dp, 1.0_dp, 0.0_dp]
        flat = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp]

        call fortbo_preference_log_likelihood(consistent, winners, losers, &
            0.5_dp, good_value, gradient, status)
        call fortbo_preference_log_likelihood(reversed, winners, losers, &
            0.5_dp, bad_value, gradient, status)
        call fortbo_preference_log_likelihood(flat, winners, losers, &
            0.5_dp, flat_value, gradient, status)

        call expect(good_value > bad_value, &
            "the consistent ordering beats the reversed one", failures)
        call expect(good_value > flat_value, &
            "the consistent ordering beats an uninformative flat one", failures)

        ! With no information, every comparison is a coin flip: 6 log(1/2).
        call expect(abs(flat_value - 6.0_dp*log(0.5_dp)) < 1.0e-12_dp, &
            "a flat latent gives exactly even odds on every comparison", failures)
    end subroutine check_likelihood_prefers_the_truth

    !! Dominance under uncertainty, checked by sampling the posteriors.
    subroutine check_dominance_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        integer, parameter :: n_objectives = 3
        real(dp) :: mean_a(n_objectives), sd_a(n_objectives)
        real(dp) :: mean_b(n_objectives), sd_b(n_objectives)
        real(dp) :: draw_a(n_objectives), draw_b(n_objectives), noise
        real(dp) :: predicted, estimate, standard_error
        integer :: k, m, hits

        call rng_seed(generator, int(90210, int64), status)

        mean_a = [0.0_dp, 0.5_dp, -1.0_dp]
        sd_a = [0.4_dp, 1.0_dp, 0.2_dp]
        mean_b = [0.6_dp, 1.0_dp, -0.9_dp]
        sd_b = [0.3_dp, 0.5_dp, 0.7_dp]

        hits = 0
        do k = 1, n_samples
            do m = 1, n_objectives
                call rng_normal(generator, noise)
                draw_a(m) = mean_a(m) + sd_a(m)*noise
                call rng_normal(generator, noise)
                draw_b(m) = mean_b(m) + sd_b(m)*noise
            end do
            if (all(draw_a < draw_b)) hits = hits + 1
        end do
        estimate = real(hits, dp)/real(n_samples, dp)

        call fortbo_probability_dominates(mean_a, sd_a, mean_b, sd_b, predicted, &
            status)
        call expect(status%code == FORTNUM_OK, "the dominance probability computes", &
            failures)
        standard_error = sqrt(predicted*(1.0_dp - predicted)/real(n_samples, dp))
        call expect(abs(estimate - predicted) < 5.0_dp*standard_error, &
            "the dominance probability matches sampled posteriors", failures)
    end subroutine check_dominance_against_monte_carlo

    !! The noiseless limit must reproduce ordinary Pareto dominance exactly.
    !! This is the property that makes the noisy version a generalization
    !! rather than a different thing wearing the same name.
    subroutine check_dominance_limits(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: a(2), b(2), zero(2), probability, wide(2)

        zero = 0.0_dp

        a = [0.0_dp, 0.0_dp]
        b = [1.0_dp, 1.0_dp]
        call fortbo_probability_dominates(a, zero, b, zero, probability, status)
        call expect(probability == 1.0_dp, &
            "certain dominance has probability one", failures)

        call fortbo_probability_dominates(b, zero, a, zero, probability, status)
        call expect(probability == 0.0_dp, &
            "certain non-dominance has probability zero", failures)

        ! Better in one objective, worse in the other: no dominance either way.
        b = [-1.0_dp, 1.0_dp]
        call fortbo_probability_dominates(a, zero, b, zero, probability, status)
        call expect(probability == 0.0_dp, &
            "an incomparable pair never dominates", failures)

        ! Identical beliefs: by symmetry each dominates the other with the same
        ! probability, and with two objectives that is a quarter.
        wide = [1.0_dp, 1.0_dp]
        a = [0.0_dp, 0.0_dp]
        call fortbo_probability_dominates(a, wide, a, wide, probability, status)
        call expect(abs(probability - 0.25_dp) < 1.0e-12_dp, &
            "identical beliefs dominate each other a quarter of the time", &
            failures)
    end subroutine check_dominance_limits

    subroutine check_non_dominated_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 300000
        integer, parameter :: n_objectives = 2, n_members = 3
        real(dp) :: mean(n_objectives), sd(n_objectives)
        real(dp) :: front_mean(n_objectives, n_members)
        real(dp) :: front_sd(n_objectives, n_members)
        real(dp) :: candidate(n_objectives), member(n_objectives), noise
        real(dp) :: predicted, estimate, standard_error
        integer :: k, m, j, survivals
        logical :: dominated

        call rng_seed(generator, int(31415, int64), status)

        mean = [0.0_dp, 0.0_dp]
        sd = [0.5_dp, 0.5_dp]
        front_mean = reshape([-0.2_dp, 0.4_dp, &
            0.3_dp, -0.3_dp, &
            -0.1_dp, -0.1_dp], [n_objectives, n_members])
        front_sd = reshape([0.3_dp, 0.3_dp, &
            0.4_dp, 0.2_dp, &
            0.6_dp, 0.6_dp], [n_objectives, n_members])

        survivals = 0
        do k = 1, n_samples
            do m = 1, n_objectives
                call rng_normal(generator, noise)
                candidate(m) = mean(m) + sd(m)*noise
            end do
            dominated = .false.
            do j = 1, n_members
                do m = 1, n_objectives
                    call rng_normal(generator, noise)
                    member(m) = front_mean(m, j) + front_sd(m, j)*noise
                end do
                if (all(member < candidate)) dominated = .true.
            end do
            if (.not. dominated) survivals = survivals + 1
        end do
        estimate = real(survivals, dp)/real(n_samples, dp)

        call fortbo_probability_non_dominated(mean, sd, front_mean, front_sd, &
            predicted, status)
        call expect(status%code == FORTNUM_OK, &
            "the non-dominance probability computes", failures)
        standard_error = sqrt(predicted*(1.0_dp - predicted)/real(n_samples, dp))
        ! The candidate is drawn once and compared against every member, so the
        ! "member j dominates the candidate" events all move together with it
        ! and are positively associated. For such events P(none occurs) is at
        ! least the product of the individual non-occurrence probabilities, so
        ! the module's independent product must come out *below* the truth.
        ! That direction is the guarantee the documentation claims, and it is
        ! what is asserted here rather than an equality that does not hold.
        call expect(predicted <= estimate + 4.0_dp*standard_error, &
            "the independent product understates non-dominance, as documented", &
            failures)
        call expect(predicted > 0.75_dp*estimate, &
            "the approximation stays within a useful factor of the truth", &
            failures)

        ! With a single member there is nothing to be dependent with, so the
        ! product formula is exact and must match the simulation outright.
        call check_single_member_front_is_exact(failures)

        ! An empty front dominates nothing.
        call fortbo_probability_non_dominated(mean, sd, &
            front_mean(:, 1:0), front_sd(:, 1:0), predicted, status)
        call expect(predicted == 1.0_dp, &
            "an empty front leaves every candidate non-dominated", failures)
    end subroutine check_non_dominated_against_monte_carlo

    !! A one-member front makes the independence assumption vacuous, so here the
    !! product formula is exact and sampling error is the only slack allowed.
    subroutine check_single_member_front_is_exact(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 400000
        real(dp) :: mean(2), sd(2), front_mean(2, 1), front_sd(2, 1)
        real(dp) :: candidate(2), member(2), noise
        real(dp) :: predicted, estimate, standard_error
        integer :: k, m, survivals

        call rng_seed(generator, int(2718, int64), status)

        mean = [0.0_dp, 0.0_dp]
        sd = [0.5_dp, 0.5_dp]
        front_mean = reshape([-0.3_dp, -0.2_dp], [2, 1])
        front_sd = reshape([0.4_dp, 0.6_dp], [2, 1])

        survivals = 0
        do k = 1, n_samples
            do m = 1, 2
                call rng_normal(generator, noise)
                candidate(m) = mean(m) + sd(m)*noise
                call rng_normal(generator, noise)
                member(m) = front_mean(m, 1) + front_sd(m, 1)*noise
            end do
            if (.not. all(member < candidate)) survivals = survivals + 1
        end do
        estimate = real(survivals, dp)/real(n_samples, dp)

        call fortbo_probability_non_dominated(mean, sd, front_mean, front_sd, &
            predicted, status)
        standard_error = sqrt(predicted*(1.0_dp - predicted)/real(n_samples, dp))
        call expect(abs(estimate - predicted) < 5.0_dp*standard_error, &
            "a one-member front is handled exactly", failures)
    end subroutine check_single_member_front_is_exact

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: latent(3), gradient(3), short_gradient(2), value
        real(dp) :: a(2), b(2), sd(2), bad_sd(2), probability
        integer :: winners(2), losers(2)

        latent = [0.0_dp, 1.0_dp, 2.0_dp]
        winners = [1, 2]
        losers = [2, 3]

        call fortbo_preference_log_likelihood(latent, winners, losers, &
            0.5_dp, value, short_gradient, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-sized gradient is refused", failures)

        call fortbo_preference_log_likelihood(latent, winners, [1, 1], &
            0.5_dp, value, gradient, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "comparing a point with itself is refused", failures)

        call fortbo_preference_log_likelihood(latent, winners, [2, 9], &
            0.5_dp, value, gradient, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an out-of-range comparison index is refused", failures)

        call fortbo_preference_log_likelihood(latent, winners, losers, &
            0.0_dp, value, gradient, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero noise scale is refused when fitting", failures)

        a = [0.0_dp, 0.0_dp]
        b = [1.0_dp, 1.0_dp]
        sd = [0.1_dp, 0.1_dp]
        bad_sd = [0.1_dp, -0.1_dp]
        call fortbo_probability_dominates(a, sd, b, bad_sd, probability, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative standard deviation is refused", failures)

        call fortbo_probability_dominates(a, sd, [1.0_dp], [0.1_dp], probability, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched objective counts are refused", failures)
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

end program test_preference
