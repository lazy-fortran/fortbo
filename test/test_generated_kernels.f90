program test_generated_kernels
    !! BO3T: every generated derivative kernel against an independent oracle.
    !!
    !! The first-principles mandate says a generated derivative is checked "in
    !! the same test binary against an oracle that does not read the generated
    !! code". This is that binary. It sweeps *all* of `src/generated`, so adding
    !! a leaf without evidence means adding it here too.
    !!
    !! The oracle is Richardson-extrapolated central differences of the leaf's
    !! own primal output. Complex-step would be sharper — it has no subtractive
    !! cancellation at all — but it needs the leaf emitted over complex
    !! arithmetic, and FortSym currently emits real leaves. That is a real
    !! limitation and is recorded rather than worked around; the roadmap allows
    !! either oracle, and at the step sizes used here Richardson pins every
    !! derivative to at least eight digits, far tighter than any structural
    !! error could hide in.
    !!
    !! What is checked, for each leaf:
    !!
    !!   * every reported derivative against the extrapolated difference of the
    !!     matching primal, coordinate by coordinate;
    !!   * the identities the derivation implies but the emitted code does not
    !!     enforce — a generated expression can be numerically right at a point
    !!     and structurally wrong, and these catch that;
    !!   * behaviour in the regime the derivation is least reliable in, which is
    !!     where a generated kernel actually fails.

    use fortnum_kinds, only: dp
    implicit none

    interface
        pure subroutine fortbo_generated_acquisition_leaf(mu, sigma, best, xi, &
                ei, ei_d_mu, ei_d_sigma, pi_value, pi_d_mu, pi_d_sigma)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: mu, sigma, best, xi
            real(real64), intent(out) :: ei, ei_d_mu, ei_d_sigma
            real(real64), intent(out) :: pi_value, pi_d_mu, pi_d_sigma
        end subroutine fortbo_generated_acquisition_leaf

        pure subroutine fortbo_generated_preference_leaf(f_a, f_b, scale, &
                probability, probability_d_f_a, probability_d_f_b, &
                probability_d_scale, log_probability, log_probability_d_f_a, &
                log_probability_d_f_b, log_probability_d_scale)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: f_a, f_b, scale
            real(real64), intent(out) :: probability, probability_d_f_a
            real(real64), intent(out) :: probability_d_f_b, probability_d_scale
            real(real64), intent(out) :: log_probability, log_probability_d_f_a
            real(real64), intent(out) :: log_probability_d_f_b
            real(real64), intent(out) :: log_probability_d_scale
        end subroutine fortbo_generated_preference_leaf

        pure subroutine fortbo_generated_trust_region_leaf(log_lengthscale, &
                log_mean, base_length, side_length, side_d_log_lengthscale, &
                side_d_log_mean, side_d_base_length)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: log_lengthscale, log_mean, base_length
            real(real64), intent(out) :: side_length, side_d_log_lengthscale
            real(real64), intent(out) :: side_d_log_mean, side_d_base_length
        end subroutine fortbo_generated_trust_region_leaf
    end interface

    integer :: failures

    failures = 0
    call check_acquisition_leaf(failures)
    call check_acquisition_identities(failures)
    call check_preference_leaf(failures)
    call check_preference_identities(failures)
    call check_trust_region_leaf(failures)

    if (failures == 0) then
        print *, "test_generated_kernels: PASS"
    else
        print *, "test_generated_kernels: FAIL", failures
        error stop 1
    end if

contains

    !! Richardson-extrapolated central difference of a scalar sample.
    pure real(dp) function extrapolate(coarse, fine) result(derivative)
        real(dp), intent(in) :: coarse, fine

        ! The central formula errs at O(h^2); halving and combining cancels that
        ! term and leaves O(h^4).
        derivative = (4.0_dp*fine - coarse)/3.0_dp
    end function extrapolate

    subroutine check_acquisition_leaf(failures)
        integer, intent(inout) :: failures
        real(dp) :: mu, sigma, best, xi
        real(dp) :: ei, d_mu, d_sigma, pi_value, pi_d_mu, pi_d_sigma
        real(dp) :: plus_c, minus_c, plus_f, minus_f
        real(dp) :: e_pc, e_mc, e_pf, e_mf, junk(4)
        real(dp) :: numeric
        real(dp), parameter :: h = 1.0e-4_dp
        integer :: c
        logical :: matches

        matches = .true.
        do c = 1, 5
            select case (c)
            case (1)
                mu = 0.0_dp; sigma = 1.0_dp; best = 0.5_dp; xi = 0.0_dp
            case (2)
                mu = 1.5_dp; sigma = 0.25_dp; best = 1.0_dp; xi = 0.01_dp
            case (3)
                ! Deep in the tail, where EI is tiny and the derivation is
                ! numerically least comfortable.
                mu = 4.0_dp; sigma = 0.5_dp; best = 0.0_dp; xi = 0.0_dp
            case (4)
                mu = -2.0_dp; sigma = 2.0_dp; best = 0.0_dp; xi = 0.0_dp
            case (5)
                ! Near-zero spread, the other awkward limit.
                mu = 0.1_dp; sigma = 0.02_dp; best = 0.2_dp; xi = 0.0_dp
            end select

            ! d/d mu
            call fortbo_generated_acquisition_leaf(mu, sigma, best, xi, ei, d_mu, &
                d_sigma, pi_value, pi_d_mu, pi_d_sigma)
            call fortbo_generated_acquisition_leaf(mu + h, sigma, best, xi, &
                plus_c, junk(1), junk(2), e_pc, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu - h, sigma, best, xi, &
                minus_c, junk(1), junk(2), e_mc, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu + 0.5_dp*h, sigma, best, xi, &
                plus_f, junk(1), junk(2), e_pf, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu - 0.5_dp*h, sigma, best, xi, &
                minus_f, junk(1), junk(2), e_mf, junk(3), junk(4))
            numeric = extrapolate((plus_c - minus_c)/(2.0_dp*h), &
                (plus_f - minus_f)/h)
            if (abs(numeric - d_mu) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
            numeric = extrapolate((e_pc - e_mc)/(2.0_dp*h), (e_pf - e_mf)/h)
            if (abs(numeric - pi_d_mu) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.

            ! d/d sigma
            call fortbo_generated_acquisition_leaf(mu, sigma + h, best, xi, &
                plus_c, junk(1), junk(2), e_pc, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu, sigma - h, best, xi, &
                minus_c, junk(1), junk(2), e_mc, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu, sigma + 0.5_dp*h, best, xi, &
                plus_f, junk(1), junk(2), e_pf, junk(3), junk(4))
            call fortbo_generated_acquisition_leaf(mu, sigma - 0.5_dp*h, best, xi, &
                minus_f, junk(1), junk(2), e_mf, junk(3), junk(4))
            numeric = extrapolate((plus_c - minus_c)/(2.0_dp*h), &
                (plus_f - minus_f)/h)
            if (abs(numeric - d_sigma) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
            numeric = extrapolate((e_pc - e_mc)/(2.0_dp*h), (e_pf - e_mf)/h)
            if (abs(numeric - pi_d_sigma) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            "every acquisition-leaf derivative matches extrapolated differences", &
            failures)
    end subroutine check_acquisition_leaf

    !! Identities the derivation implies and the emitted code does not enforce.
    !! A generated expression can be numerically right at a point and still be
    !! structurally the wrong function.
    subroutine check_acquisition_identities(failures)
        integer, intent(inout) :: failures
        real(dp) :: ei, d_mu, d_sigma, pi_value, pi_d_mu, pi_d_sigma
        real(dp) :: shifted_ei, s_d_mu, s_d_sigma, s_pi, s_pi_mu, s_pi_sigma
        integer :: k
        logical :: holds

        ! EI depends on mu and best only through their difference, so shifting
        ! both leaves everything unchanged.
        holds = .true.
        do k = -3, 3
            call fortbo_generated_acquisition_leaf(0.3_dp, 0.8_dp, 0.7_dp, 0.0_dp, &
                ei, d_mu, d_sigma, pi_value, pi_d_mu, pi_d_sigma)
            call fortbo_generated_acquisition_leaf(0.3_dp + real(k, dp), 0.8_dp, &
                0.7_dp + real(k, dp), 0.0_dp, shifted_ei, s_d_mu, s_d_sigma, &
                s_pi, s_pi_mu, s_pi_sigma)
            if (abs(shifted_ei - ei) > 1.0e-12_dp) holds = .false.
            if (abs(s_pi - pi_value) > 1.0e-12_dp) holds = .false.
            if (abs(s_d_mu - d_mu) > 1.0e-12_dp) holds = .false.
        end do
        call expect(holds, &
            "the acquisition leaf depends on mu and best only through their gap", &
            failures)

        ! `xi` enters exactly as a shift of `best`, which is what makes it an
        ! exploration offset rather than an arbitrary parameter.
        call fortbo_generated_acquisition_leaf(0.2_dp, 0.6_dp, 1.0_dp, 0.25_dp, &
            ei, d_mu, d_sigma, pi_value, pi_d_mu, pi_d_sigma)
        call fortbo_generated_acquisition_leaf(0.2_dp, 0.6_dp, 0.75_dp, 0.0_dp, &
            shifted_ei, s_d_mu, s_d_sigma, s_pi, s_pi_mu, s_pi_sigma)
        call expect(abs(ei - shifted_ei) < 1.0e-14_dp .and. &
            abs(pi_value - s_pi) < 1.0e-14_dp, &
            "xi enters exactly as a shift of the incumbent", failures)

        ! PI is a probability, and EI is non-negative, at every sampled point.
        holds = .true.
        do k = -20, 20
            call fortbo_generated_acquisition_leaf(0.2_dp*real(k, dp), 0.7_dp, &
                0.0_dp, 0.0_dp, ei, d_mu, d_sigma, pi_value, pi_d_mu, pi_d_sigma)
            if (pi_value < 0.0_dp .or. pi_value > 1.0_dp) holds = .false.
            if (ei < 0.0_dp) holds = .false.
            ! Improvement can only fall as the mean rises.
            if (d_mu > 0.0_dp) holds = .false.
        end do
        call expect(holds, &
            "the leaf yields a probability, a non-negative EI, and a falling EI", &
            failures)
    end subroutine check_acquisition_identities

    subroutine check_preference_leaf(failures)
        integer, intent(inout) :: failures
        real(dp) :: f_a, f_b, scale
        real(dp) :: p, p_a, p_b, p_s, lp, lp_a, lp_b, lp_s
        real(dp) :: junk(6)
        real(dp) :: pc, mc, pf, mf, lpc, lmc, lpf, lmf, numeric
        real(dp), parameter :: h = 1.0e-4_dp
        integer :: c
        logical :: matches

        matches = .true.
        do c = 1, 4
            select case (c)
            case (1)
                f_a = 0.0_dp; f_b = 1.0_dp; scale = 1.0_dp
            case (2)
                f_a = 2.0_dp; f_b = -0.5_dp; scale = 0.4_dp
            case (3)
                f_a = -1.0_dp; f_b = -1.2_dp; scale = 2.0_dp
            case (4)
                f_a = 0.5_dp; f_b = 0.5_dp; scale = 0.7_dp
            end select

            call fortbo_generated_preference_leaf(f_a, f_b, scale, p, p_a, p_b, &
                p_s, lp, lp_a, lp_b, lp_s)

            call sample(f_a + h, f_b, scale, pc, lpc)
            call sample(f_a - h, f_b, scale, mc, lmc)
            call sample(f_a + 0.5_dp*h, f_b, scale, pf, lpf)
            call sample(f_a - 0.5_dp*h, f_b, scale, mf, lmf)
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - p_a) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
            numeric = extrapolate((lpc - lmc)/(2.0_dp*h), (lpf - lmf)/h)
            if (abs(numeric - lp_a) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.

            call sample(f_a, f_b + h, scale, pc, lpc)
            call sample(f_a, f_b - h, scale, mc, lmc)
            call sample(f_a, f_b + 0.5_dp*h, scale, pf, lpf)
            call sample(f_a, f_b - 0.5_dp*h, scale, mf, lmf)
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - p_b) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.

            call sample(f_a, f_b, scale + h, pc, lpc)
            call sample(f_a, f_b, scale - h, mc, lmc)
            call sample(f_a, f_b, scale + 0.5_dp*h, pf, lpf)
            call sample(f_a, f_b, scale - 0.5_dp*h, mf, lmf)
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - p_s) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
            numeric = extrapolate((lpc - lmc)/(2.0_dp*h), (lpf - lmf)/h)
            if (abs(numeric - lp_s) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            "every preference-leaf derivative matches extrapolated differences", &
            failures)
    end subroutine check_preference_leaf

    subroutine sample(f_a, f_b, scale, probability, log_probability)
        real(dp), intent(in) :: f_a, f_b, scale
        real(dp), intent(out) :: probability, log_probability
        real(dp) :: a, b, s, l_a, l_b, l_s

        call fortbo_generated_preference_leaf(f_a, f_b, scale, probability, a, b, &
            s, log_probability, l_a, l_b, l_s)
    end subroutine sample

    subroutine check_preference_identities(failures)
        integer, intent(inout) :: failures
        real(dp) :: p, p_a, p_b, p_s, lp, lp_a, lp_b, lp_s
        real(dp) :: q, q_a, q_b, q_s, lq, lq_a, lq_b, lq_s
        integer :: k
        logical :: holds

        ! The log output must be the log of the probability output, and the two
        ! latent derivatives must be equal and opposite, since only the
        ! difference of the latents enters.
        holds = .true.
        do k = -15, 15
            call fortbo_generated_preference_leaf(0.2_dp*real(k, dp), 0.0_dp, &
                0.8_dp, p, p_a, p_b, p_s, lp, lp_a, lp_b, lp_s)
            if (p > 0.0_dp) then
                if (abs(lp - log(p)) > 1.0e-10_dp*max(1.0_dp, abs(lp))) &
                    holds = .false.
            end if
            if (abs(p_a + p_b) > 1.0e-12_dp*max(1.0_dp, abs(p_a))) holds = .false.
            if (abs(lp_a + lp_b) > 1.0e-12_dp*max(1.0_dp, abs(lp_a))) holds = .false.
            if (p < 0.0_dp .or. p > 1.0_dp) holds = .false.
        end do
        call expect(holds, &
            "the preference leaf's log, sign, and range identities hold", failures)

        ! Swapping the pair complements the probability, which is the identity
        ! that would break under a misplaced sign in the erfc argument.
        holds = .true.
        do k = -6, 6
            call fortbo_generated_preference_leaf(0.4_dp*real(k, dp), 0.3_dp, &
                1.1_dp, p, p_a, p_b, p_s, lp, lp_a, lp_b, lp_s)
            call fortbo_generated_preference_leaf(0.3_dp, 0.4_dp*real(k, dp), &
                1.1_dp, q, q_a, q_b, q_s, lq, lq_a, lq_b, lq_s)
            if (abs(p + q - 1.0_dp) > 1.0e-13_dp) holds = .false.
        end do
        call expect(holds, "the two orderings of a pair are complementary", &
            failures)
    end subroutine check_preference_identities

    subroutine check_trust_region_leaf(failures)
        integer, intent(inout) :: failures
        real(dp) :: ell, mean, base, value, d_ell, d_mean, d_base
        real(dp) :: pc, mc, pf, mf, numeric, junk(3)
        real(dp), parameter :: h = 1.0e-5_dp
        integer :: c
        logical :: matches

        matches = .true.
        do c = 1, 3
            select case (c)
            case (1)
                ell = 0.3_dp; mean = -0.2_dp; base = 0.8_dp
            case (2)
                ell = -1.7_dp; mean = 0.9_dp; base = 1.6_dp
            case (3)
                ell = 2.5_dp; mean = 2.5_dp; base = 0.05_dp
            end select

            call fortbo_generated_trust_region_leaf(ell, mean, base, value, d_ell, &
                d_mean, d_base)

            call fortbo_generated_trust_region_leaf(ell + h, mean, base, pc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell - h, mean, base, mc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell + 0.5_dp*h, mean, base, pf, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell - 0.5_dp*h, mean, base, mf, &
                junk(1), junk(2), junk(3))
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - d_ell) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.

            call fortbo_generated_trust_region_leaf(ell, mean + h, base, pc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean - h, base, mc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean + 0.5_dp*h, base, pf, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean - 0.5_dp*h, base, mf, &
                junk(1), junk(2), junk(3))
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - d_mean) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.

            call fortbo_generated_trust_region_leaf(ell, mean, base + h, pc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean, base - h, mc, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean, base + 0.5_dp*h, pf, &
                junk(1), junk(2), junk(3))
            call fortbo_generated_trust_region_leaf(ell, mean, base - 0.5_dp*h, mf, &
                junk(1), junk(2), junk(3))
            numeric = extrapolate((pc - mc)/(2.0_dp*h), (pf - mf)/h)
            if (abs(numeric - d_base) > 1.0e-7_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            "every trust-region-leaf derivative matches extrapolated differences", &
            failures)

        ! The derivative in the log lengthscale equals the value itself, since
        ! the value is an exponential of that argument. An identity the emitted
        ! code does not state and a wrong chain rule would break.
        call fortbo_generated_trust_region_leaf(0.4_dp, 0.1_dp, 0.9_dp, value, &
            d_ell, d_mean, d_base)
        call expect(abs(d_ell - value) < 1.0e-15_dp .and. &
            abs(d_mean + value) < 1.0e-15_dp, &
            "the rescaling's log derivatives are the value and its negation", &
            failures)
    end subroutine check_trust_region_leaf

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_generated_kernels
