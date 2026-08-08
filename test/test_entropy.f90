program test_entropy
    !! BO1: max-value entropy search.
    !!
    !! Oracles:
    !!
    !!   * the truncated-normal entropy is checked against numerical
    !!     integration of `-p log p` over the truncated density, computed here
    !!     by Simpson's rule. That shares no code with the closed form and is
    !!     the definition of entropy rather than a rearrangement of it;
    !!   * MES is checked against the same integration applied to the identity
    !!     it is built from, `H(f) - H(f | f >= y*)`, so the acquisition is
    !!     validated as a difference of two independently computed entropies;
    !!   * the behavior entropy search exists for: a point that is informative
    !!     without being good must score above one that is neither. Expected
    !!     improvement cannot distinguish those, and this is what separates the
    !!     two families.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_entropy, only: fortbo_truncated_normal_entropy, &
        fortbo_max_value_entropy_search
    implicit none

    integer :: failures

    failures = 0
    call check_truncated_entropy_against_quadrature(failures)
    call check_mes_against_entropy_difference(failures)
    call check_mes_limits(failures)
    call check_mes_prefers_informative_points(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_entropy: PASS"
    else
        print *, "test_entropy: FAIL", failures
        error stop 1
    end if

contains

    !! `-integral p log p` over a normal truncated to `[lower, infinity)`,
    !! by Simpson's rule out to where the density is negligible.
    real(dp) function quadrature_entropy(mean, sd, lower) result(entropy)
        real(dp), intent(in) :: mean, sd, lower
        integer, parameter :: n_intervals = 200000
        real(dp) :: alpha, tail, upper, h, x, density, term, total
        integer :: k

        alpha = (lower - mean)/sd
        tail = 0.5_dp*(1.0_dp - erf(alpha/sqrt(2.0_dp)))
        ! Twelve standard deviations above the truncation point leaves less
        ! than 1e-32 of the mass outside, far below the quadrature error.
        upper = max(lower, mean) + 12.0_dp*sd
        h = (upper - lower)/real(n_intervals, dp)

        total = 0.0_dp
        do k = 0, n_intervals
            x = lower + h*real(k, dp)
            density = exp(-0.5_dp*((x - mean)/sd)**2) &
                /(sd*sqrt(8.0_dp*atan(1.0_dp))*tail)
            if (density > 0.0_dp) then
                term = -density*log(density)
            else
                term = 0.0_dp
            end if
            if (k == 0 .or. k == n_intervals) then
                total = total + term
            else if (mod(k, 2) == 1) then
                total = total + 4.0_dp*term
            else
                total = total + 2.0_dp*term
            end if
        end do
        entropy = total*h/3.0_dp
    end function quadrature_entropy

    subroutine check_truncated_entropy_against_quadrature(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean, sd, lower, closed_form, numeric
        integer :: case_index
        logical :: matches

        matches = .true.
        do case_index = 1, 4
            select case (case_index)
            case (1)
                mean = 0.0_dp; sd = 1.0_dp; lower = -3.0_dp
            case (2)
                ! Truncation at the mean: exactly half the mass survives.
                mean = 1.5_dp; sd = 0.75_dp; lower = 1.5_dp
            case (3)
                ! Truncation well above the mean: a thin tail.
                mean = 0.0_dp; sd = 1.0_dp; lower = 1.5_dp
            case (4)
                mean = -2.0_dp; sd = 3.0_dp; lower = -1.0_dp
            end select

            call fortbo_truncated_normal_entropy(mean, sd, lower, closed_form, &
                status)
            if (status%code /= FORTNUM_OK) matches = .false.
            numeric = quadrature_entropy(mean, sd, lower)
            if (abs(closed_form - numeric) > 1.0e-6_dp*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            "the truncated entropy matches numerical integration of -p log p", &
            failures)

        ! With the truncation far below the mean almost nothing is removed, so
        ! the entropy must approach that of the untruncated normal.
        call fortbo_truncated_normal_entropy(0.0_dp, 1.0_dp, -40.0_dp, closed_form, &
            status)
        call expect(abs(closed_form - 0.5_dp*log(2.0_dp*exp(1.0_dp) &
            *4.0_dp*atan(1.0_dp))) < 1.0e-10_dp, &
            "a vacuous truncation recovers the normal's own entropy", failures)

        ! Truncation removes mass and so must reduce entropy.
        block
            real(dp) :: tight
            call fortbo_truncated_normal_entropy(0.0_dp, 1.0_dp, 1.0_dp, tight, &
                status)
            call expect(tight < closed_form, &
                "truncating a distribution lowers its entropy", failures)
        end block
    end subroutine check_truncated_entropy_against_quadrature

    !! MES is `H(f) - H(f | f >= y*)`. Both entropies are computed here by
    !! quadrature, so the acquisition is validated against its own definition.
    subroutine check_mes_against_entropy_difference(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(1), sd(1), samples(1), value(1)
        real(dp) :: full_entropy, conditional, expected
        integer :: case_index
        logical :: matches

        matches = .true.
        do case_index = 1, 3
            select case (case_index)
            case (1)
                mean(1) = 0.0_dp; sd(1) = 1.0_dp; samples(1) = -0.5_dp
            case (2)
                mean(1) = 2.0_dp; sd(1) = 0.5_dp; samples(1) = 1.0_dp
            case (3)
                mean(1) = -1.0_dp; sd(1) = 2.0_dp; samples(1) = -1.5_dp
            end select

            call fortbo_max_value_entropy_search(mean, sd, samples, value, status)
            if (status%code /= FORTNUM_OK) matches = .false.

            full_entropy = 0.5_dp*log(2.0_dp*exp(1.0_dp)*4.0_dp*atan(1.0_dp)) &
                + log(sd(1))
            conditional = quadrature_entropy(mean(1), sd(1), samples(1))
            expected = full_entropy - conditional
            if (abs(value(1) - expected) > 1.0e-6_dp*max(1.0_dp, abs(expected))) &
                matches = .false.
        end do
        call expect(matches, &
            "MES equals the entropy reduction it is defined as", failures)
    end subroutine check_mes_against_entropy_difference

    subroutine check_mes_limits(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(3), sd(3), samples(2), value(3)

        mean = [0.0_dp, 0.0_dp, 0.0_dp]
        samples = [-0.2_dp, -0.4_dp]

        ! A point known exactly teaches nothing whatever the optimum is.
        sd = [0.0_dp, 1.0_dp, 2.0_dp]
        call fortbo_max_value_entropy_search(mean, sd, samples, value, status)
        call expect(status%code == FORTNUM_OK, "MES computes", failures)
        call expect(value(1) == 0.0_dp, &
            "a point with no posterior uncertainty carries no information", &
            failures)
        call expect(all(value >= 0.0_dp), &
            "MES is never negative, as a mutual information", failures)

        ! At equal means, more uncertainty means more to learn.
        call expect(value(3) > value(2), &
            "a more uncertain point is more informative", failures)

        ! A candidate the sampled optimum has already ruled out far below
        ! contributes nothing usable rather than an enormous number.
        call fortbo_max_value_entropy_search([100.0_dp], [1.0_dp], [-50.0_dp], &
            value(1:1), status)
        call expect(status%code == FORTNUM_OK .and. value(1) >= 0.0_dp .and. &
            value(1) < 1.0e-6_dp, &
            "a candidate the optimum has ruled out is worth nothing", failures)
    end subroutine check_mes_limits

    !! The claim that separates entropy search from improvement. Point A is
    !! uncertain and sits where the sampled optimum might be beaten; point B is
    !! equally uncertain but its posterior mean is so far above the optimum that
    !! knowing its value settles nothing.
    subroutine check_mes_prefers_informative_points(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: mean(2), sd(2), samples(3), value(2)

        samples = [-1.0_dp, -1.2_dp, -0.8_dp]
        mean = [-0.9_dp, 6.0_dp]
        sd = [1.0_dp, 1.0_dp]

        call fortbo_max_value_entropy_search(mean, sd, samples, value, status)
        call expect(value(1) > value(2), &
            "a point that could settle where the optimum is beats one that cannot", &
            failures)

        ! Uncertainty alone is not enough: a very uncertain point far from the
        ! sampled optimum still loses to a moderately uncertain one near it.
        mean = [-0.9_dp, 6.0_dp]
        sd = [0.5_dp, 3.0_dp]
        call fortbo_max_value_entropy_search(mean, sd, samples, value, status)
        call expect(value(1) > value(2), &
            "relevance beats raw uncertainty", failures)
    end subroutine check_mes_prefers_informative_points

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: entropy, value(2)
        real(dp) :: empty(0)

        call fortbo_truncated_normal_entropy(0.0_dp, 0.0_dp, 0.0_dp, entropy, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero standard deviation is refused", failures)

        call fortbo_truncated_normal_entropy(0.0_dp, 1.0_dp, 60.0_dp, entropy, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a truncation leaving no usable mass is refused", failures)

        call fortbo_max_value_entropy_search([0.0_dp, 0.0_dp], [1.0_dp], &
            [0.0_dp], value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched moment arrays are refused", failures)

        call fortbo_max_value_entropy_search([0.0_dp, 0.0_dp], [1.0_dp, 1.0_dp], &
            empty, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an empty set of optimum samples is refused", failures)

        call fortbo_max_value_entropy_search([0.0_dp, 0.0_dp], [1.0_dp, -1.0_dp], &
            [0.0_dp], value, status)
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

end program test_entropy
