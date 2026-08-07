program test_turbo
    !! BO3T: TuRBO candidate generation and Thompson selection.
    !!
    !! Oracles:
    !!   * the perturbation rule is checked statistically against its own
    !!     definition — the mean number of moved coordinates per candidate must
    !!     equal `d * min(1, 20/d)`, which is 20 in high dimension. That is the
    !!     property the method depends on, and a dense-sampling regression would
    !!     show up here and nowhere else;
    !!   * every candidate must lie inside the region bounds computed
    !!     independently in the test, and must differ from the center;
    !!   * Thompson selection is checked against a brute-force arg-min written
    !!     in the test, including the distinctness and tie-breaking rules;
    !!   * the across-region bandit behavior is checked by the observable
    !!     consequence: a region whose realizations are uniformly better must
    !!     win the whole batch, and two equally good regions must split it.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_trust_region, only: fortbo_trust_region_t
    use fortbo_turbo, only: fortbo_candidate_count, fortbo_perturbation_probability, &
        fortbo_turbo_candidates, fortbo_thompson_select, &
        fortbo_thompson_gradient_refusal
    implicit none

    integer :: failures

    failures = 0
    call check_reference_counts(failures)
    call check_candidates_lie_in_region(failures)
    call check_perturbation_sparsity(failures)
    call check_candidates_are_replayable(failures)
    call check_thompson_matches_brute_force(failures)
    call check_thompson_is_a_bandit(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_turbo: PASS"
    else
        print *, "test_turbo: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_reference_counts(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_candidate_count(1) == 100, "n_cand is 100 d in low dimension", &
            failures)
        call expect(fortbo_candidate_count(20) == 2000, "n_cand scales with dimension", &
            failures)
        call expect(fortbo_candidate_count(200) == 5000, "n_cand caps at 5000", failures)

        call expect(fortbo_perturbation_probability(5) == 1.0_dp, &
            "every coordinate moves below twenty dimensions", failures)
        call expect(fortbo_perturbation_probability(20) == 1.0_dp, &
            "every coordinate still moves at twenty dimensions", failures)
        call expect(abs(fortbo_perturbation_probability(200) - 0.1_dp) < 1.0e-14_dp, &
            "p is 20/d above twenty dimensions", failures)
    end subroutine check_reference_counts

    subroutine check_candidates_lie_in_region(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(4), lower(4), upper(4)
        real(dp) :: candidates(400, 4)
        integer :: i, j
        logical :: inside, moved

        call region%initialize(4, 8, status)
        call region%restart([0.5_dp, 0.3_dp, 0.7_dp, 0.5_dp], 1.0_dp, status)
        lengthscales = [1.0_dp, 2.0_dp, 0.5_dp, 1.0_dp]
        call region%bounds(lengthscales, lower, upper, status)

        call rng_seed(generator, int(4242, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, candidates, status)
        call expect(status%code == FORTNUM_OK, "candidate generation succeeds", failures)

        inside = .true.
        moved = .true.
        do i = 1, size(candidates, 1)
            do j = 1, 4
                if (candidates(i, j) < lower(j) - 1.0e-12_dp) inside = .false.
                if (candidates(i, j) > upper(j) + 1.0e-12_dp) inside = .false.
            end do
            if (maxval(abs(candidates(i, :) - region%center)) == 0.0_dp) moved = .false.
        end do
        call expect(inside, "every candidate lies inside the region bounds", failures)
        call expect(moved, "no candidate is the unperturbed center", failures)
        call expect(all(candidates >= 0.0_dp) .and. all(candidates <= 1.0_dp), &
            "every candidate stays in the unit cube", failures)
    end subroutine check_candidates_lie_in_region

    !! In high dimension the expected number of moved coordinates is exactly
    !! twenty, whatever the dimension. This is the sparsity that lets TuRBO work
    !! at hundreds of dimensions.
    subroutine check_perturbation_sparsity(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: d = 200
        integer, parameter :: n = 2000
        real(dp), allocatable :: lengthscales(:), candidates(:, :)
        real(dp) :: average_moved, standard_error
        integer :: i, j, moved_total

        allocate (lengthscales(d), candidates(n, d))
        lengthscales = 1.0_dp

        call region%initialize(d, 50, status)
        call region%restart(spread(0.5_dp, 1, d), 1.0_dp, status)
        call rng_seed(generator, int(777, int64), status)

        ! Dimension 200 exceeds the tabulated Sobol dimensions, so the caller
        ! must opt into pseudorandom perturbations explicitly.
        call fortbo_turbo_candidates(region, lengthscales, generator, candidates, &
            status, quasi_random=.false.)
        call expect(status%code == FORTNUM_OK, &
            "high-dimensional generation succeeds with an explicit opt-in", &
            failures)

        moved_total = 0
        do i = 1, n
            do j = 1, d
                if (candidates(i, j) /= region%center(j)) moved_total = moved_total + 1
            end do
        end do
        average_moved = real(moved_total, dp)/real(n, dp)

        ! Binomial(d, 20/d) has variance 20(1 - 20/d) < 20.
        standard_error = sqrt(20.0_dp/real(n, dp))
        call expect(abs(average_moved - 20.0_dp) < 5.0_dp*standard_error, &
            "about twenty coordinates move per candidate at d = 200", failures)
        call expect(average_moved < 0.2_dp*real(d, dp), &
            "the perturbation is sparse, not dense", failures)
    end subroutine check_perturbation_sparsity

    subroutine check_candidates_are_replayable(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(3), first(200, 3), second(200, 3)

        call region%initialize(3, 4, status)
        call region%restart([0.4_dp, 0.6_dp, 0.5_dp], 1.0_dp, status)
        lengthscales = [1.0_dp, 1.0_dp, 1.0_dp]

        call rng_seed(generator, int(31, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, first, status)
        call rng_seed(generator, int(31, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, second, status)
        call expect(maxval(abs(first - second)) == 0.0_dp, &
            "the same seed reproduces the same candidates bitwise", failures)

        ! At three dimensions p = 1, so every coordinate is perturbed and the
        ! whole candidate cloud comes from the Sobol sequence. It is therefore
        ! seed-independent by construction, and that is the point: the cover of
        ! the region is designed rather than drawn. The seed only chooses which
        ! coordinates move, which matters above twenty dimensions.
        call rng_seed(generator, int(32, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, second, status)
        call expect(maxval(abs(first - second)) == 0.0_dp, &
            "the quasi-random cover is seed-independent when p = 1", failures)

        call rng_seed(generator, int(31, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, first, status, &
            quasi_random=.false.)
        call rng_seed(generator, int(32, int64), status)
        call fortbo_turbo_candidates(region, lengthscales, generator, second, status, &
            quasi_random=.false.)
        call expect(maxval(abs(first - second)) > 0.0_dp, &
            "pseudorandom perturbations do depend on the seed", failures)
    end subroutine check_candidates_are_replayable

    !! Brute-force oracle written in the test: repeatedly take the smallest
    !! unclaimed value of each realization.
    subroutine check_thompson_matches_brute_force(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: n_candidates = 40
        integer, parameter :: batch_size = 6
        real(dp) :: samples(n_candidates, batch_size)
        integer :: selected(batch_size), expected(batch_size)
        logical :: taken(n_candidates)
        real(dp) :: best_value
        integer :: s, c, best
        logical :: distinct

        do s = 1, batch_size
            do c = 1, n_candidates
                samples(c, s) = sin(1.7_dp*real(c, dp) + 0.9_dp*real(s, dp))
            end do
        end do

        call fortbo_thompson_select(samples, selected, status)
        call expect(status%code == FORTNUM_OK, "selection succeeds", failures)

        taken = .false.
        do s = 1, batch_size
            best = 0
            best_value = huge(1.0_dp)
            do c = 1, n_candidates
                if (taken(c)) cycle
                if (samples(c, s) < best_value) then
                    best_value = samples(c, s)
                    best = c
                end if
            end do
            expected(s) = best
            taken(best) = .true.
        end do
        call expect(all(selected == expected), &
            "selection matches a brute-force per-realization arg-min", failures)

        distinct = .true.
        do s = 1, batch_size
            do c = 1, s - 1
                if (selected(s) == selected(c)) distinct = .false.
            end do
        end do
        call expect(distinct, "the batch holds distinct candidates", failures)

        ! Ties must go to the lowest index so a replay picks the same batch.
        samples = 1.0_dp
        call fortbo_thompson_select(samples, selected, status)
        call expect(all(selected == [1, 2, 3, 4, 5, 6]), &
            "ties resolve to the lowest index", failures)
    end subroutine check_thompson_matches_brute_force

    !! The bandit behavior is emergent, so it is checked by consequence: with
    !! two regions' candidates concatenated, a uniformly better region takes the
    !! whole batch, and two equally good regions share it.
    subroutine check_thompson_is_a_bandit(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: per_region = 20
        integer, parameter :: batch_size = 8
        real(dp) :: samples(2*per_region, batch_size)
        integer :: selected(batch_size)
        integer :: s, c, from_first

        ! Region one occupies rows 1..20 and is uniformly better.
        do s = 1, batch_size
            do c = 1, per_region
                samples(c, s) = -5.0_dp + 0.01_dp*real(c + s, dp)
                samples(per_region + c, s) = 5.0_dp + 0.01_dp*real(c + s, dp)
            end do
        end do
        call fortbo_thompson_select(samples, selected, status)
        from_first = count(selected <= per_region)
        call expect(from_first == batch_size, &
            "a uniformly better region wins the whole batch", failures)

        ! Now make the regions alternate in quality by realization.
        do s = 1, batch_size
            do c = 1, per_region
                if (mod(s, 2) == 1) then
                    samples(c, s) = -1.0_dp + 0.001_dp*real(c, dp)
                    samples(per_region + c, s) = 1.0_dp + 0.001_dp*real(c, dp)
                else
                    samples(c, s) = 1.0_dp + 0.001_dp*real(c, dp)
                    samples(per_region + c, s) = -1.0_dp + 0.001_dp*real(c, dp)
                end if
            end do
        end do
        call fortbo_thompson_select(samples, selected, status)
        from_first = count(selected <= per_region)
        call expect(from_first == batch_size/2, &
            "regions that alternate in quality split the batch", failures)
    end subroutine check_thompson_is_a_bandit

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(30), candidates(10, 30), narrow(10, 2)
        real(dp) :: wide_lengthscales(1200), wide_candidates(4, 1200)
        real(dp) :: samples(5, 2)
        integer :: selected(2), oversized(9)

        lengthscales = 1.0_dp
        call rng_seed(generator, int(5, int64), status)

        call region%initialize(30, 4, status)
        call region%restart(spread(0.5_dp, 1, 30), 1.0_dp, status)
        call fortbo_turbo_candidates(region, lengthscales, generator, candidates, status)
        call expect(status%code == FORTNUM_OK, &
            "thirty dimensions are now within the Sobol construction", failures)

        ! Past the enumeration limit the refusal must still be explicit rather
        ! than a silent substitution.
        block
            type(fortbo_trust_region_t) :: huge_region
            real(dp) :: wide_lengthscales(1200), wide_candidates(4, 1200)

            wide_lengthscales = 1.0_dp
            call huge_region%initialize(1200, 4, status)
            call huge_region%restart(spread(0.5_dp, 1, 1200), 1.0_dp, status)
            call fortbo_turbo_candidates(huge_region, wide_lengthscales, generator, &
                wide_candidates, status)
            call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
                "quasi-random beyond the enumeration limit is refused", failures)
            call expect(index(status%msg, "fortnum_sobol") > 0, &
                "the refusal names the upstream limit", failures)
            call fortbo_turbo_candidates(huge_region, wide_lengthscales, generator, &
                wide_candidates, status, quasi_random=.false.)
            call expect(status%code == FORTNUM_OK, &
                "an explicit pseudorandom opt-in still works there", failures)
        end block

        call fortbo_turbo_candidates(region, lengthscales, generator, narrow, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a candidate width mismatch is refused", failures)

        samples = 0.0_dp
        call fortbo_thompson_select(samples, oversized, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a batch larger than the candidate set is refused", failures)
        call fortbo_thompson_select(samples(:, 1:1), selected, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "fewer realizations than batch members is refused", failures)

        call fortbo_thompson_gradient_refusal(status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "the discrete arg-min refuses a derivative", failures)
        call expect(index(status%msg, "arg-min") > 0, &
            "the refusal explains why no derivative exists", failures)
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

end program test_turbo
