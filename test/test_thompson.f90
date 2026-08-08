program test_thompson
    !! BO1: Thompson sampling as a standalone batch rule.
    !!
    !! Oracles:
    !!
    !!   * the rule must run against a *non-GP* provider unchanged, which is
    !!     what "standalone" means operationally;
    !!   * selection is without replacement, so a batch never spends two slots
    !!     on the same candidate;
    !!   * the rule must actually prefer good candidates. Measured over many
    !!     draws against a posterior whose means are known, not asserted;
    !!   * a posterior offering only marginal moments is refused, since
    !!     independent marginals would let two near-identical candidates take
    !!     two slots.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_thompson, only: fortbo_thompson_batch
    use fortbo_test_posteriors, only: demo_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_selection_is_without_replacement(failures)
    call check_good_candidates_are_preferred(failures)
    call check_replayable(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_thompson: PASS"
    else
        print *, "test_thompson: FAIL", failures
        error stop 1
    end if

contains

    !! A batch that evaluated one point q times would be a correct arg-min per
    !! slot and a useless batch.
    subroutine check_selection_is_without_replacement(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n = 12, q = 4
        real(dp) :: candidates(n, 1), points(q, 1)
        integer :: selected(q), trial, i, j
        logical :: distinct, in_range, matched

        posterior%dimension = 1
        do i = 1, n
            candidates(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n - 1, dp)
        end do

        distinct = .true.
        in_range = .true.
        matched = .true.
        call rng_seed(generator, int(191919, int64), status)
        do trial = 1, 60
            call fortbo_thompson_batch(posterior, candidates, generator, selected, &
                status, points)
            if (status%code /= FORTNUM_OK) distinct = .false.
            do i = 1, q
                if (selected(i) < 1 .or. selected(i) > n) in_range = .false.
                if (abs(points(i, 1) - candidates(selected(i), 1)) > 0.0_dp) &
                    matched = .false.
                do j = i + 1, q
                    if (selected(i) == selected(j)) distinct = .false.
                end do
            end do
        end do
        call expect(in_range, "every selection indexes a real candidate", failures)
        call expect(distinct, "no candidate takes two slots in one batch", failures)
        call expect(matched, "the returned points are the selected rows", failures)
    end subroutine check_selection_is_without_replacement

    !! The rule must prefer candidates the posterior thinks are good. With a
    !! known mean this is measurable: over many batches the better half of the
    !! candidate set must be selected more often than the worse half.
    subroutine check_good_candidates_are_preferred(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n = 10, q = 2, trials = 400
        real(dp) :: candidates(n, 1), mean(n), variance(n)
        integer :: selected(q), counts(n), trial, i, order(n)
        integer :: better, worse

        posterior%dimension = 1
        do i = 1, n
            candidates(i, 1) = -1.5_dp + 3.0_dp*real(i - 1, dp)/real(n - 1, dp)
        end do
        call posterior%moments(candidates, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "the posterior evaluates", failures)

        counts = 0
        call rng_seed(generator, int(838383, int64), status)
        do trial = 1, trials
            call fortbo_thompson_batch(posterior, candidates, generator, selected, &
                status)
            if (status%code /= FORTNUM_OK) exit
            do i = 1, q
                counts(selected(i)) = counts(selected(i)) + 1
            end do
        end do

        ! Split by posterior mean, lower being better under minimization.
        call rank_ascending(mean, order)
        better = 0
        worse = 0
        do i = 1, n/2
            better = better + counts(order(i))
        end do
        do i = n/2 + 1, n
            worse = worse + counts(order(i))
        end do
        call expect(better > worse, &
            "candidates the posterior rates better are selected more often", &
            failures)
        call expect(worse > 0, &
            "the rule still explores, rather than collapsing onto the best mean", &
            failures)
    end subroutine check_good_candidates_are_preferred

    subroutine rank_ascending(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, moving

        do i = 1, size(order)
            order(i) = i
        end do
        do i = 2, size(order)
            moving = order(i)
            j = i - 1
            do while (j >= 1)
                if (values(order(j)) <= values(moving)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = moving
        end do
    end subroutine rank_ascending

    !! Same seed, same batch. Without this a run cannot be reproduced from its
    !! record, and two policies cannot be compared on the same realizations.
    subroutine check_replayable(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(rng_t) :: first, second
        type(fortnum_status_t) :: status
        integer, parameter :: n = 8, q = 3
        real(dp) :: candidates(n, 1)
        integer :: selected_a(q), selected_b(q), i

        posterior%dimension = 1
        do i = 1, n
            candidates(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n - 1, dp)
        end do

        call rng_seed(first, int(4242, int64), status)
        call rng_seed(second, int(4242, int64), status)
        call fortbo_thompson_batch(posterior, candidates, first, selected_a, status)
        call fortbo_thompson_batch(posterior, candidates, second, selected_b, status)
        call expect(all(selected_a == selected_b), &
            "the same seed replays the same batch", failures)

        call rng_seed(second, int(9999, int64), status)
        call fortbo_thompson_batch(posterior, candidates, second, selected_b, status)
        call expect(any(selected_a /= selected_b), &
            "a different seed gives a different batch", failures)
    end subroutine check_replayable

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(demo_posterior_t) :: posterior
        type(moments_only_posterior_t) :: plain
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: candidates(3, 1), wide(2, 3)
        integer :: selected(2), oversized(5)

        posterior%dimension = 1
        plain%dimension = 1
        candidates(:, 1) = [-0.5_dp, 0.0_dp, 0.5_dp]
        call rng_seed(generator, int(1, int64), status)

        ! Independent marginals would let two near-identical candidates take two
        ! slots, so a posterior without joint sampling is refused rather than
        ! accommodated.
        call fortbo_thompson_batch(plain, candidates, generator, selected, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a posterior without joint sampling refuses by capability", failures)

        call fortbo_thompson_batch(posterior, candidates, generator, oversized, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a batch larger than the candidate set is refused", failures)

        call fortbo_thompson_batch(posterior, candidates, generator, selected, &
            status, wide)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-shaped point buffer is refused", failures)
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

end program test_thompson
