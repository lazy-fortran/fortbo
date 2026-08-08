program test_workers
    !! BO3: asynchronous workers, fantasies, retries, and failures.
    !!
    !! The oracles are the behaviours that separate a correct asynchronous run
    !! from one that merely runs:
    !!
    !!   * a pending point is reported as pending, so the acquisition can
    !!     condition on it. Without this an asynchronous pool dispatches the
    !!     same point to every worker, which the test demonstrates rather than
    !!     describes;
    !!   * a failure never becomes an objective value, however convenient that
    !!     would be;
    !!   * retries are bounded and counted, and an abandoned point is
    !!     distinguishable from a completed one;
    !!   * cost is charged for failed attempts, since a timeout that burned an
    !!     hour cost an hour.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_workers, only: fortbo_worker_pool_t, fortbo_fantasy_name, &
        FORTBO_FANTASY_MEAN, FORTBO_FANTASY_INCUMBENT, FORTBO_FANTASY_WORST, &
        FORTBO_TASK_PENDING, FORTBO_TASK_DONE, FORTBO_TASK_FAILED
    implicit none

    integer :: failures

    failures = 0
    call check_pending_points_are_visible(failures)
    call check_fantasy_policies(failures)
    call check_failure_is_not_a_value(failures)
    call check_retries_are_bounded(failures)
    call check_cost_covers_failed_attempts(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_workers: PASS"
    else
        print *, "test_workers: FAIL", failures
        error stop 1
    end if

contains

    !! The point of the whole module. Three workers each get a different point,
    !! and all three must be visible as pending so the acquisition can avoid
    !! proposing them again.
    subroutine check_pending_points_are_visible(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: pool
        type(fortnum_status_t) :: status
        real(dp) :: pending(4, 2)
        integer :: n_found, k
        logical :: distinct

        call pool%initialize(3, 2, status)
        call expect(status%code == FORTNUM_OK, "the pool initializes", failures)

        call pool%dispatch(1, [0.1_dp, 0.2_dp], status)
        call pool%dispatch(2, [0.5_dp, 0.6_dp], status)
        call pool%dispatch(3, [0.9_dp, 0.1_dp], status)
        call expect(pool%pending_count() == 3, "all three are pending", failures)

        call pool%pending_points(pending, n_found, status)
        call expect(status%code == FORTNUM_OK .and. n_found == 3, &
            "every pending point is reported", failures)

        distinct = .true.
        do k = 2, n_found
            if (maxval(abs(pending(k, :) - pending(1, :))) == 0.0_dp) &
                distinct = .false.
        end do
        call expect(distinct, "the pending points are the ones dispatched", &
            failures)

        ! Completing one frees the worker and removes the point from the
        ! pending set, so the acquisition stops avoiding it.
        call pool%complete(2, 1.5_dp, 0.25_dp, status)
        call expect(status%code == FORTNUM_OK, "an evaluation completes", failures)
        call expect(pool%pending_count() == 2, &
            "a completed evaluation is no longer pending", failures)
        call expect(pool%idle_worker() == 2, "the freed worker becomes idle", &
            failures)
    end subroutine check_pending_points_are_visible

    !! The three policies must actually differ, and the run must be able to say
    !! which one it used. A run that fantasized the worst observed value is not
    !! comparable with one that fantasized the mean.
    subroutine check_fantasy_policies(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: mean_pool, liar_pool, worst_pool
        type(fortnum_status_t) :: status
        real(dp) :: posterior(2), fantasy(2)
        integer :: n_found

        posterior = [0.3_dp, -0.2_dp]

        call mean_pool%initialize(2, 1, status, fantasy_policy=FORTBO_FANTASY_MEAN)
        call mean_pool%dispatch(1, [0.1_dp], status)
        call mean_pool%dispatch(2, [0.7_dp], status)
        call mean_pool%fantasize(posterior, -5.0_dp, 9.0_dp, fantasy, n_found, &
            status)
        call expect(n_found == 2 .and. &
            maxval(abs(fantasy(:2) - posterior)) < 1.0e-14_dp, &
            "the mean policy fantasizes the posterior mean", failures)

        call liar_pool%initialize(2, 1, status, &
            fantasy_policy=FORTBO_FANTASY_INCUMBENT)
        call liar_pool%dispatch(1, [0.1_dp], status)
        call liar_pool%dispatch(2, [0.7_dp], status)
        call liar_pool%fantasize(posterior, -5.0_dp, 9.0_dp, fantasy, n_found, &
            status)
        call expect(all(fantasy(:2) == -5.0_dp), &
            "the incumbent policy fantasizes the incumbent", failures)

        call worst_pool%initialize(2, 1, status, fantasy_policy=FORTBO_FANTASY_WORST)
        call worst_pool%dispatch(1, [0.1_dp], status)
        call worst_pool%dispatch(2, [0.7_dp], status)
        call worst_pool%fantasize(posterior, -5.0_dp, 9.0_dp, fantasy, n_found, &
            status)
        call expect(all(fantasy(:2) == 9.0_dp), &
            "the worst policy fantasizes the worst observed value", failures)

        call expect(fortbo_fantasy_name(FORTBO_FANTASY_MEAN) == "posterior_mean" &
            .and. fortbo_fantasy_name(FORTBO_FANTASY_WORST) == "worst_observed", &
            "the policy is nameable for the run record", failures)
        call expect(fortbo_fantasy_name(-3) == "unknown", &
            "an unrecognized policy is named unknown", failures)
    end subroutine check_fantasy_policies

    !! A crashed job says nothing about the objective. Recording a large value
    !! teaches the surrogate the region is bad, which nobody measured.
    subroutine check_failure_is_not_a_value(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: pool
        type(fortnum_status_t) :: status
        logical :: retrying

        call pool%initialize(1, 1, status, max_attempts=1)
        call pool%dispatch(1, [0.4_dp], status)
        call pool%fail(1, 0.5_dp, retrying, status)
        call expect(status%code == FORTNUM_OK, "a failure records", failures)
        call expect(.not. retrying, "a single-attempt failure is not retried", &
            failures)

        call expect(pool%tasks(1)%state == FORTBO_TASK_FAILED, &
            "the task is marked failed, not done", failures)
        call expect(pool%completed == 0, &
            "a failure is not counted as a completed evaluation", failures)
        call expect(pool%abandoned == 1, "the abandonment is counted", failures)
        call expect(pool%tasks(1)%value == 0.0_dp, &
            "no objective value is invented for a failure", failures)
    end subroutine check_failure_is_not_a_value

    subroutine check_retries_are_bounded(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: pool
        type(fortnum_status_t) :: status
        logical :: retrying
        integer :: attempt

        call pool%initialize(1, 1, status, max_attempts=3)
        call pool%dispatch(1, [0.4_dp], status)

        ! Two failures leave it retrying; the third exhausts the budget.
        do attempt = 1, 2
            call pool%fail(1, 0.1_dp, retrying, status)
            call expect(retrying, "an attempt within budget is retried", failures)
            call expect(pool%tasks(1)%state == FORTBO_TASK_PENDING, &
                "a retried task stays pending", failures)
        end do
        call pool%fail(1, 0.1_dp, retrying, status)
        call expect(.not. retrying, "the attempt budget is respected", failures)
        call expect(pool%retries == 2, "the retries are counted", failures)
        call expect(pool%abandoned == 1, "the final failure abandons the point", &
            failures)

        ! A retried point stays pending throughout, so the acquisition keeps
        ! avoiding it rather than proposing it again mid-retry.
        call expect(pool%pending_count() == 0, &
            "an abandoned point is no longer pending", failures)
    end subroutine check_retries_are_bounded

    !! Charging only successes makes an unreliable configuration look cheap,
    !! which is exactly backwards for a cost-aware policy.
    subroutine check_cost_covers_failed_attempts(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: pool
        type(fortnum_status_t) :: status
        logical :: retrying

        call pool%initialize(1, 1, status, max_attempts=3)
        call pool%dispatch(1, [0.4_dp], status)
        call pool%fail(1, 2.0_dp, retrying, status)
        call pool%fail(1, 3.0_dp, retrying, status)
        call pool%complete(1, 1.0_dp, 1.0_dp, status)

        call expect(abs(pool%total_cost - 6.0_dp) < 1.0e-12_dp, &
            "failed attempts are charged alongside the successful one", failures)
        call expect(pool%completed == 1, "the eventual success is counted once", &
            failures)
        call expect(pool%tasks(1)%cost == 1.0_dp, &
            "the task records only its own successful cost", failures)
    end subroutine check_cost_covers_failed_attempts

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_worker_pool_t) :: pool, fresh
        type(fortnum_status_t) :: status
        real(dp) :: pending(1, 3)
        logical :: retrying
        integer :: n_found

        call pool%initialize(0, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a pool with no workers is refused", failures)

        call pool%initialize(2, 1, status, max_attempts=0)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero attempt budget is refused", failures)

        call pool%initialize(2, 1, status, fantasy_policy=99)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unknown fantasy policy is refused", failures)

        call fresh%dispatch(1, [0.0_dp], status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "dispatching from an uninitialized pool is refused", failures)

        call pool%initialize(2, 1, status)
        call pool%dispatch(1, [0.5_dp], status)
        ! Overwriting a pending task would lose the evaluation and leave a
        ! fantasy for a point nobody is working on.
        call pool%dispatch(1, [0.6_dp], status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "dispatching over a pending evaluation is refused", failures)

        call pool%complete(2, 1.0_dp, 1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "completing an idle worker is refused", failures)

        call pool%fail(1, -1.0_dp, retrying, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative cost is refused", failures)

        call pool%pending_points(pending, n_found, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a pending buffer of the wrong width is refused", failures)
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

end program test_workers
