program test_optimize
    !! BO3: bound-constrained acquisition optimization.
    !!
    !! Oracles:
    !!   * a dense grid search over the same box. The optimizer must find a
    !!     point at least as good as the best grid point, which is an
    !!     independent computation that shares no code with L-BFGS-B;
    !!   * the returned point must satisfy the first-order conditions: either
    !!     the projected gradient vanishes or the point sits on a bound. That
    !!     is checked from the acquisition's own gradient, so a returned point
    !!     that merely looks good cannot pass;
    !!   * the multistart claim is checked by consequence on a deliberately
    !!     multimodal surface: a single bad start must be recoverable by adding
    !!     starts, and the answer must not depend on start order;
    !!   * bounds are honored exactly, and a surrogate without moment gradients
    !!     is refused rather than differenced.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_acquisition, only: fortbo_ei_t, fortbo_ucb_t
    use fortbo_optimize, only: fortbo_optimize_acquisition
    use fortbo_test_posteriors, only: curved_posterior_t, moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_beats_grid_search(failures)
    call check_first_order_conditions(failures)
    call check_bounds_are_respected(failures)
    call check_multistart_recovers(failures)
    call check_start_order_does_not_matter(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_optimize: PASS"
    else
        print *, "test_optimize: FAIL", failures
        error stop 1
    end if

contains

    !! Independent oracle: exhaustive grid evaluation of the same acquisition
    !! over the same box.
    subroutine grid_best(acquisition, posterior, lower, upper, resolution, &
            best_point, best_value)
        type(fortbo_ucb_t), intent(in) :: acquisition
        type(curved_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: lower(:), upper(:)
        integer, intent(in) :: resolution
        real(dp), intent(out) :: best_point(:)
        real(dp), intent(out) :: best_value
        type(fortnum_status_t) :: status
        real(dp) :: query(1, 2), values(1)
        integer :: i, j

        best_value = -huge(1.0_dp)
        best_point = lower
        do i = 0, resolution
            do j = 0, resolution
                query(1, 1) = lower(1) + (upper(1) - lower(1))*real(i, dp) &
                    /real(resolution, dp)
                query(1, 2) = lower(2) + (upper(2) - lower(2))*real(j, dp) &
                    /real(resolution, dp)
                call acquisition%value(posterior, query, values, status)
                if (status%code /= FORTNUM_OK) cycle
                if (values(1) > best_value) then
                    best_value = values(1)
                    best_point = query(1, :)
                end if
            end do
        end do
    end subroutine grid_best

    subroutine check_beats_grid_search(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), starts(9, 2)
        real(dp) :: best_point(2), best_value
        real(dp) :: grid_point(2), grid_value
        integer :: i, j, k

        posterior%dimension = 2
        ucb%beta = 2.0_dp
        lower = [-2.0_dp, -2.0_dp]
        upper = [2.0_dp, 2.0_dp]

        k = 0
        do i = 0, 2
            do j = 0, 2
                k = k + 1
                starts(k, 1) = -1.5_dp + 1.5_dp*real(i, dp)
                starts(k, 2) = -1.5_dp + 1.5_dp*real(j, dp)
            end do
        end do

        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_OK, "optimization succeeds", failures)

        call grid_best(ucb, posterior, lower, upper, 200, grid_point, grid_value)
        call expect(best_value >= grid_value - 1.0e-6_dp, &
            "the optimizer matches or beats a dense grid search", failures)
    end subroutine check_beats_grid_search

    !! At the returned point, each coordinate must either have a vanishing
    !! gradient component or sit on a bound with the gradient pointing out.
    subroutine check_first_order_conditions(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), starts(4, 2)
        real(dp) :: best_point(2), best_value
        real(dp) :: query(1, 2), gradient(1, 2)
        integer :: j
        logical :: stationary

        posterior%dimension = 2
        ucb%beta = 1.5_dp
        lower = [-3.0_dp, -3.0_dp]
        upper = [3.0_dp, 3.0_dp]
        starts = reshape([-1.0_dp, 0.5_dp, 1.0_dp, -0.5_dp, &
            0.7_dp, -1.2_dp, 0.3_dp, 1.4_dp], [4, 2])

        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_OK, "optimization succeeds", failures)

        query(1, :) = best_point
        call ucb%value_gradient(posterior, query, gradient, status)
        call expect(status%code == FORTNUM_OK, "the gradient is available there", &
            failures)

        stationary = .true.
        do j = 1, 2
            if (abs(gradient(1, j)) < 1.0e-5_dp) cycle
            ! Not stationary in this coordinate, so it must be pinned to the
            ! bound the ascent direction points towards.
            if (gradient(1, j) > 0.0_dp .and. &
                abs(best_point(j) - upper(j)) < 1.0e-8_dp) cycle
            if (gradient(1, j) < 0.0_dp .and. &
                abs(best_point(j) - lower(j)) < 1.0e-8_dp) cycle
            stationary = .false.
        end do
        call expect(stationary, &
            "the result satisfies the first-order conditions on the box", &
            failures)
    end subroutine check_first_order_conditions

    subroutine check_bounds_are_respected(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), starts(3, 2)
        real(dp) :: best_point(2), best_value

        posterior%dimension = 2
        ucb%beta = 3.0_dp
        ! A narrow box far from where the acquisition would like to go.
        lower = [0.25_dp, 0.25_dp]
        upper = [0.35_dp, 0.35_dp]
        starts = reshape([0.3_dp, 0.26_dp, 0.34_dp, &
            0.3_dp, 0.34_dp, 0.26_dp], [3, 2])

        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_OK, "optimization succeeds in a narrow box", &
            failures)
        call expect(all(best_point >= lower - 1.0e-12_dp), &
            "the result respects the lower bound", failures)
        call expect(all(best_point <= upper + 1.0e-12_dp), &
            "the result respects the upper bound", failures)

        ! Starts outside the box must be clamped in, not rejected.
        starts = reshape([-5.0_dp, 9.0_dp, 0.3_dp, &
            -5.0_dp, 9.0_dp, 0.3_dp], [3, 2])
        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_OK, "starts outside the box are clamped", &
            failures)
        call expect(all(best_point >= lower - 1.0e-12_dp) .and. &
            all(best_point <= upper + 1.0e-12_dp), &
            "a clamped start still yields a feasible result", failures)
    end subroutine check_bounds_are_respected

    !! One deliberately poor start versus that same start plus better company.
    !! Adding starts must never make the answer worse.
    subroutine check_multistart_recovers(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2)
        real(dp) :: single(1, 2), many(5, 2)
        real(dp) :: single_point(2), single_value
        real(dp) :: many_point(2), many_value
        integer :: converged

        posterior%dimension = 2
        ucb%beta = 2.0_dp
        lower = [-2.0_dp, -2.0_dp]
        upper = [2.0_dp, 2.0_dp]

        single = reshape([0.0_dp, 0.0_dp], [1, 2])
        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, single, &
            single_point, single_value, status)
        call expect(status%code == FORTNUM_OK, "a single start runs", failures)

        many(1, :) = [0.0_dp, 0.0_dp]
        many(2, :) = [-1.9_dp, -1.9_dp]
        many(3, :) = [1.9_dp, 1.9_dp]
        many(4, :) = [-1.9_dp, 1.9_dp]
        many(5, :) = [1.9_dp, -1.9_dp]
        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, many, &
            many_point, many_value, status, &
            converged_starts=converged)
        call expect(status%code == FORTNUM_OK, "multistart runs", failures)
        call expect(many_value >= single_value - 1.0e-12_dp, &
            "adding starts never makes the answer worse", failures)
        call expect(converged > 0, "at least one start converged", failures)
        call expect(converged <= 5, "the converged count cannot exceed the starts", &
            failures)
    end subroutine check_multistart_recovers

    !! Reordering the starts must not change the answer, because ties break on
    !! value and the search from each start is independent.
    subroutine check_start_order_does_not_matter(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_ucb_t) :: ucb
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), forward(4, 2), reversed(4, 2)
        real(dp) :: point_a(2), value_a, point_b(2), value_b
        integer :: k

        posterior%dimension = 2
        ucb%beta = 2.0_dp
        lower = [-2.0_dp, -2.0_dp]
        upper = [2.0_dp, 2.0_dp]

        forward(1, :) = [-1.5_dp, -0.5_dp]
        forward(2, :) = [0.5_dp, 1.5_dp]
        forward(3, :) = [1.2_dp, -1.2_dp]
        forward(4, :) = [-0.3_dp, 0.8_dp]
        do k = 1, 4
            reversed(k, :) = forward(5 - k, :)
        end do

        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, forward, &
            point_a, value_a, status)
        call fortbo_optimize_acquisition(ucb, posterior, lower, upper, reversed, &
            point_b, value_b, status)
        call expect(abs(value_a - value_b) < 1.0e-8_dp, &
            "the best value does not depend on start order", failures)
    end subroutine check_start_order_does_not_matter

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(moments_only_posterior_t) :: partial
        type(fortbo_ei_t) :: ei
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), starts(2, 2)
        real(dp) :: best_point(2), best_value
        real(dp) :: narrow_starts(2, 1)

        posterior%dimension = 2
        partial%dimension = 2
        lower = [0.0_dp, 0.0_dp]
        upper = [1.0_dp, 1.0_dp]
        starts = 0.5_dp

        call fortbo_optimize_acquisition(ei, partial, lower, upper, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a surrogate without moment gradients is refused", failures)
        call expect(index(status%msg, "sampling search") > 0, &
            "the refusal names the alternative", failures)

        call fortbo_optimize_acquisition(ei, posterior, upper, lower, starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "inverted bounds are refused", failures)

        narrow_starts = 0.5_dp
        call fortbo_optimize_acquisition(ei, posterior, lower, upper, narrow_starts, &
            best_point, best_value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "starts of the wrong width are refused", failures)
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

end program test_optimize
