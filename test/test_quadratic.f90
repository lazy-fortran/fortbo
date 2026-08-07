program test_quadratic
    !! BO3T: the bound-constrained quadratic subproblem.
    !!
    !! Oracles:
    !!   * for a positive-definite Hessian with an interior minimizer, the exact
    !!     answer is `-H^{-1} g`, which the test computes independently by
    !!     Cholesky. No optimizer is involved in producing the expected value;
    !!   * when the box is active, a dense grid over the box provides the
    !!     reference. The solver must match or beat it;
    !!   * for an indefinite Hessian the test checks that the flag is raised,
    !!     that the step lands on the boundary, and that the model value is
    !!     genuinely lower than at the start — the behavior that positive-
    !!     definite projection would destroy;
    !!   * the value and gradient of the model are checked against each other by
    !!     central differences, so the callback the optimizer sees is correct.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortbo_quadratic, only: fortbo_solve_quadratic_subproblem, &
        fortbo_quadratic_value
    implicit none

    integer :: failures

    failures = 0
    call check_model_value_and_gradient(failures)
    call check_interior_newton_step(failures)
    call check_active_bounds_against_grid(failures)
    call check_indefinite_is_not_projected(failures)
    call check_negative_curvature_direction(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_quadratic: PASS"
    else
        print *, "test_quadratic: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_model_value_and_gradient(failures)
        integer, intent(inout) :: failures
        real(dp) :: gradient(3), hessian(3, 3), step(3), shifted(3)
        real(dp) :: analytic(3), numeric, plus, minus
        real(dp), parameter :: h = 1.0e-6_dp
        integer :: j
        logical :: matches

        gradient = [0.5_dp, -1.25_dp, 2.0_dp]
        hessian = reshape([3.0_dp, 0.4_dp, -0.2_dp, &
            0.4_dp, 2.5_dp, 0.7_dp, &
            -0.2_dp, 0.7_dp, 1.8_dp], [3, 3])
        step = [0.3_dp, -0.6_dp, 0.15_dp]

        analytic = gradient + matmul(hessian, step)
        matches = .true.
        do j = 1, 3
            shifted = step
            shifted(j) = step(j) + h
            plus = fortbo_quadratic_value(gradient, hessian, shifted)
            shifted(j) = step(j) - h
            minus = fortbo_quadratic_value(gradient, hessian, shifted)
            numeric = (plus - minus)/(2.0_dp*h)
            if (abs(analytic(j) - numeric) > 1.0e-6_dp) matches = .false.
        end do
        call expect(matches, "the model gradient matches its own value", failures)
        call expect(fortbo_quadratic_value(gradient, hessian, [0.0_dp, 0.0_dp, 0.0_dp]) &
            == 0.0_dp, "the model vanishes at the origin", failures)
    end subroutine check_model_value_and_gradient

    !! Independent oracle: solve H s = -g by Cholesky in the test.
    subroutine check_interior_newton_step(failures)
        integer, intent(inout) :: failures
        type(cholesky_factorization_t) :: factorization
        type(fortnum_status_t) :: status
        real(dp) :: gradient(3), hessian(3, 3), lower(3), upper(3)
        real(dp) :: step(3), expected(3), model_value
        logical :: indefinite

        gradient = [0.5_dp, -1.25_dp, 2.0_dp]
        hessian = reshape([3.0_dp, 0.4_dp, -0.2_dp, &
            0.4_dp, 2.5_dp, 0.7_dp, &
            -0.2_dp, 0.7_dp, 1.8_dp], [3, 3])
        lower = -10.0_dp
        upper = 10.0_dp

        expected = -gradient
        call factorization%factorize(hessian, status)
        call expect(status%code == FORTNUM_OK, "the reference Hessian factorizes", &
            failures)
        call factorization%solve(expected, status)

        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            [0.0_dp, 0.0_dp, 0.0_dp], step, &
            model_value, status, indefinite)
        call expect(status%code == FORTNUM_OK, "the subproblem solves", failures)
        call expect(.not. indefinite, "a positive-definite Hessian is not flagged", &
            failures)
        call expect(maxval(abs(step - expected)) < 1.0e-6_dp, &
            "the interior step is the Newton step", failures)
        call expect(abs(model_value - fortbo_quadratic_value(gradient, hessian, &
            expected)) < 1.0e-9_dp, &
            "the reported model value matches the step", failures)
    end subroutine check_interior_newton_step

    !! With the box tight enough to bind, compare against a dense grid.
    subroutine check_active_bounds_against_grid(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: gradient(2), hessian(2, 2), lower(2), upper(2)
        real(dp) :: step(2), model_value, trial(2), best_value
        integer, parameter :: resolution = 600
        integer :: i, j
        logical :: indefinite

        gradient = [2.0_dp, -3.0_dp]
        hessian = reshape([1.0_dp, 0.3_dp, 0.3_dp, 1.5_dp], [2, 2])
        lower = [-0.25_dp, -0.25_dp]
        upper = [0.25_dp, 0.25_dp]

        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            [0.0_dp, 0.0_dp], step, model_value, &
            status, indefinite)
        call expect(status%code == FORTNUM_OK, "the bounded subproblem solves", failures)
        call expect(all(step >= lower - 1.0e-12_dp) .and. &
            all(step <= upper + 1.0e-12_dp), &
            "the step respects the box", failures)

        best_value = huge(1.0_dp)
        do i = 0, resolution
            do j = 0, resolution
                trial(1) = lower(1) + (upper(1) - lower(1))*real(i, dp) &
                    /real(resolution, dp)
                trial(2) = lower(2) + (upper(2) - lower(2))*real(j, dp) &
                    /real(resolution, dp)
                best_value = min(best_value, &
                    fortbo_quadratic_value(gradient, hessian, trial))
            end do
        end do
        call expect(model_value <= best_value + 1.0e-8_dp, &
            "the solver matches or beats a dense grid over the box", failures)
    end subroutine check_active_bounds_against_grid

    !! An indefinite Hessian must be reported and must not be repaired. If the
    !! solver silently added a multiple of the identity, the step would stop
    !! short of the boundary and the model decrease would be smaller.
    subroutine check_indefinite_is_not_projected(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: gradient(2), hessian(2, 2), lower(2), upper(2)
        real(dp) :: step(2), model_value, start_value
        logical :: indefinite, on_boundary

        ! Eigenvalues 1 and -1: a saddle.
        gradient = [0.0_dp, 0.0_dp]
        hessian = reshape([1.0_dp, 0.0_dp, 0.0_dp, -1.0_dp], [2, 2])
        lower = [-1.0_dp, -1.0_dp]
        upper = [1.0_dp, 1.0_dp]

        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            [0.1_dp, 0.1_dp], step, model_value, &
            status, indefinite)
        call expect(status%code == FORTNUM_OK, "the indefinite subproblem solves", &
            failures)
        call expect(indefinite, "the indefinite Hessian is reported", failures)

        start_value = fortbo_quadratic_value(gradient, hessian, [0.1_dp, 0.1_dp])
        call expect(model_value < start_value, &
            "the step decreases the model despite negative curvature", failures)

        on_boundary = abs(abs(step(2)) - 1.0_dp) < 1.0e-6_dp
        call expect(on_boundary, &
            "negative curvature drives the step to the boundary", failures)
        call expect(abs(model_value + 0.5_dp) < 1.0e-4_dp, &
            "the model value reaches the exact saddle minimum on the box", &
            failures)
    end subroutine check_indefinite_is_not_projected

    !! A purely negative-definite model: every direction curves down, so the
    !! solution is a corner and the value is strictly below any interior point.
    subroutine check_negative_curvature_direction(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: gradient(2), hessian(2, 2), lower(2), upper(2)
        real(dp) :: step(2), model_value
        logical :: indefinite

        gradient = [0.2_dp, -0.1_dp]
        hessian = reshape([-2.0_dp, 0.0_dp, 0.0_dp, -3.0_dp], [2, 2])
        lower = [-1.0_dp, -1.0_dp]
        upper = [1.0_dp, 1.0_dp]

        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            [0.0_dp, 0.0_dp], step, model_value, &
            status, indefinite)
        call expect(indefinite, "a negative-definite Hessian is reported", failures)
        call expect(maxval(abs(abs(step) - 1.0_dp)) < 1.0e-6_dp, &
            "the solution is a corner of the box", failures)
        call expect(model_value < fortbo_quadratic_value(gradient, hessian, &
            [0.0_dp, 0.0_dp]), &
            "the corner beats the origin", failures)
    end subroutine check_negative_curvature_direction

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: gradient(2), hessian(2, 2), lower(2), upper(2)
        real(dp) :: step(2), model_value
        real(dp) :: wide(3, 3)

        gradient = [1.0_dp, 1.0_dp]
        hessian = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], [2, 2])
        lower = [-1.0_dp, -1.0_dp]
        upper = [1.0_dp, 1.0_dp]

        wide = 0.0_dp
        call fortbo_solve_quadratic_subproblem(gradient, wide, lower, upper, &
            [0.0_dp, 0.0_dp], step, model_value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-shaped Hessian is refused", failures)

        call fortbo_solve_quadratic_subproblem(gradient, hessian, upper, lower, &
            [0.0_dp, 0.0_dp], step, model_value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "inverted bounds are refused", failures)

        ! A nonsymmetric Hessian means two derivations disagree; refuse rather
        ! than silently averaging them.
        hessian = reshape([1.0_dp, 0.5_dp, -0.5_dp, 1.0_dp], [2, 2])
        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            [0.0_dp, 0.0_dp], step, model_value, &
            status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a nonsymmetric Hessian is refused, not symmetrized", failures)
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

end program test_quadratic
