program test_benchmarks
    !! BO6: synthetic objectives with known optima.
    !!
    !! Oracles:
    !!   * the recorded optimal value must be attained at the recorded
    !!     optimizer, to the precision the literature quotes it;
    !!   * no point of a dense grid, and no point of a fine local sweep around
    !!     the optimizer, may beat the recorded value. That is what makes the
    !!     recorded value usable as a regret baseline rather than an unverified
    !!     constant copied from a table;
    !!   * the gradient must vanish at every interior optimizer, and must match
    !!     central differences everywhere else. An analytic gradient with a sign
    !!     error passes a smoke test and fails both of these;
    !!   * the Ackley gradient is checked *at the origin*, which is both its
    !!     optimum and the one point where the natural expression divides by
    !!     zero.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_benchmarks, only: fortbo_benchmark_t, fortbo_benchmark_name, &
        fortbo_benchmark_fixed_dimension, FORTBO_BENCH_BRANIN, FORTBO_BENCH_HARTMANN3, &
        FORTBO_BENCH_HARTMANN6, FORTBO_BENCH_ACKLEY, FORTBO_BENCH_ROSENBROCK, &
        FORTBO_BENCH_LEVY, FORTBO_BENCH_SPHERE
    implicit none

    integer :: failures

    failures = 0
    call check_optimal_values(failures)
    call check_no_grid_point_beats_the_optimum(failures)
    call check_local_sweep_around_optimum(failures)
    call check_gradients_against_differences(failures)
    call check_gradient_vanishes_at_optimum(failures)
    call check_ackley_origin(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_benchmarks: PASS"
    else
        print *, "test_benchmarks: FAIL", failures
        error stop 1
    end if

contains

    subroutine make(kind, dimension, benchmark)
        integer, intent(in) :: kind
        integer, intent(in) :: dimension
        type(fortbo_benchmark_t), intent(out) :: benchmark

        benchmark%kind = kind
        benchmark%dimension = dimension
        if (fortbo_benchmark_fixed_dimension(kind) > 0) then
            benchmark%dimension = fortbo_benchmark_fixed_dimension(kind)
        end if
    end subroutine make

    subroutine check_optimal_values(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:)
        real(dp) :: value
        integer :: kinds(7), k

        kinds = [FORTBO_BENCH_BRANIN, FORTBO_BENCH_HARTMANN3, FORTBO_BENCH_HARTMANN6, &
            FORTBO_BENCH_ACKLEY, FORTBO_BENCH_ROSENBROCK, FORTBO_BENCH_LEVY, &
            FORTBO_BENCH_SPHERE]

        do k = 1, size(kinds)
            call make(kinds(k), 4, benchmark)
            allocate (point(benchmark%dimension))
            call benchmark%optimum(point, status)
            call expect(status%code == FORTNUM_OK, &
                "the optimizer is available for "// &
                fortbo_benchmark_name(kinds(k)), failures)
            call benchmark%value(point, value, status)
            call expect(status%code == FORTNUM_OK, &
                "the value is available for "// &
                fortbo_benchmark_name(kinds(k)), failures)
            ! Hartmann's quoted optimizer is given to six digits, so the value
            ! matches only to about that precision; the others are exact.
            call expect(abs(value - benchmark%optimal_value()) < 1.0e-5_dp, &
                "the recorded optimum is attained for "// &
                fortbo_benchmark_name(kinds(k)), failures)
            deallocate (point)
        end do
    end subroutine check_optimal_values

    !! Dense grid over the declared box for the low-dimensional functions. If a
    !! grid point beats the recorded optimum, the constant is wrong.
    subroutine check_no_grid_point_beats_the_optimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp) :: lower(2), upper(2), point(2), value, best
        integer :: i, j
        integer, parameter :: resolution = 400

        call make(FORTBO_BENCH_BRANIN, 2, benchmark)
        call benchmark%bounds(lower, upper, status)
        best = huge(1.0_dp)
        do i = 0, resolution
            do j = 0, resolution
                point(1) = lower(1) + (upper(1) - lower(1))*real(i, dp) &
                    /real(resolution, dp)
                point(2) = lower(2) + (upper(2) - lower(2))*real(j, dp) &
                    /real(resolution, dp)
                call benchmark%value(point, value, status)
                best = min(best, value)
            end do
        end do
        call expect(best >= benchmark%optimal_value() - 1.0e-9_dp, &
            "no Branin grid point beats the recorded optimum", failures)
        call expect(best < benchmark%optimal_value() + 1.0e-2_dp, &
            "the Branin grid comes close to the recorded optimum", failures)
    end subroutine check_no_grid_point_beats_the_optimum

    !! Local sweep: perturb each coordinate of the optimizer in both directions
    !! and require that nothing improves. This catches a wrong optimizer for the
    !! higher-dimensional functions where a full grid is impossible.
    subroutine check_local_sweep_around_optimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:), trial(:)
        real(dp) :: base, value, step
        integer :: kinds(4), k, j, s
        logical :: minimal

        kinds = [FORTBO_BENCH_HARTMANN3, FORTBO_BENCH_HARTMANN6, &
            FORTBO_BENCH_LEVY, FORTBO_BENCH_ROSENBROCK]

        do k = 1, size(kinds)
            call make(kinds(k), 5, benchmark)
            allocate (point(benchmark%dimension), trial(benchmark%dimension))
            call benchmark%optimum(point, status)
            call benchmark%value(point, base, status)

            minimal = .true.
            do j = 1, benchmark%dimension
                do s = 1, 6
                    step = 10.0_dp**(-real(s, dp))
                    trial = point
                    trial(j) = point(j) + step
                    call benchmark%value(trial, value, status)
                    if (value < base - 1.0e-9_dp) minimal = .false.
                    trial(j) = point(j) - step
                    call benchmark%value(trial, value, status)
                    if (value < base - 1.0e-9_dp) minimal = .false.
                end do
            end do
            call expect(minimal, "no local perturbation improves on the optimum of "// &
                fortbo_benchmark_name(kinds(k)), failures)
            deallocate (point, trial)
        end do
    end subroutine check_local_sweep_around_optimum

    subroutine check_gradients_against_differences(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:), trial(:), gradient(:)
        real(dp) :: plus, minus, numeric, worst
        real(dp), parameter :: step = 1.0e-6_dp
        integer :: kinds(7), k, j, trial_index

        kinds = [FORTBO_BENCH_BRANIN, FORTBO_BENCH_HARTMANN3, FORTBO_BENCH_HARTMANN6, &
            FORTBO_BENCH_ACKLEY, FORTBO_BENCH_ROSENBROCK, FORTBO_BENCH_LEVY, &
            FORTBO_BENCH_SPHERE]

        do k = 1, size(kinds)
            call make(kinds(k), 4, benchmark)
            allocate (point(benchmark%dimension), trial(benchmark%dimension))
            allocate (gradient(benchmark%dimension))
            worst = 0.0_dp
            do trial_index = 1, 5
                do j = 1, benchmark%dimension
                    point(j) = 0.35_dp + 0.11_dp*real(j, dp) &
                        + 0.07_dp*real(trial_index, dp)
                end do
                call benchmark%gradient(point, gradient, status)
                if (status%code /= FORTNUM_OK) cycle
                do j = 1, benchmark%dimension
                    trial = point
                    trial(j) = point(j) + step
                    call benchmark%value(trial, plus, status)
                    trial(j) = point(j) - step
                    call benchmark%value(trial, minus, status)
                    numeric = (plus - minus)/(2.0_dp*step)
                    worst = max(worst, abs(gradient(j) - numeric) &
                        /max(1.0_dp, abs(numeric)))
                end do
            end do
            call expect(worst < 1.0e-5_dp, &
                "the analytic gradient matches differences for "// &
                fortbo_benchmark_name(kinds(k)), failures)
            deallocate (point, trial, gradient)
        end do
    end subroutine check_gradients_against_differences

    !! Every one of these optima is interior to its declared box, so the
    !! gradient must vanish there.
    subroutine check_gradient_vanishes_at_optimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:), gradient(:)
        integer :: kinds(6), k

        kinds = [FORTBO_BENCH_BRANIN, FORTBO_BENCH_HARTMANN3, FORTBO_BENCH_HARTMANN6, &
            FORTBO_BENCH_ROSENBROCK, FORTBO_BENCH_LEVY, FORTBO_BENCH_SPHERE]

        do k = 1, size(kinds)
            call make(kinds(k), 4, benchmark)
            allocate (point(benchmark%dimension), gradient(benchmark%dimension))
            call benchmark%optimum(point, status)
            call benchmark%gradient(point, gradient, status)
            call expect(maxval(abs(gradient)) < 1.0e-3_dp, &
                "the gradient vanishes at the optimum of "// &
                fortbo_benchmark_name(kinds(k)), failures)
            deallocate (point, gradient)
        end do
    end subroutine check_gradient_vanishes_at_optimum

    !! Ackley's optimum is the origin, which is exactly where the first term's
    !! derivative has a removable singularity. A naive implementation returns
    !! NaN here and the whole high-dimensional benchmark silently dies.
    subroutine check_ackley_origin(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp) :: point(200), gradient(200), value

        benchmark%kind = FORTBO_BENCH_ACKLEY
        benchmark%dimension = 200
        point = 0.0_dp

        call benchmark%value(point, value, status)
        call expect(status%code == FORTNUM_OK, "Ackley evaluates at the origin", &
            failures)
        call expect(abs(value) < 1.0e-12_dp, "Ackley is zero at the origin", failures)

        call benchmark%gradient(point, gradient, status)
        call expect(status%code == FORTNUM_OK, "the Ackley gradient exists at the origin", &
            failures)
        call expect(all(gradient == gradient), &
            "the Ackley gradient is not NaN at the origin", failures)
        call expect(maxval(abs(gradient)) < 1.0e-12_dp, &
            "the Ackley gradient vanishes at the origin", failures)

        ! And two hundred dimensions must still evaluate finitely away from it.
        point = 0.3_dp
        call benchmark%value(point, value, status)
        call expect(value == value .and. abs(value) < huge(1.0_dp), &
            "Ackley stays finite at two hundred dimensions", failures)
    end subroutine check_ackley_origin

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp) :: point(3), value

        benchmark%kind = FORTBO_BENCH_BRANIN
        benchmark%dimension = 3
        call expect(.not. benchmark%is_valid(), &
            "a fixed-dimension function rejects the wrong dimension", failures)
        call benchmark%value(point, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an invalid configuration refuses to evaluate", failures)

        benchmark%kind = FORTBO_BENCH_ROSENBROCK
        benchmark%dimension = 1
        call expect(.not. benchmark%is_valid(), &
            "Rosenbrock rejects a single dimension", failures)

        benchmark%kind = FORTBO_BENCH_SPHERE
        benchmark%dimension = 3
        benchmark%noise = -1.0_dp
        call expect(.not. benchmark%is_valid(), &
            "a negative noise level is rejected", failures)
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

end program test_benchmarks
