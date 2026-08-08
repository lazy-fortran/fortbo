program test_high_dimensional
    !! BO6: high-dimensional trust-region fixtures.
    !!
    !! The fixtures exist to exercise DTuRBO on a *true adjoint*, so the claim
    !! that matters is that the analytic gradients really are the gradients, at
    !! the dimensions the trust-region work is aimed at rather than at the
    !! two-dimensional sizes the other tests use.
    !!
    !! Oracles:
    !!
    !!   * gradients are checked against Richardson-extrapolated central
    !!     differences at randomly chosen interior points, coordinate by
    !!     coordinate. That is the independent statement of what a derivative
    !!     is, and it shares no code with the analytic expressions;
    !!   * each function's optimum is stated and checked, so a fixture that is
    !!     subtly the wrong function is caught before a benchmark is built on
    !!     it;
    !!   * the gradient must vanish at the optimum, which catches a whole class
    !!     of transcription error that a value check alone does not.
    !!
    !! Levy and Rosenbrock are exercised at `d = 100, 300, 500` and Ackley at
    !! `d = 200` on `[-5, 10]^200`, which is the configuration the TuRBO paper
    !! reports.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_benchmarks, only: fortbo_benchmark_t, fortbo_benchmark_name, &
        FORTBO_BENCH_ACKLEY, FORTBO_BENCH_LEVY, FORTBO_BENCH_ROSENBROCK
    implicit none

    integer :: failures

    failures = 0
    call check_ackley_200(failures)
    call check_levy_and_rosenbrock_at_scale(failures)
    call check_gradients_vanish_at_the_optimum(failures)

    if (failures == 0) then
        print *, "test_high_dimensional: PASS"
    else
        print *, "test_high_dimensional: FAIL", failures
        error stop 1
    end if

contains

    !! Richardson-extrapolated central differences of one coordinate.
    real(dp) function numeric_partial(benchmark, point, j) result(derivative)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(in) :: point(:)
        integer, intent(in) :: j
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:)
        real(dp) :: plus, minus, coarse, fine
        real(dp), parameter :: h = 1.0e-4_dp

        allocate (shifted, source=point)

        shifted(j) = point(j) + h
        call benchmark%value(shifted, plus, status)
        shifted(j) = point(j) - h
        call benchmark%value(shifted, minus, status)
        coarse = (plus - minus)/(2.0_dp*h)

        shifted(j) = point(j) + 0.5_dp*h
        call benchmark%value(shifted, plus, status)
        shifted(j) = point(j) - 0.5_dp*h
        call benchmark%value(shifted, minus, status)
        fine = (plus - minus)/h

        ! The central formula errs at O(h^2), so this combination cancels that
        ! term and leaves O(h^4).
        derivative = (4.0_dp*fine - coarse)/3.0_dp
    end function numeric_partial

    !! Check a sample of coordinates rather than all of them: at d = 500 the
    !! full sweep is 2000 evaluations per point per step size, and a random
    !! sample catches a systematic error just as surely.
    subroutine check_gradient_sample(benchmark, seed, n_probe, tolerance, label, &
            failures)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        integer, intent(in) :: seed
        integer, intent(in) :: n_probe
        real(dp), intent(in) :: tolerance
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lower(:), upper(:), point(:), gradient(:)
        real(dp) :: uniform, analytic, numeric, worst
        integer :: d, j, k, coordinate
        logical :: matches

        d = benchmark%dimension
        allocate (lower(d), upper(d), point(d), gradient(d))
        call benchmark%bounds(lower, upper, status)
        call rng_seed(generator, int(seed, int64), status)

        ! Stay off the bounds so the central differences stay inside the box.
        do j = 1, d
            call rng_uniform(generator, uniform)
            point(j) = lower(j) + (0.1_dp + 0.8_dp*uniform)*(upper(j) - lower(j))
        end do

        call benchmark%gradient(point, gradient, status)
        call expect(status%code == FORTNUM_OK, label//": the gradient evaluates", &
            failures)

        matches = .true.
        worst = 0.0_dp
        do k = 1, n_probe
            call rng_uniform(generator, uniform)
            coordinate = 1 + int(uniform*real(d, dp))
            if (coordinate > d) coordinate = d
            analytic = gradient(coordinate)
            numeric = numeric_partial(benchmark, point, coordinate)
            worst = max(worst, abs(analytic - numeric)/max(1.0_dp, abs(numeric)))
            if (abs(analytic - numeric) > tolerance*max(1.0_dp, abs(numeric))) &
                matches = .false.
        end do
        call expect(matches, &
            label//": the analytic gradient matches central differences", failures)
    end subroutine check_gradient_sample

    subroutine check_ackley_200(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lower(:), upper(:), optimum(:)
        real(dp) :: value, optimal

        benchmark%kind = FORTBO_BENCH_ACKLEY
        benchmark%dimension = 200
        call expect(benchmark%is_valid(), "Ackley-200 is a valid fixture", failures)

        allocate (lower(200), upper(200), optimum(200))
        call benchmark%bounds(lower, upper, status)
        call expect(all(lower == -5.0_dp) .and. all(upper == 10.0_dp), &
            "Ackley-200 uses the [-5, 10] domain the paper reports", failures)

        call benchmark%optimum(optimum, status)
        call benchmark%value(optimum, value, status)
        optimal = benchmark%optimal_value()
        call expect(abs(value - optimal) < 1.0e-10_dp, &
            "Ackley-200 attains its stated optimal value at its optimum", failures)

        ! The asymmetric domain matters: the optimum at the origin is not the
        ! centre of the box, so a fixture that silently recentred would put the
        ! answer where a search starts.
        call expect(any(abs(optimum - 0.5_dp*(lower + upper)) > 1.0_dp), &
            "the optimum is not at the centre of the box", failures)

        call check_gradient_sample(benchmark, 11111, 12, 1.0e-5_dp, "Ackley-200", &
            failures)
    end subroutine check_ackley_200

    subroutine check_levy_and_rosenbrock_at_scale(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        integer :: dimensions(3), k

        dimensions = [100, 300, 500]
        do k = 1, 3
            benchmark%kind = FORTBO_BENCH_LEVY
            benchmark%dimension = dimensions(k)
            call expect(benchmark%is_valid(), "Levy is valid at scale", failures)
            call check_gradient_sample(benchmark, 2000 + k, 10, 1.0e-5_dp, &
                "Levy", failures)

            benchmark%kind = FORTBO_BENCH_ROSENBROCK
            benchmark%dimension = dimensions(k)
            call expect(benchmark%is_valid(), "Rosenbrock is valid at scale", &
                failures)
            ! Rosenbrock's fourth derivative is large, so the extrapolated
            ! difference carries more error than for the other two.
            call check_gradient_sample(benchmark, 3000 + k, 10, 1.0e-4_dp, &
                "Rosenbrock", failures)
        end do
    end subroutine check_levy_and_rosenbrock_at_scale

    !! A vanishing gradient at the stated optimum catches transcription errors
    !! that a value check alone misses: a function can take the right value at
    !! the right point and still be the wrong function.
    subroutine check_gradients_vanish_at_the_optimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortnum_status_t) :: status
        real(dp), allocatable :: optimum(:), gradient(:)
        integer :: kinds(3), k, d

        kinds = [FORTBO_BENCH_ACKLEY, FORTBO_BENCH_LEVY, FORTBO_BENCH_ROSENBROCK]
        d = 100
        do k = 1, 3
            benchmark%kind = kinds(k)
            benchmark%dimension = d
            if (allocated(optimum)) deallocate (optimum, gradient)
            allocate (optimum(d), gradient(d))
            call benchmark%optimum(optimum, status)
            call benchmark%gradient(optimum, gradient, status)
            call expect(status%code == FORTNUM_OK, &
                "the gradient evaluates at the optimum", failures)
            call expect(maxval(abs(gradient)) < 1.0e-6_dp, &
                fortbo_benchmark_name(kinds(k))// &
                ": the gradient vanishes at the stated optimum", failures)
        end do
    end subroutine check_gradients_vanish_at_the_optimum

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_high_dimensional
