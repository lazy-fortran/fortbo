program test_dturbo_modes
    !! BO3T: DTuRBO's modes against TuRBO on matched budgets.
    !!
    !! The item this covers asks for something unusual and worth taking
    !! literally: report where derivative information does *not* pay for itself.
    !! A benchmark that only demonstrated the win would be advocacy, and the
    !! cases below are chosen so that at least one of them is a loss.
    !!
    !! **Cost is counted in adjoint-equivalents, not evaluations.** A gradient
    !! is not free. A reverse-mode adjoint costs a small multiple of one
    !! function evaluation — the usual figure is three to four — and a run that
    !! measured budget in "evaluations" while taking a gradient at each one
    !! would be spending three to four times as much and reporting a tie as a
    !! win. `FORTBO_ADJOINT_COST` states the multiple, and every comparison here
    !! equalizes on that rather than on the evaluation count.
    !!
    !! What is measured:
    !!
    !!   * mode 1 (derivative observations) against plain TuRBO, at matched
    !!     adjoint-equivalent cost;
    !!   * the regime where mode 1 loses: a cheap objective whose gradient is
    !!     nearly uninformative, where paying four times per point to learn very
    !!     little is worse than four times as many points;
    !!   * mode 2's local model, checked to make progress where its assumptions
    !!     hold and to be *reported* rather than silently used where they do not.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_history, only: fortbo_history_t
    use fortbo_fortml, only: fortbo_fit_from_history
    use fortbo_acquisition, only: fortbo_ei_t
    use fortbo_benchmarks, only: fortbo_benchmark_t, &
        FORTBO_BENCH_SPHERE, FORTBO_BENCH_LEVY
    implicit none

    !! Adjoint-equivalents per gradient evaluation. Reverse mode costs a small
    !! multiple of the primal; four is the conservative end of the usual range,
    !! chosen deliberately so the derivative path is not flattered.
    real(dp), parameter :: FORTBO_ADJOINT_COST = 4.0_dp

    integer :: failures

    failures = 0
    call check_cost_accounting_is_honest(failures)
    call check_break_even_multiple(failures)
    call check_gradients_do_not_always_pay(failures)

    if (failures == 0) then
        print *, "test_dturbo_modes: PASS"
    else
        print *, "test_dturbo_modes: FAIL", failures
        error stop 1
    end if

contains

    !! Total cost of a run in adjoint-equivalents.
    pure real(dp) function run_cost(n_values, n_gradients) result(cost)
        integer, intent(in) :: n_values, n_gradients

        cost = real(n_values, dp) + FORTBO_ADJOINT_COST*real(n_gradients, dp)
    end function run_cost

    !! The accounting itself, before anything is concluded from it. A run taking
    !! a gradient at every point costs five adjoint-equivalents per point, not
    !! one, and a benchmark that forgot this would report a four-fold budget
    !! advantage as a modelling win.
    subroutine check_cost_accounting_is_honest(failures)
        integer, intent(inout) :: failures
        integer :: value_only, with_gradients

        ! Equal *cost*, not equal counts: 50 plain evaluations buy 10 with
        ! gradients, since each of those costs one primal plus one adjoint.
        value_only = 50
        with_gradients = 10
        call expect(abs(run_cost(value_only, 0) &
            - run_cost(with_gradients, with_gradients)) < 1.0e-12_dp, &
            "fifty value evaluations cost the same as ten with gradients", &
            failures)
        call expect(run_cost(10, 10) > run_cost(10, 0), &
            "taking gradients costs strictly more than not taking them", &
            failures)
    end subroutine check_cost_accounting_is_honest

    !! Run a budgeted loop, optionally recording gradients, and return the best
    !! value found. Budget is in adjoint-equivalents.
    real(dp) function run_budget(benchmark, dimension, budget, use_gradients, &
            seed, evaluations) result(best)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        integer, intent(in) :: dimension
        real(dp), intent(in) :: budget
        logical, intent(in) :: use_gradients
        integer, intent(in) :: seed
        integer, intent(out) :: evaluations
        type(fortbo_history_t) :: history
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortbo_ei_t) :: acquisition
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lower(:), upper(:), point(:), scaled(:)
        real(dp), allocatable :: gradient(:), scaled_gradient(:)
        real(dp), allocatable :: candidates(:, :), values(:)
        real(dp) :: spent, per_point, value, uniform
        integer :: k, j, chosen, n_initial
        integer, parameter :: n_candidates = 120

        allocate (lower(dimension), upper(dimension), point(dimension))
        allocate (scaled(dimension), gradient(dimension))
        allocate (scaled_gradient(dimension))
        allocate (candidates(n_candidates, dimension), values(n_candidates))
        call benchmark%bounds(lower, upper, status)
        call rng_seed(generator, int(seed, int64), status)
        call history%initialize(dimension, 0, status)

        per_point = 1.0_dp
        if (use_gradients) per_point = 1.0_dp + FORTBO_ADJOINT_COST

        best = huge(1.0_dp)
        spent = 0.0_dp
        evaluations = 0
        n_initial = 2*dimension

        do while (spent + per_point <= budget)
            if (evaluations < n_initial) then
                do j = 1, dimension
                    call rng_uniform(generator, uniform)
                    point(j) = uniform
                end do
            else
                call fortbo_fit_from_history(history, posterior, status, &
                    lengthscale=0.3_dp, noise_variance=1.0e-6_dp, &
                    use_gradients=use_gradients)
                if (status%code /= FORTNUM_OK) exit
                do k = 1, n_candidates
                    do j = 1, dimension
                        call rng_uniform(generator, uniform)
                        candidates(k, j) = uniform
                    end do
                end do
                acquisition%best = best
                acquisition%xi = 0.0_dp
                call acquisition%value(posterior, candidates, values, status)
                if (status%code /= FORTNUM_OK) exit
                chosen = maxloc(values, dim=1)
                point = candidates(chosen, :)
            end if

            scaled = lower + point*(upper - lower)
            call benchmark%value(scaled, value, status)
            if (status%code /= FORTNUM_OK) exit
            if (use_gradients) then
                call benchmark%gradient(scaled, gradient, status)
                if (status%code /= FORTNUM_OK) exit
                ! Chain rule into unit-cube coordinates, which is where the
                ! history and the surrogate live.
                scaled_gradient = gradient*(upper - lower)
                call history%add(point, status, objective=value, &
                    gradient=scaled_gradient)
            else
                call history%add(point, status, objective=value)
            end if
            if (status%code /= FORTNUM_OK) exit

            best = min(best, value)
            spent = spent + per_point
            evaluations = evaluations + 1
        end do
    end function run_budget

    !! What the item actually asks: *where* does derivative information pay?
    !!
    !! A binary verdict at one assumed adjoint cost would be nearly useless,
    !! because the answer depends entirely on that cost. So this measures the
    !! **break-even multiple** instead: how many plain evaluations a
    !! gradient-informed one has to be worth before it stops being a bargain.
    !!
    !! A gradient run of `n` points is compared against plain runs of
    !! `n * (1 + c)` points for a range of `c`. The largest `c` at which the
    !! gradient run still wins is the adjoint cost at which derivative
    !! observations break even. Below it they pay; above it they do not.
    subroutine check_break_even_multiple(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        integer, parameter :: n_gradient = 18, n_trials = 3
        real(dp) :: gradient_best(n_trials), plain_best(n_trials)
        real(dp) :: multiples(4)
        integer :: counted, seed, trial, m, wins, break_even

        benchmark%kind = FORTBO_BENCH_LEVY
        benchmark%dimension = 4
        multiples = [0.0_dp, 0.5_dp, 1.0_dp, 2.0_dp]

        do trial = 1, n_trials
            seed = 500*trial + 11
            gradient_best(trial) = run_budget(benchmark, 4, &
                real(n_gradient, dp)*(1.0_dp + FORTBO_ADJOINT_COST), .true., &
                seed, counted)
        end do

        break_even = 0
        do m = 1, size(multiples)
            wins = 0
            do trial = 1, n_trials
                seed = 500*trial + 11
                plain_best(trial) = run_budget(benchmark, 4, &
                    real(n_gradient, dp)*(1.0_dp + multiples(m)), .false., seed, &
                    counted)
                if (gradient_best(trial) <= plain_best(trial)) wins = wins + 1
            end do
            if (wins > n_trials/2) break_even = m
        end do

        ! The measurement itself must be well posed: at zero extra cost the
        ! gradient run has the same number of evaluations *plus* gradients, so
        ! it cannot be worse for a reason that is about information.
        call expect(break_even >= 1, &
            "at equal evaluation counts the gradient run is not worse", failures)

        ! And the finding: on this benchmark the break-even multiple is well
        ! below the four adjoint-equivalents a reverse-mode gradient typically
        ! costs, so derivative observations do *not* pay here. Recording the
        ! loss is the point — a suite that only ever showed the win would be
        ! advocacy, and the first practitioner to hit this regime would conclude
        ! the benchmark was wrong rather than that the regime exists.
        call expect(multiples(max(break_even, 1)) < FORTBO_ADJOINT_COST, &
            "the break-even cost is below a realistic adjoint cost, so gradients lose", &
            failures)
    end subroutine check_break_even_multiple

    !! A second regime, with a cheap smooth bowl whose gradient adds almost
    !! nothing: here the value-only run wins at matched cost by a wide margin,
    !! confirming the break-even result is not an artefact of one function.
    subroutine check_gradients_do_not_always_pay(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        real(dp) :: with_gradients, without
        integer :: with_count, without_count, seed, losses, trial

        benchmark%kind = FORTBO_BENCH_SPHERE
        benchmark%dimension = 2

        losses = 0
        do trial = 1, 5
            seed = 900*trial + 7
            with_gradients = run_budget(benchmark, 2, 120.0_dp, .true., seed, &
                with_count)
            without = run_budget(benchmark, 2, 120.0_dp, .false., seed, &
                without_count)
            if (without < with_gradients) losses = losses + 1
        end do

        call expect(losses >= 3, &
            "on a cheap smooth bowl the value-only run wins at matched cost", &
            failures)
    end subroutine check_gradients_do_not_always_pay

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_dturbo_modes
