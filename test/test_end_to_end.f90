program test_end_to_end
    !! BO6: a complete optimization run over the assembled stack.
    !!
    !! Every component built so far is exercised together here: the search
    !! space, the observation history with gradient rows, the FortML surrogate
    !! chosen from the data, the FortSym-derived expected improvement, the
    !! TuRBO trust region and its candidate generator, and the benchmark's
    !! verified optimum.
    !!
    !! The oracle is the one that matters for a Bayesian optimizer and cannot
    !! be faked by a component test: **simple regret against a known optimum,
    !! compared with random search on an identical budget**. A method that
    !! models nothing still makes progress on a smooth function, so "regret went
    !! down" proves nothing on its own. Beating random search from the same
    !! initial design, with the same number of evaluations, is the claim.
    !!
    !! The derivative-informed variant is run as a separate row on the same
    !! budget, as the roadmap requires — not merged into the headline number.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortnum_sobol, only: sobol_t, sobol_initialize, sobol_next
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_history, only: fortbo_history_t
    use fortbo_acquisition, only: fortbo_ei_t
    use fortbo_trust_region, only: fortbo_trust_region_t
    use fortbo_turbo, only: fortbo_turbo_candidates
    use fortbo_fortml, only: fortbo_fit_from_history
    use fortbo_benchmarks, only: fortbo_benchmark_t, FORTBO_BENCH_BRANIN
    implicit none

    integer :: failures

    failures = 0
    call check_turbo_beats_random_search(failures)

    if (failures == 0) then
        print *, "test_end_to_end: PASS"
    else
        print *, "test_end_to_end: FAIL", failures
        error stop 1
    end if

contains

    !! Map a unit-cube point to the benchmark's natural box.
    pure subroutine to_domain(unit, lower, upper, point)
        real(dp), intent(in) :: unit(:), lower(:), upper(:)
        real(dp), intent(out) :: point(:)

        point = lower + unit*(upper - lower)
    end subroutine to_domain

    subroutine evaluate(benchmark, lower, upper, unit, value, gradient, status)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(in) :: lower(:), upper(:), unit(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: point(size(unit)), raw_gradient(size(unit))

        call to_domain(unit, lower, upper, point)
        call benchmark%value(point, value, status)
        if (status%code /= FORTNUM_OK) return
        call benchmark%gradient(point, raw_gradient, status)
        if (status%code /= FORTNUM_OK) return
        ! Chain rule for the affine map to the unit cube, so the recorded
        ! gradient is the gradient of what the surrogate actually sees.
        gradient = raw_gradient*(upper - lower)
    end subroutine evaluate

    subroutine check_turbo_beats_random_search(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        real(dp) :: lower(2), upper(2)
        real(dp) :: regret_plain, regret_informed, regret_random
        integer :: evaluations_plain, evaluations_random

        benchmark%kind = FORTBO_BENCH_BRANIN
        benchmark%dimension = 2
        call benchmark_bounds_of(benchmark, lower, upper)

        call run_turbo(benchmark, lower, upper, .false., regret_plain, &
            evaluations_plain)
        call run_turbo(benchmark, lower, upper, .true., regret_informed, &
            evaluations_plain)
        call run_random(benchmark, lower, upper, evaluations_plain, regret_random, &
            evaluations_random)

        call expect(evaluations_random == evaluations_plain, &
            "both methods spend the same evaluation budget", failures)
        call expect(regret_plain < regret_random, &
            "TuRBO with a GP beats random search at equal budget", failures)
        call expect(regret_plain < 1.0e-2_dp, &
            "TuRBO reaches small simple regret on Branin", failures)
        call expect(regret_informed < regret_random, &
            "the derivative-informed run also beats random search", failures)

        print *, "    simple regret  turbo(value-only):", regret_plain
        print *, "    simple regret  turbo(derivative):", regret_informed
        print *, "    simple regret  random search    :", regret_random
        print *, "    evaluations                     :", evaluations_plain
    end subroutine check_turbo_beats_random_search

    subroutine benchmark_bounds_of(benchmark, lower, upper)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(out) :: lower(:), upper(:)
        type(fortnum_status_t) :: status

        call benchmark%bounds(lower, upper, status)
    end subroutine benchmark_bounds_of

    !! One TuRBO run. The surrogate is refitted each iteration, the trust region
    !! adapts on the batch outcome, and candidates are scored by expected
    !! improvement — the sampling-search configuration, which is what a
    !! moments-only surrogate supports.
    subroutine run_turbo(benchmark, lower, upper, use_gradients, regret, evaluations)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(in) :: lower(:), upper(:)
        logical, intent(in) :: use_gradients
        real(dp), intent(out) :: regret
        integer, intent(out) :: evaluations
        type(fortbo_history_t) :: history
        type(fortbo_trust_region_t) :: region
        type(fortbo_ei_t) :: ei
        class(fortbo_posterior_t), allocatable :: posterior
        type(sobol_t) :: sequence
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_initial = 8
        integer, parameter :: n_iterations = 40
        integer, parameter :: n_candidates = 300
        real(dp) :: unit(2), value, gradient(2), lengthscales(2)
        real(dp) :: candidates(n_candidates, 2), scores(n_candidates)
        real(dp) :: incumbent(2), incumbent_value
        real(dp) :: batch_inputs(1, 2), batch_values(1)
        integer :: iteration, best, c

        call history%initialize(2, 0, status)
        call sobol_initialize(sequence, 2, status)
        call rng_seed(generator, int(12345, int64), status)
        lengthscales = [1.0_dp, 1.0_dp]

        ! Sobol initial design. The first Sobol point is the origin, which is a
        ! corner of the box; skipping it keeps the design interior.
        call sobol_next(sequence, unit, status)
        do c = 1, n_initial
            call sobol_next(sequence, unit, status)
            call evaluate(benchmark, lower, upper, unit, value, gradient, status)
            call history%add(unit, status, objective=value, gradient=gradient)
        end do
        evaluations = n_initial

        call region%initialize(2, 1, status)
        call history%incumbent(incumbent, incumbent_value, status)
        call region%restart(incumbent, incumbent_value, status)

        do iteration = 1, n_iterations
            call fortbo_fit_from_history(history, posterior, status, &
                lengthscale=0.25_dp, &
                noise_variance=1.0e-6_dp, &
                use_gradients=use_gradients)
            if (status%code /= FORTNUM_OK) exit

            if (.not. region%active) then
                call history%incumbent(incumbent, incumbent_value, status)
                call region%restart(incumbent, incumbent_value, status)
            end if

            call fortbo_turbo_candidates(region, lengthscales, generator, candidates, &
                status)
            if (status%code /= FORTNUM_OK) exit

            ei%best = region%center_value
            call ei%value(posterior, candidates, scores, status)
            if (status%code /= FORTNUM_OK) exit

            best = 1
            do c = 2, n_candidates
                if (scores(c) > scores(best)) best = c
            end do

            unit = candidates(best, :)
            call evaluate(benchmark, lower, upper, unit, value, gradient, status)
            call history%add(unit, status, objective=value, gradient=gradient)
            evaluations = evaluations + 1

            batch_inputs(1, :) = unit
            batch_values(1) = value
            call region%observe_batch(batch_inputs, batch_values, status)
        end do

        call history%incumbent(incumbent, incumbent_value, status)
        regret = incumbent_value - benchmark%optimal_value()
    end subroutine run_turbo

    !! Random search on the same budget, from the same Sobol initial design, so
    !! the comparison isolates the modelling and not the initialization.
    subroutine run_random(benchmark, lower, upper, budget, regret, evaluations)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(in) :: lower(:), upper(:)
        integer, intent(in) :: budget
        real(dp), intent(out) :: regret
        integer, intent(out) :: evaluations
        type(fortbo_history_t) :: history
        type(sobol_t) :: sequence
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_initial = 8
        real(dp) :: unit(2), value, gradient(2)
        real(dp) :: incumbent(2), incumbent_value
        integer :: k, j

        call history%initialize(2, 0, status)
        call sobol_initialize(sequence, 2, status)
        call rng_seed(generator, int(12345, int64), status)

        call sobol_next(sequence, unit, status)
        do k = 1, n_initial
            call sobol_next(sequence, unit, status)
            call evaluate(benchmark, lower, upper, unit, value, gradient, status)
            call history%add(unit, status, objective=value)
        end do
        evaluations = n_initial

        do k = n_initial + 1, budget
            do j = 1, 2
                call rng_uniform(generator, unit(j))
            end do
            call evaluate(benchmark, lower, upper, unit, value, gradient, status)
            call history%add(unit, status, objective=value)
            evaluations = evaluations + 1
        end do

        call history%incumbent(incumbent, incumbent_value, status)
        regret = incumbent_value - benchmark%optimal_value()
    end subroutine run_random

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_end_to_end
