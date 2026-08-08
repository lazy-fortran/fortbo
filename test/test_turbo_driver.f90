program test_turbo_driver
    !! BO3T: TuRBO-1 and TuRBO-`m` as a running policy.
    !!
    !! The components are tested elsewhere; what is tested here is that they
    !! compose into a method. The oracles are behavioral and independent of the
    !! implementation:
    !!
    !!   * TuRBO must beat uniform random search on a benchmark, at a matched
    !!     evaluation budget and over several seeds. This is the only claim that
    !!     matters and the only one that cannot be faked by bookkeeping;
    !!   * the implicit bandit must actually reallocate. Given two regions, one
    !!     seeded on a good basin and one on a bad one, the batch must drift
    !!     toward the good region *without* any explicit arm selection. Measured
    !!     as a share of proposals, not asserted;
    !!   * a collapsed region must restart, and restart with an empty history,
    !!     since restarting on the old data rebuilds the model that collapsed;
    !!   * ask/tell must be replayable: the same seed and the same answers give
    !!     the same proposals.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_turbo_driver, only: fortbo_turbo_driver_t, fortbo_turbo_config_t
    use fortbo_benchmarks, only: fortbo_benchmark_t, FORTBO_BENCH_BRANIN
    implicit none

    integer :: failures

    failures = 0
    call check_turbo_one_beats_random(failures)
    call check_bandit_reallocates(failures)
    call check_restart_clears_history(failures)
    call check_replayable(failures)
    call check_frozen_candidate_pool(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_turbo_driver: PASS"
    else
        print *, "test_turbo_driver: FAIL", failures
        error stop 1
    end if

contains

    !! Evaluate the benchmark at a point given in the unit cube.
    real(dp) function evaluate(benchmark, unit_point) result(value)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        real(dp), intent(in) :: unit_point(:)
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lower(:), upper(:), scaled(:)

        allocate (lower(size(unit_point)), upper(size(unit_point)))
        allocate (scaled(size(unit_point)))
        call benchmark%bounds(lower, upper, status)
        scaled = lower + unit_point*(upper - lower)
        call benchmark%value(scaled, value, status)
    end function evaluate

    !! Run TuRBO-`m` for a fixed budget and return the best value found.
    real(dp) function run_turbo(benchmark, n_inputs, config, seed, budget, &
            driver) result(best)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        integer, intent(in) :: n_inputs
        type(fortbo_turbo_config_t), intent(in) :: config
        integer, intent(in) :: seed
        integer, intent(in) :: budget
        type(fortbo_turbo_driver_t), intent(out) :: driver
        type(fortnum_status_t) :: status
        real(dp), allocatable :: points(:, :), values(:)
        integer, allocatable :: regions(:)
        integer :: used, i

        call driver%initialize(n_inputs, config, seed, status)
        allocate (points(config%batch_size, n_inputs))
        allocate (regions(config%batch_size), values(config%batch_size))
        used = 0
        do while (used < budget)
            call driver%ask(points, regions, status)
            if (status%code /= FORTNUM_OK) exit
            do i = 1, config%batch_size
                values(i) = evaluate(benchmark, points(i, :))
            end do
            call driver%tell(points, regions, values, status)
            if (status%code /= FORTNUM_OK) exit
            used = used + config%batch_size
        end do
        best = driver%best_value
    end function run_turbo

    real(dp) function run_random(benchmark, n_inputs, seed, budget) result(best)
        type(fortbo_benchmark_t), intent(in) :: benchmark
        integer, intent(in) :: n_inputs
        integer, intent(in) :: seed
        integer, intent(in) :: budget
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:)
        real(dp) :: uniform, value
        integer :: k, j

        call rng_seed(generator, int(seed, int64), status)
        allocate (point(n_inputs))
        best = huge(1.0_dp)
        do k = 1, budget
            do j = 1, n_inputs
                call rng_uniform(generator, uniform)
                point(j) = uniform
            end do
            value = evaluate(benchmark, point)
            best = min(best, value)
        end do
    end function run_random

    !! The claim that matters. Random search is a genuinely hard baseline at
    !! small budgets, so this is checked across seeds and on the median rather
    !! than on one lucky run.
    subroutine check_turbo_one_beats_random(failures)
        integer, intent(inout) :: failures
        type(fortbo_benchmark_t) :: benchmark
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: driver
        type(fortnum_status_t) :: status
        real(dp) :: turbo_best(5), random_best(5)
        integer, parameter :: budget = 60
        integer :: seed, wins, k

        benchmark%kind = FORTBO_BENCH_BRANIN
        benchmark%dimension = 2
        call expect(benchmark%is_valid(), "the benchmark is valid", failures)

        config%n_regions = 1
        config%batch_size = 1
        config%lengthscale = 0.25_dp

        wins = 0
        do k = 1, 5
            seed = 1000*k + 7
            turbo_best(k) = run_turbo(benchmark, 2, config, seed, budget, driver)
            random_best(k) = run_random(benchmark, 2, seed, budget)
            if (turbo_best(k) < random_best(k)) wins = wins + 1
        end do

        call expect(wins >= 4, &
            "TuRBO-1 beats random search on most seeds at a matched budget", &
            failures)
        call expect(sum(turbo_best) < sum(random_best), &
            "TuRBO-1 beats random search in aggregate", failures)
    end subroutine check_turbo_one_beats_random

    !! Two regions given deliberately unequal evidence: one seeded on a design
    !! clustered around the bowl's minimum, the other on a design far away in
    !! the opposite corner. The bandit is never told which is which.
    !!
    !! An earlier version of this test let both regions explore freely and
    !! compared whichever happened to hold better values. That measured almost
    !! nothing — both regions converge on the same bowl, the difference between
    !! them is noise, and the result sat at chance whether the bandit worked or
    !! not. Constructing the asymmetry is what makes the measurement mean
    !! something: with one region's posterior genuinely lower everywhere, a
    !! working bandit must send it nearly the whole batch, and a driver that
    !! allocated uniformly or by region index cannot.
    subroutine check_bandit_reallocates(failures)
        integer, intent(inout) :: failures
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: driver
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 2), values(2), seeded(1, 2), seed_value(1)
        integer :: regions(2), seed_region(1)
        integer :: proposals(2), i, step, k
        real(dp) :: offset

        config%n_regions = 2
        config%batch_size = 2
        config%n_initial = 5
        config%lengthscale = 0.3_dp

        call driver%initialize(2, config, 4242, status)
        call expect(status%code == FORTNUM_OK, "the two-region driver starts", &
            failures)

        ! Region 1 gets a design hugging the minimum at (0.2, 0.8); region 2
        ! gets one in the far corner. Both designs are supplied directly, so
        ! the asymmetry is exact rather than left to the seed.
        do k = 1, config%n_initial
            offset = 0.02_dp*real(k, dp)
            seeded(1, :) = [0.20_dp + offset, 0.80_dp - offset]
            seed_value(1) = bowl(seeded(1, :))
            seed_region(1) = 1
            call driver%tell(seeded, seed_region, seed_value, status)
            call expect(status%code == FORTNUM_OK, "the good design is recorded", &
                failures)

            seeded(1, :) = [0.90_dp - offset, 0.10_dp + offset]
            seed_value(1) = bowl(seeded(1, :))
            seed_region(1) = 2
            call driver%tell(seeded, seed_region, seed_value, status)
            call expect(status%code == FORTNUM_OK, "the poor design is recorded", &
                failures)
        end do

        proposals = 0
        do step = 1, 25
            call driver%ask(points, regions, status)
            if (status%code /= FORTNUM_OK) exit
            do i = 1, 2
                values(i) = bowl(points(i, :))
                proposals(regions(i)) = proposals(regions(i)) + 1
            end do
            call driver%tell(points, regions, values, status)
            if (status%code /= FORTNUM_OK) exit
        end do

        call expect(sum(proposals) > 0, "the bandit made proposals", failures)
        call expect(proposals(1) > proposals(2), &
            "the implicit bandit favors the region holding better values", &
            failures)
        call expect(proposals(1) > 3*proposals(2), &
            "the favored region takes the large majority of the batch", failures)
    end subroutine check_bandit_reallocates

    !! A simple bowl with a single minimum, so "better region" is unambiguous.
    pure real(dp) function bowl(x) result(value)
        real(dp), intent(in) :: x(:)

        value = (x(1) - 0.2_dp)**2 + (x(2) - 0.8_dp)**2
    end function bowl

    !! Drive a region to collapse by never improving, and confirm it restarts
    !! with no data. Restarting on the old history would rebuild the model that
    !! just collapsed and collapse again immediately.
    subroutine check_restart_clears_history(failures)
        integer, intent(inout) :: failures
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: driver
        type(fortnum_status_t) :: status
        real(dp), allocatable :: points(:, :), values(:)
        integer, allocatable :: regions(:)
        integer :: step
        logical :: saw_restart

        config%n_regions = 1
        config%batch_size = 1
        config%n_initial = 4

        call driver%initialize(3, config, 99, status)
        allocate (points(1, 3), regions(1), values(1))

        saw_restart = .false.
        do step = 1, 400
            call driver%ask(points, regions, status)
            if (status%code /= FORTNUM_OK) exit
            ! A constant objective never improves, so every batch is a failure
            ! and the region must eventually shrink past its floor.
            values(1) = 1.0_dp
            call driver%tell(points, regions, values, status)
            if (status%code /= FORTNUM_OK) exit
            if (driver%restarts > 0) then
                saw_restart = .true.
                exit
            end if
        end do

        call expect(saw_restart, "a region that never improves is restarted", &
            failures)
        call expect(driver%best_value == 1.0_dp, &
            "the best value survives the restart", failures)
        call expect(driver%histories(1)%count == 0, &
            "a restarted region begins with an empty history", failures)
        ! A restarted region is deliberately left *unplaced*: it draws a fresh
        ! initial design and is re-centered on the best point of that design.
        ! Re-centering on an arbitrary point would throw away the design the
        ! run is about to pay for anyway.
        call expect(driver%active_regions() == 0, &
            "a restarted region is unplaced until its new design completes", &
            failures)

        ! It must come back, though. Keep running and confirm it is placed
        ! again rather than left dead.
        do step = 1, 40
            call driver%ask(points, regions, status)
            if (status%code /= FORTNUM_OK) exit
            values(1) = bowl(points(1, :))
            call driver%tell(points, regions, values, status)
            if (status%code /= FORTNUM_OK) exit
            if (driver%active_regions() == 1) exit
        end do
        call expect(driver%active_regions() == 1, &
            "the restarted region is placed again once its design completes", &
            failures)
    end subroutine check_restart_clears_history

    !! Same seed, same answers, same proposals. Without this a run cannot be
    !! reproduced from its record.
    subroutine check_replayable(failures)
        integer, intent(inout) :: failures
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: first, second
        type(fortnum_status_t) :: status
        real(dp) :: points_a(2, 2), points_b(2, 2), values(2)
        integer :: regions_a(2), regions_b(2)
        integer :: step
        logical :: identical

        config%n_regions = 2
        config%batch_size = 2
        config%n_initial = 4

        call first%initialize(2, config, 271828, status)
        call second%initialize(2, config, 271828, status)

        identical = .true.
        do step = 1, 20
            call first%ask(points_a, regions_a, status)
            call second%ask(points_b, regions_b, status)
            if (maxval(abs(points_a - points_b)) /= 0.0_dp) identical = .false.
            if (any(regions_a /= regions_b)) identical = .false.
            values = [bowl(points_a(1, :)), bowl(points_a(2, :))]
            call first%tell(points_a, regions_a, values, status)
            call second%tell(points_b, regions_b, values, status)
        end do
        call expect(identical, "the same seed replays the same proposals", &
            failures)

        ! A different seed must not replay the same run, or the seed is not
        ! reaching the generator.
        call second%initialize(2, config, 314159, status)
        call second%ask(points_b, regions_b, status)
        call first%initialize(2, config, 271828, status)
        call first%ask(points_a, regions_a, status)
        call expect(maxval(abs(points_a - points_b)) > 0.0_dp, &
            "a different seed gives a different run", failures)
    end subroutine check_replayable

    !! A caller-supplied candidate pool is the cross-language replay escape
    !! hatch: the posterior and selection still run, but the candidate bits
    !! come from the frozen pool rather than from FortNum's Sobol stream.
    subroutine check_frozen_candidate_pool(failures)
        integer, intent(inout) :: failures
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: driver
        type(fortnum_status_t) :: status
        real(dp) :: point(1, 2), value(1), pool(4, 2)
        integer :: region(1), i
        logical :: found

        config%n_regions = 1
        config%batch_size = 1
        config%n_initial = 1
        allocate (config%frozen_candidates(4, 2))
        pool(1, :) = [0.45_dp, 0.45_dp]
        pool(2, :) = [0.45_dp, 0.55_dp]
        pool(3, :) = [0.55_dp, 0.45_dp]
        pool(4, :) = [0.55_dp, 0.55_dp]
        config%frozen_candidates = pool

        call driver%initialize(2, config, 7, status)
        call expect(status%code == FORTNUM_OK, &
            "a frozen candidate pool is accepted", failures)
        point(1, :) = [0.5_dp, 0.5_dp]
        region = 1
        value = 0.0_dp
        call driver%tell(point, region, value, status)
        call expect(status%code == FORTNUM_OK, &
            "the frozen-pool initial observation is recorded", failures)
        call driver%ask(point, region, status)
        call expect(status%code == FORTNUM_OK, &
            "selection from a frozen pool succeeds", failures)

        found = .false.
        do i = 1, size(pool, 1)
            if (maxval(abs(point(1, :) - pool(i, :))) == 0.0_dp) found = .true.
        end do
        call expect(found, "selection returns one of the frozen candidates", failures)
    end subroutine check_frozen_candidate_pool

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_turbo_config_t) :: config
        type(fortbo_turbo_driver_t) :: driver, fresh
        type(fortnum_status_t) :: status
        real(dp) :: points(1, 2), wide(3, 2), values(1)
        integer :: regions(1), wide_regions(3)

        config%n_regions = 1
        config%batch_size = 1

        call driver%initialize(0, config, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero-width problem is refused", failures)

        config%n_regions = 0
        call driver%initialize(2, config, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero-region configuration is refused", failures)

        call fresh%ask(points, regions, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "asking an uninitialized driver is refused", failures)

        config%n_regions = 1
        call driver%initialize(2, config, 1, status)
        wide_regions = 1
        call driver%ask(wide, wide_regions, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-sized batch is refused", failures)

        call driver%ask(points, regions, status)
        values = 0.0_dp
        regions = 7
        call driver%tell(points, regions, values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an out-of-range region index is refused", failures)
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

end program test_turbo_driver
