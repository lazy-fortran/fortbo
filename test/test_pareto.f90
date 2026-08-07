program test_pareto
    !! BO4: Pareto archives, hypervolume, and scalarization.
    !!
    !! Oracles:
    !!   * hypervolume is checked against Monte Carlo integration — sample the
    !!     box below the reference point, count the fraction dominated by the
    !!     front, multiply by the box volume. That shares no code with the
    !!     dimension sweep, and the tolerance comes from the binomial standard
    !!     error rather than being tuned;
    !!   * in two dimensions it is also checked against a closed-form staircase
    !!     area computed in the test;
    !!   * the archive is checked against a brute-force non-dominated scan over
    !!     everything inserted;
    !!   * monotonicity is checked as a property: adding a non-dominated point
    !!     must strictly increase hypervolume, and adding a dominated one must
    !!     leave it exactly unchanged. That is the property that makes
    !!     hypervolume improvement a sound acquisition.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_pareto, only: fortbo_pareto_archive_t, fortbo_dominates, &
        fortbo_hypervolume, fortbo_hypervolume_improvement, fortbo_weighted_sum, &
        fortbo_chebyshev
    implicit none

    integer :: failures

    failures = 0
    call check_dominance(failures)
    call check_archive_matches_brute_force(failures)
    call check_two_dimensional_area(failures)
    call check_against_monte_carlo(failures)
    call check_monotonicity(failures)
    call check_improvement(failures)
    call check_scalarizations(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_pareto: PASS"
    else
        print *, "test_pareto: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_dominance(failures)
        integer, intent(inout) :: failures

        call expect(fortbo_dominates([1.0_dp, 1.0_dp], [2.0_dp, 2.0_dp]), &
            "strictly better in both dominates", failures)
        call expect(fortbo_dominates([1.0_dp, 2.0_dp], [1.0_dp, 3.0_dp]), &
            "equal in one and better in the other dominates", failures)
        call expect(.not. fortbo_dominates([1.0_dp, 3.0_dp], [2.0_dp, 2.0_dp]), &
            "a trade-off does not dominate", failures)
        call expect(.not. fortbo_dominates([1.0_dp, 1.0_dp], [1.0_dp, 1.0_dp]), &
            "an identical point does not dominate", failures)
        call expect(.not. fortbo_dominates([2.0_dp, 2.0_dp], [1.0_dp, 1.0_dp]), &
            "a worse point does not dominate", failures)
    end subroutine check_dominance

    !! Brute-force oracle: after inserting everything, the archive must equal
    !! the set of points not dominated by any other inserted point.
    subroutine check_archive_matches_brute_force(failures)
        integer, intent(inout) :: failures
        type(fortbo_pareto_archive_t) :: archive
        type(fortnum_status_t) :: status
        integer, parameter :: n = 60
        real(dp) :: candidates(n, 2), front(0, 0)
        real(dp), allocatable :: reported(:, :)
        logical :: accepted, dominated, found
        integer :: i, j, k, expected_count

        do i = 1, n
            candidates(i, 1) = sin(0.7_dp*real(i, dp))*2.0_dp + 3.0_dp
            candidates(i, 2) = cos(1.1_dp*real(i, dp))*2.0_dp + 3.0_dp
        end do

        call archive%initialize(2, 1, status)
        call expect(status%code == FORTNUM_OK, "the archive initializes", failures)
        do i = 1, n
            call archive%insert([real(i, dp)], candidates(i, :), accepted, status)
        end do

        expected_count = 0
        do i = 1, n
            dominated = .false.
            do j = 1, n
                if (i == j) cycle
                if (fortbo_dominates(candidates(j, :), candidates(i, :))) then
                    dominated = .true.
                    exit
                end if
                if (j < i .and. all(candidates(j, :) == candidates(i, :))) then
                    dominated = .true.
                    exit
                end if
            end do
            if (.not. dominated) expected_count = expected_count + 1
        end do

        call expect(archive%count == expected_count, &
            "the archive size matches a brute-force non-dominated scan", &
            failures)

        call archive%front(reported, status)
        found = .true.
        do k = 1, archive%count
            dominated = .false.
            do j = 1, n
                if (fortbo_dominates(candidates(j, :), reported(k, :))) dominated = .true.
            end do
            if (dominated) found = .false.
        end do
        call expect(found, "no archived point is dominated by any inserted point", &
            failures)
        call expect(size(front, 1) == 0, "an empty front is representable", failures)
    end subroutine check_archive_matches_brute_force

    !! Two dimensions have a closed form: sort by the first objective and sum
    !! the staircase rectangles.
    subroutine check_two_dimensional_area(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: points(3, 2), reference(2), volume, expected

        ! Front: (1,4), (2,2), (4,1). Reference (5,5).
        points = reshape([1.0_dp, 2.0_dp, 4.0_dp, 4.0_dp, 2.0_dp, 1.0_dp], [3, 2])
        reference = [5.0_dp, 5.0_dp]

        ! Staircase, computed by hand: the region dominated by the three points
        ! below (5,5) splits into rectangles
        !   (5-1)*(5-4) = 4      strip above y=4
        !   (5-2)*(4-2) = 6      strip between y=2 and y=4
        !   (5-4)*(2-1) = 1      strip between y=1 and y=2
        expected = 4.0_dp + 6.0_dp + 1.0_dp

        call fortbo_hypervolume(points, reference, volume, status)
        call expect(status%code == FORTNUM_OK, "the hypervolume computes", failures)
        call expect(abs(volume - expected) < 1.0e-12_dp, &
            "the two-dimensional hypervolume matches the staircase area", &
            failures)

        ! A single point is a plain rectangle.
        call fortbo_hypervolume(reshape([1.0_dp, 2.0_dp], [1, 2]), reference, volume, &
            status)
        call expect(abs(volume - 4.0_dp*3.0_dp) < 1.0e-12_dp, &
            "a single point gives its rectangle", failures)

        ! A point outside the reference contributes nothing.
        call fortbo_hypervolume(reshape([9.0_dp, 9.0_dp], [1, 2]), reference, volume, &
            status)
        call expect(volume == 0.0_dp, "a point beyond the reference contributes zero", &
            failures)
    end subroutine check_two_dimensional_area

    !! Monte Carlo oracle in three and four dimensions, where no hand formula is
    !! practical. The estimator is a binomial proportion, so its standard error
    !! is known and the tolerance follows from it.
    subroutine check_against_monte_carlo(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        type(rng_t) :: generator
        integer, parameter :: n_samples = 400000
        real(dp) :: points3(4, 3), reference3(3)
        real(dp) :: points4(5, 4), reference4(4)
        real(dp) :: exact, estimate, standard_error, box

        points3 = reshape([ &
            1.0_dp, 2.0_dp, 3.0_dp, 1.5_dp, &
            3.0_dp, 1.0_dp, 2.0_dp, 2.5_dp, &
            2.0_dp, 3.0_dp, 1.0_dp, 2.0_dp], [4, 3])
        reference3 = [5.0_dp, 5.0_dp, 5.0_dp]

        call fortbo_hypervolume(points3, reference3, exact, status)
        call expect(status%code == FORTNUM_OK, "the three-dimensional volume computes", &
            failures)
        call rng_seed(generator, int(4711, int64), status)
        call monte_carlo_volume(points3, reference3, n_samples, generator, estimate, &
            standard_error, box)
        call expect(abs(exact - estimate) < 5.0_dp*standard_error, &
            "the three-dimensional volume matches Monte Carlo", failures)
        call expect(exact > 0.0_dp .and. exact < box, &
            "the volume is positive and below the bounding box", failures)

        points4 = reshape([ &
            1.0_dp, 2.0_dp, 3.0_dp, 2.5_dp, 1.5_dp, &
            3.0_dp, 1.0_dp, 2.0_dp, 1.5_dp, 2.5_dp, &
            2.0_dp, 3.0_dp, 1.0_dp, 2.5_dp, 1.5_dp, &
            2.5_dp, 1.5_dp, 2.5_dp, 1.0_dp, 3.0_dp], [5, 4])
        reference4 = [4.0_dp, 4.0_dp, 4.0_dp, 4.0_dp]

        call fortbo_hypervolume(points4, reference4, exact, status)
        call expect(status%code == FORTNUM_OK, "the four-dimensional volume computes", &
            failures)
        call rng_seed(generator, int(1234, int64), status)
        call monte_carlo_volume(points4, reference4, n_samples, generator, estimate, &
            standard_error, box)
        call expect(abs(exact - estimate) < 5.0_dp*standard_error, &
            "the four-dimensional volume matches Monte Carlo", failures)
    end subroutine check_against_monte_carlo

    subroutine monte_carlo_volume(points, reference, n_samples, generator, estimate, &
            standard_error, box)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: reference(:)
        integer, intent(in) :: n_samples
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: estimate
        real(dp), intent(out) :: standard_error
        real(dp), intent(out) :: box
        real(dp), allocatable :: lower(:), sample(:)
        real(dp) :: uniform, fraction
        integer :: d, i, j, k, inside
        logical :: dominated

        d = size(reference)
        allocate (lower(d), sample(d))
        do j = 1, d
            lower(j) = minval(points(:, j))
        end do
        box = product(reference - lower)

        inside = 0
        do i = 1, n_samples
            do j = 1, d
                call rng_uniform(generator, uniform)
                sample(j) = lower(j) + uniform*(reference(j) - lower(j))
            end do
            dominated = .false.
            do k = 1, size(points, 1)
                if (all(points(k, :) <= sample)) then
                    dominated = .true.
                    exit
                end if
            end do
            if (dominated) inside = inside + 1
        end do

        fraction = real(inside, dp)/real(n_samples, dp)
        estimate = fraction*box
        standard_error = box*sqrt(max(fraction*(1.0_dp - fraction), 1.0e-12_dp) &
            /real(n_samples, dp))
    end subroutine monte_carlo_volume

    !! The property that makes hypervolume a sound objective.
    subroutine check_monotonicity(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 2), extended(3, 2), reference(2)
        real(dp) :: before, after

        points = reshape([1.0_dp, 3.0_dp, 3.0_dp, 1.0_dp], [2, 2])
        reference = [5.0_dp, 5.0_dp]
        call fortbo_hypervolume(points, reference, before, status)

        ! A non-dominated addition must strictly increase the volume.
        extended(1:2, :) = points
        extended(3, :) = [2.0_dp, 2.0_dp]
        call fortbo_hypervolume(extended, reference, after, status)
        call expect(after > before, &
            "a non-dominated point strictly increases hypervolume", failures)

        ! A dominated addition must leave it exactly unchanged.
        extended(3, :) = [4.0_dp, 4.0_dp]
        call fortbo_hypervolume(extended, reference, after, status)
        call expect(after == before, &
            "a dominated point leaves hypervolume exactly unchanged", failures)

        ! Improving an existing point must increase it.
        extended(1:2, :) = points
        extended(1, :) = [0.5_dp, 3.0_dp]
        call fortbo_hypervolume(extended(1:2, :), reference, after, status)
        call expect(after > before, "improving a point increases hypervolume", failures)
    end subroutine check_monotonicity

    subroutine check_improvement(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: points(2, 2), reference(2), improvement
        real(dp) :: before, after
        real(dp) :: extended(3, 2)

        points = reshape([1.0_dp, 3.0_dp, 3.0_dp, 1.0_dp], [2, 2])
        reference = [5.0_dp, 5.0_dp]

        call fortbo_hypervolume_improvement(points, [2.0_dp, 2.0_dp], reference, &
            improvement, status)
        call expect(status%code == FORTNUM_OK, "improvement computes", failures)
        call fortbo_hypervolume(points, reference, before, status)
        extended(1:2, :) = points
        extended(3, :) = [2.0_dp, 2.0_dp]
        call fortbo_hypervolume(extended, reference, after, status)
        call expect(abs(improvement - (after - before)) < 1.0e-12_dp, &
            "improvement is the difference of the two volumes", failures)
        call expect(improvement > 0.0_dp, "a filling point has positive improvement", &
            failures)

        call fortbo_hypervolume_improvement(points, [4.0_dp, 4.0_dp], reference, &
            improvement, status)
        call expect(improvement == 0.0_dp, &
            "a dominated point has zero improvement", failures)

        call fortbo_hypervolume_improvement(points, [9.0_dp, 9.0_dp], reference, &
            improvement, status)
        call expect(improvement == 0.0_dp, &
            "a point beyond the reference has zero improvement", failures)
    end subroutine check_improvement

    !! The augmentation term is what stops the Chebyshev scalarization from
    !! rating a weakly dominated point as well as the point that dominates it.
    subroutine check_scalarizations(failures)
        integer, intent(inout) :: failures
        real(dp) :: weights(2), ideal(2)
        real(dp) :: dominating, dominated

        weights = [0.5_dp, 0.5_dp]
        ideal = [0.0_dp, 0.0_dp]

        call expect(abs(fortbo_weighted_sum([2.0_dp, 4.0_dp], weights) - 3.0_dp) &
            < 1.0e-14_dp, "the weighted sum is the weighted mean", failures)

        ! (1,2) dominates (2,2); both share the same worst weighted objective.
        dominating = fortbo_chebyshev([1.0_dp, 2.0_dp], weights, ideal, 0.0_dp)
        dominated = fortbo_chebyshev([2.0_dp, 2.0_dp], weights, ideal, 0.0_dp)
        call expect(dominating == dominated, &
            "without augmentation the two score identically", failures)

        dominating = fortbo_chebyshev([1.0_dp, 2.0_dp], weights, ideal, 0.01_dp)
        dominated = fortbo_chebyshev([2.0_dp, 2.0_dp], weights, ideal, 0.01_dp)
        call expect(dominating < dominated, &
            "augmentation separates the dominating point", failures)
    end subroutine check_scalarizations

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_pareto_archive_t) :: archive
        type(fortnum_status_t) :: status
        real(dp) :: volume, points(1, 2)
        logical :: accepted

        call archive%initialize(1, 1, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a single-objective archive is refused", failures)

        call archive%initialize(2, 1, status)
        call archive%insert([0.0_dp], [1.0_dp, 2.0_dp, 3.0_dp], accepted, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an objective-width mismatch is refused", failures)

        points = reshape([1.0_dp, 2.0_dp], [1, 2])
        call fortbo_hypervolume(points, [1.0_dp, 2.0_dp, 3.0_dp], volume, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a reference of the wrong width is refused", failures)
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

end program test_pareto
