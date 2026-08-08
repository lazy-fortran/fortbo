program test_mixed
    !! BO3: mixed-integer and categorical candidate search.
    !!
    !! Oracles:
    !!
    !!   * the neighbourhood is checked against exhaustive enumeration of the
    !!     discrete product, which the test can afford at this size and the
    !!     search deliberately cannot;
    !!   * the local search is checked against the exhaustive minimum on a
    !!     problem where the discrete neighbourhood is enough to reach it, and
    !!     is checked to *stop* rather than claim convergence on one where it is
    !!     not — a local search that reported a truncated walk as a local
    !!     optimum would be the dangerous failure;
    !!   * the refusal is checked to fire on a mixed space and *not* on a purely
    !!     continuous one, since refusing there would break the ordinary
    !!     gradient path.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_space, only: fortbo_space_t
    use fortbo_mixed, only: fortbo_mixed_gradient_refusal, &
        fortbo_discrete_neighbours, fortbo_mixed_local_search
    implicit none

    integer :: failures

    failures = 0
    call check_gradient_refusal(failures)
    call check_neighbourhood(failures)
    call check_local_search_finds_the_minimum(failures)
    call check_truncated_walk_is_reported(failures)
    call check_continuous_space(failures)

    if (failures == 0) then
        print *, "test_mixed: PASS"
    else
        print *, "test_mixed: FAIL", failures
        error stop 1
    end if

contains

    !! An integer in [1,4] and a categorical with three levels.
    subroutine build_mixed_space(space, status)
        type(fortbo_space_t), intent(out) :: space
        type(fortnum_status_t), intent(out) :: status

        call space%add_integer("width", 1, 4, status)
        if (status%code /= FORTNUM_OK) return
        call space%add_categorical("mode", 3, status)
        if (status%code /= FORTNUM_OK) return
        call space%finalize(status)
    end subroutine build_mixed_space

    subroutine check_gradient_refusal(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: mixed, continuous
        type(fortnum_status_t) :: status

        call build_mixed_space(mixed, status)
        call fortbo_mixed_gradient_refusal(mixed, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a mixed space refuses a gradient by name", failures)

        call continuous%add_continuous("x", 0.0_dp, 1.0_dp, status)
        call continuous%add_continuous("y", -1.0_dp, 1.0_dp, status)
        call continuous%finalize(status)
        call fortbo_mixed_gradient_refusal(continuous, status)
        call expect(status%code == FORTNUM_OK, &
            "a purely continuous space does not refuse", failures)
    end subroutine check_gradient_refusal

    !! Against exhaustive enumeration: the neighbourhood of (w, m) must be
    !! exactly the integer's in-range neighbours plus the categorical's other
    !! levels, and nothing else.
    subroutine check_neighbourhood(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp), allocatable :: point(:), neighbours(:, :), decoded(:)
        integer :: n_found, k, expected
        integer :: width, mode
        logical :: correct

        call build_mixed_space(space, status)
        allocate (neighbours(16, space%n_coordinates()))
        allocate (point(space%n_coordinates()), decoded(space%n_variables()))

        correct = .true.
        do width = 1, 4
            do mode = 1, 3
                call encode(space, real(width, dp), real(mode, dp), point, status)
                call fortbo_discrete_neighbours(space, point, neighbours, n_found, &
                    status)
                if (status%code /= FORTNUM_OK) correct = .false.

                ! Two integer neighbours unless at a bound, plus two other modes.
                expected = 2
                if (width == 1 .or. width == 4) expected = 1
                expected = expected + 2
                if (n_found /= expected) correct = .false.

                ! Every neighbour must differ from the start in exactly one
                ! variable, and must be a legal point.
                do k = 1, n_found
                    call space%from_unit(neighbours(k, :), decoded, status)
                    if (status%code /= FORTNUM_OK) correct = .false.
                    if (differing(decoded, real(width, dp), real(mode, dp)) /= 1) &
                        correct = .false.
                    if (decoded(1) < 1.0_dp .or. decoded(1) > 4.0_dp) correct = .false.
                    if (decoded(2) < 1.0_dp .or. decoded(2) > 3.0_dp) correct = .false.
                end do
            end do
        end do
        call expect(correct, &
            "the neighbourhood is exactly the one-coordinate discrete moves", &
            failures)
    end subroutine check_neighbourhood

    integer function differing(decoded, width, mode) result(count)
        real(dp), intent(in) :: decoded(:), width, mode

        count = 0
        if (abs(decoded(1) - width) > 0.5_dp) count = count + 1
        if (abs(decoded(2) - mode) > 0.5_dp) count = count + 1
    end function differing

    subroutine encode(space, width, mode, point, status)
        type(fortbo_space_t), intent(in) :: space
        real(dp), intent(in) :: width, mode
        real(dp), intent(out) :: point(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: decoded(2)

        decoded = [width, mode]
        call space%to_unit(decoded, point, status)
    end subroutine encode

    !! A bowl over the discrete grid, whose minimum the neighbourhood walk can
    !! reach from anywhere. The exhaustive minimum is computed here.
    subroutine check_local_search_finds_the_minimum(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp), allocatable :: start(:), best_point(:), decoded(:)
        real(dp) :: best_value, exhaustive, trial(1)
        real(dp), allocatable :: probe(:)
        integer :: rounds, width, mode
        logical :: converged, reached

        call build_mixed_space(space, status)
        allocate (start(space%n_coordinates()), best_point(space%n_coordinates()))
        allocate (decoded(space%n_variables()), probe(space%n_coordinates()))

        ! Exhaustive minimum over the twelve combinations.
        exhaustive = huge(1.0_dp)
        do width = 1, 4
            do mode = 1, 3
                call encode(space, real(width, dp), real(mode, dp), probe, status)
                call bowl(reshape(probe, [1, size(probe)]), trial, status)
                exhaustive = min(exhaustive, trial(1))
            end do
        end do

        reached = .true.
        do width = 1, 4
            do mode = 1, 3
                call encode(space, real(width, dp), real(mode, dp), start, status)
                call fortbo_mixed_local_search(space, bowl, start, best_point, &
                    best_value, rounds, converged, status)
                if (status%code /= FORTNUM_OK) reached = .false.
                if (.not. converged) reached = .false.
                if (abs(best_value - exhaustive) > 1.0e-12_dp) reached = .false.
            end do
        end do
        call expect(reached, &
            "the local search reaches the exhaustive minimum from every start", &
            failures)

        call encode(space, 4.0_dp, 3.0_dp, start, status)
        call fortbo_mixed_local_search(space, bowl, start, best_point, best_value, &
            rounds, converged, status)
        call expect(rounds >= 1, "the search takes at least one round", failures)
        call space%from_unit(best_point, decoded, status)
        call expect(abs(decoded(1) - 2.0_dp) < 0.5_dp .and. &
            abs(decoded(2) - 1.0_dp) < 0.5_dp, &
            "the search lands on the minimizing combination", failures)
    end subroutine check_local_search_finds_the_minimum

    !! Minimized at width 2, mode 1.
    subroutine bowl(points, values, status)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_space_t) :: space
        real(dp) :: decoded(2)
        integer :: i

        call build_mixed_space(space, status)
        do i = 1, size(points, 1)
            call space%from_unit(points(i, :), decoded, status)
            if (status%code /= FORTNUM_OK) return
            values(i) = (decoded(1) - 2.0_dp)**2 + (decoded(2) - 1.0_dp)**2
        end do
    end subroutine bowl

    !! A one-round budget on a problem needing more must report that it did not
    !! converge. Reporting a truncated walk as a local optimum is the failure
    !! that matters here.
    subroutine check_truncated_walk_is_reported(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp), allocatable :: start(:), best_point(:)
        real(dp) :: best_value
        integer :: rounds
        logical :: converged

        call build_mixed_space(space, status)
        allocate (start(space%n_coordinates()), best_point(space%n_coordinates()))
        call encode(space, 4.0_dp, 3.0_dp, start, status)

        call fortbo_mixed_local_search(space, bowl, start, best_point, best_value, &
            rounds, converged, status, max_rounds=1)
        call expect(status%code == FORTNUM_OK, "the truncated search still succeeds", &
            failures)
        call expect(.not. converged, &
            "a truncated walk is not reported as converged", failures)
        call expect(rounds == 1, "the round budget is respected", failures)

        call fortbo_mixed_local_search(space, bowl, start, best_point, best_value, &
            rounds, converged, status, max_rounds=0)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero round budget is refused", failures)
    end subroutine check_truncated_walk_is_reported

    !! A continuous space has no discrete neighbourhood, so the start is
    !! trivially a discrete local optimum. Saying so beats refusing: a caller
    !! sweeping mixed spaces should not have to special-case this one.
    subroutine check_continuous_space(failures)
        integer, intent(inout) :: failures
        type(fortbo_space_t) :: space
        type(fortnum_status_t) :: status
        real(dp) :: start(2), best_point(2), best_value
        integer :: rounds
        logical :: converged

        call space%add_continuous("x", 0.0_dp, 1.0_dp, status)
        call space%add_continuous("y", 0.0_dp, 1.0_dp, status)
        call space%finalize(status)

        start = [0.3_dp, 0.7_dp]
        call fortbo_mixed_local_search(space, constant_score, start, best_point, &
            best_value, rounds, converged, status)
        call expect(status%code == FORTNUM_OK, &
            "a continuous space is handled rather than refused", failures)
        call expect(converged, &
            "a continuous space is trivially a discrete local optimum", failures)
        call expect(maxval(abs(best_point - start)) == 0.0_dp, &
            "the start is returned unchanged", failures)
    end subroutine check_continuous_space

    subroutine constant_score(points, values, status)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        values = 1.0_dp
        call status_ok_here(status)
    end subroutine constant_score

    subroutine status_ok_here(status)
        use fortnum_status, only: status_set
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_OK, "")
    end subroutine status_ok_here

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_mixed
