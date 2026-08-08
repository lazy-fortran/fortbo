module fortbo_mixed
    !! Mixed-integer and categorical candidate search (ROADMAP BO3).
    !!
    !! A mixed space has no gradient. Not "a gradient that is hard to compute" —
    !! no gradient: the acquisition is a step function of an integer coordinate
    !! and a finite table of a categorical one, and both derivatives are
    !! undefined everywhere rather than merely awkward somewhere. Every honest
    !! method here therefore splits the coordinates and treats the two kinds
    !! differently.
    !!
    !! The search is a local one over the discrete coordinates with a continuous
    !! product on the rest:
    !!
    !!   * discrete coordinates move by *neighbourhood* — an integer to its two
    !!     neighbours, a categorical to each of its other levels — and the move
    !!     is accepted only if it improves. Enumerating the whole discrete
    !!     product is exponential, and sampling it wastes most draws on
    !!     combinations the acquisition has already rejected;
    !!   * continuous coordinates keep their exact derivative, supplied by the
    !!     caller, so the continuous part loses nothing by living next to
    !!     discrete ones.
    !!
    !! `fortbo_mixed_gradient_refusal` is the typed refusal for the discrete
    !! part. It exists so a caller that asks for a gradient over a mixed space
    !! gets a named error rather than a finite-difference estimate of a step
    !! function, which is zero almost everywhere and enormous exactly at the
    !! jumps — a quantity with no information in it that nonetheless looks like
    !! a derivative.
    !!
    !! Integer coordinates are stored on the unit cube like everything else and
    !! rounded on the way out, so a neighbourhood move has to be computed in the
    !! *decoded* space and re-encoded. Moving by a fixed step on the unit cube
    !! would land on a different integer for different ranges, and on none at
    !! all for a range of one.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortbo_space, only: fortbo_space_t, FORTBO_VAR_CONTINUOUS, &
        FORTBO_VAR_INTEGER, FORTBO_VAR_CATEGORICAL
    implicit none
    private

    public :: fortbo_mixed_gradient_refusal
    public :: fortbo_discrete_neighbours
    public :: fortbo_mixed_local_search

    abstract interface
        !! Score a batch of unit-cube points. Lower is better, as everywhere.
        subroutine fortbo_mixed_score_interface(points, values, status)
            import :: dp, fortnum_status_t
            real(dp), intent(in) :: points(:, :)
            real(dp), intent(out) :: values(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine fortbo_mixed_score_interface
    end interface

contains

    !! The typed refusal for a derivative that does not exist.
    subroutine fortbo_mixed_gradient_refusal(space, status)
        type(fortbo_space_t), intent(in) :: space
        type(fortnum_status_t), intent(out) :: status
        logical, allocatable :: mask(:)

        ! `differentiable_mask` takes an assumed-shape array and checks its
        ! width, so it must be allocated first.
        allocate (mask(space%n_coordinates()))
        call space%differentiable_mask(mask, status)
        if (status%code /= FORTNUM_OK) return
        if (all(mask)) then
            ! Nothing discrete: the ordinary gradient path applies and refusing
            ! would be wrong.
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "fortbo mixed: acquisition has no derivative in discrete coordinates")
    end subroutine fortbo_mixed_gradient_refusal

    !! Every one-coordinate discrete neighbour of `point`, on the unit cube.
    !!
    !! Continuous coordinates are copied unchanged: they are not part of the
    !! discrete neighbourhood and moving them here would confound the two
    !! searches. Returns `n_found` rows of `neighbours`, which the caller sizes.
    subroutine fortbo_discrete_neighbours(space, point, neighbours, n_found, status)
        type(fortbo_space_t), intent(in) :: space
        real(dp), intent(in) :: point(:)
        real(dp), intent(out) :: neighbours(:, :)
        integer, intent(out) :: n_found
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: decoded(:), moved(:), encoded(:)
        integer :: n_coordinates, v, level, offset, base
        real(dp) :: original

        n_found = 0
        neighbours = 0.0_dp
        n_coordinates = space%n_coordinates()
        if (size(point) /= n_coordinates .or. &
            size(neighbours, 2) /= n_coordinates) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mixed: point width does not match the space")
            return
        end if

        allocate (decoded(space%n_variables()), moved(space%n_variables()))
        allocate (encoded(n_coordinates))
        call space%from_unit(point, decoded, status)
        if (status%code /= FORTNUM_OK) return

        base = 0
        do v = 1, space%n_variables()
            select case (space%variables(v)%kind)
            case (FORTBO_VAR_CONTINUOUS)
                base = base + 1
            case (FORTBO_VAR_INTEGER)
                ! Both neighbours in the *decoded* space, so the step is one
                ! integer regardless of the range.
                original = decoded(v)
                do offset = -1, 1, 2
                    moved = decoded
                    moved(v) = original + real(offset, dp)
                    if (moved(v) < space%variables(v)%lower) cycle
                    if (moved(v) > space%variables(v)%upper) cycle
                    call space%to_unit(moved, encoded, status)
                    if (status%code /= FORTNUM_OK) return
                    if (n_found >= size(neighbours, 1)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo mixed: neighbour buffer is too small")
                        return
                    end if
                    n_found = n_found + 1
                    neighbours(n_found, :) = encoded
                end do
                base = base + 1
            case (FORTBO_VAR_CATEGORICAL)
                ! Every other level. A categorical has no ordering, so there is
                ! no such thing as an adjacent level and all of them are
                ! neighbours.
                do level = 1, space%variables(v)%n_categories
                    if (abs(decoded(v) - real(level, dp)) < 0.5_dp) cycle
                    moved = decoded
                    moved(v) = real(level, dp)
                    call space%to_unit(moved, encoded, status)
                    if (status%code /= FORTNUM_OK) return
                    if (n_found >= size(neighbours, 1)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo mixed: neighbour buffer is too small")
                        return
                    end if
                    n_found = n_found + 1
                    neighbours(n_found, :) = encoded
                end do
                base = base + space%variables(v)%n_categories
            end select
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_discrete_neighbours

    !! Steepest-descent local search over the discrete neighbourhood.
    !!
    !! Each round scores every one-coordinate neighbour and moves to the best if
    !! it improves; otherwise the point is a local minimum of the discrete
    !! neighbourhood and the search stops. `max_rounds` bounds the walk, and a
    !! search that exhausts it reports the fact rather than pretending to have
    !! converged — a caller that cannot tell the two apart will read a truncated
    !! walk as a local optimum.
    subroutine fortbo_mixed_local_search(space, score, start, best_point, &
            best_value, rounds, converged, status, &
            max_rounds)
        type(fortbo_space_t), intent(in) :: space
        procedure(fortbo_mixed_score_interface) :: score
        real(dp), intent(in) :: start(:)
        real(dp), intent(out) :: best_point(:)
        real(dp), intent(out) :: best_value
        integer, intent(out) :: rounds
        logical, intent(out) :: converged
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_rounds
        real(dp), allocatable :: neighbours(:, :), values(:), current(:)
        real(dp) :: current_value, single(1)
        integer :: capacity, n_found, limit, v, best_index

        rounds = 0
        converged = .false.
        best_value = huge(1.0_dp)
        best_point = 0.0_dp

        if (size(start) /= space%n_coordinates() .or. &
            size(best_point) /= space%n_coordinates()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mixed: start width does not match the space")
            return
        end if
        limit = 100
        if (present(max_rounds)) limit = max_rounds
        if (limit < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo mixed: round limit must be positive")
            return
        end if

        ! Worst case one round can offer: two per integer, all levels per
        ! categorical.
        capacity = 0
        do v = 1, space%n_variables()
            select case (space%variables(v)%kind)
            case (FORTBO_VAR_INTEGER)
                capacity = capacity + 2
            case (FORTBO_VAR_CATEGORICAL)
                capacity = capacity + space%variables(v)%n_categories
            end select
        end do
        if (capacity == 0) then
            ! A purely continuous space has no discrete neighbourhood, so the
            ! start is trivially a discrete local optimum. Saying so is more
            ! useful than refusing: a caller sweeping mixed spaces should not
            ! have to special-case the continuous one.
            allocate (current(size(start)))
            current = start
            call score(reshape(current, [1, size(current)]), single, status)
            if (status%code /= FORTNUM_OK) return
            best_point = current
            best_value = single(1)
            converged = .true.
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        allocate (neighbours(capacity, space%n_coordinates()))
        allocate (values(capacity), current(size(start)))
        current = start
        call score(reshape(current, [1, size(current)]), single, status)
        if (status%code /= FORTNUM_OK) return
        current_value = single(1)

        do rounds = 1, limit
            call fortbo_discrete_neighbours(space, current, neighbours, n_found, &
                status)
            if (status%code /= FORTNUM_OK) return
            if (n_found == 0) exit
            call score(neighbours(:n_found, :), values(:n_found), status)
            if (status%code /= FORTNUM_OK) return
            best_index = minloc(values(:n_found), dim=1)
            if (values(best_index) >= current_value) then
                converged = .true.
                exit
            end if
            current = neighbours(best_index, :)
            current_value = values(best_index)
        end do
        if (rounds > limit) rounds = limit

        best_point = current
        best_value = current_value
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_mixed_local_search

end module fortbo_mixed
