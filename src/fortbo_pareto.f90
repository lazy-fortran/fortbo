module fortbo_pareto
    !! Pareto archives, hypervolume, and scalarization (ROADMAP BO4).
    !!
    !! Every objective is minimized, and a point `a` dominates `b` when it is no
    !! worse in every objective and strictly better in at least one. The archive
    !! keeps exactly the non-dominated points, in insertion order.
    !!
    !! Hypervolume is the volume of the region dominated by the archive and
    !! bounded by a reference point. It is the standard scalar quality measure
    !! for a front because it is the only common indicator that is strictly
    !! monotone with respect to dominance: improving any point, or adding a
    !! non-dominated one, always increases it. That is what makes hypervolume
    !! *improvement* a sound acquisition — maximizing it cannot reward a move
    !! that makes the front worse.
    !!
    !! The computation is exact, by dimension sweep (Fonseca, Paquete, and
    !! Lopez-Ibanez). Slice along the last objective; within each slab the
    !! dominated region is a cylinder over the `d-1`-dimensional hypervolume of
    !! the points that have entered so far, so the problem recurses. Monte Carlo
    !! would have been far easier to write and is what the test uses as an
    !! independent oracle, but an approximate indicator inside an acquisition
    !! turns a tie into a coin flip and breaks replay.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter :: INITIAL_CAPACITY = 16

    public :: fortbo_pareto_archive_t
    public :: fortbo_dominates
    public :: fortbo_hypervolume
    public :: fortbo_hypervolume_improvement
    public :: fortbo_weighted_sum
    public :: fortbo_chebyshev

    type :: fortbo_pareto_archive_t
        integer :: n_objectives = 0
        integer :: count = 0
        real(dp), allocatable :: objectives(:, :)
        real(dp), allocatable :: inputs(:, :)
        integer :: n_inputs = 0
    contains
        procedure, public :: initialize => archive_initialize
        procedure, public :: insert => archive_insert
        procedure, public :: contains_dominator => archive_contains_dominator
        procedure, public :: front => archive_front
        procedure, public :: hypervolume => archive_hypervolume
    end type fortbo_pareto_archive_t

contains

    !! `a` dominates `b`: no worse everywhere, strictly better somewhere.
    pure logical function fortbo_dominates(a, b) result(dominates)
        real(dp), intent(in) :: a(:)
        real(dp), intent(in) :: b(:)

        dominates = .false.
        if (size(a) /= size(b)) return
        if (any(a > b)) return
        dominates = any(a < b)
    end function fortbo_dominates

    subroutine archive_initialize(self, n_objectives, n_inputs, status)
        class(fortbo_pareto_archive_t), intent(out) :: self
        integer, intent(in) :: n_objectives
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        if (n_objectives < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: an archive needs at least two objectives")
            return
        end if
        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: n_inputs must be positive")
            return
        end if
        self%n_objectives = n_objectives
        self%n_inputs = n_inputs
        self%count = 0
        allocate (self%objectives(INITIAL_CAPACITY, n_objectives))
        allocate (self%inputs(INITIAL_CAPACITY, n_inputs))
        self%objectives = 0.0_dp
        self%inputs = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine archive_initialize

    !! Insert a point, dropping anything it dominates. `accepted` reports
    !! whether the point entered the front; a dominated or duplicate point is
    !! rejected without changing the archive.
    subroutine archive_insert(self, input, objectives, accepted, status)
        class(fortbo_pareto_archive_t), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(in) :: objectives(:)
        logical, intent(out) :: accepted
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: kept_objectives(:, :), kept_inputs(:, :)
        integer :: i, kept

        accepted = .false.
        if (size(objectives) /= self%n_objectives .or. self%n_objectives == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: objective width does not match")
            return
        end if
        if (size(input) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: input width does not match")
            return
        end if

        ! Dominated by, or identical to, something already present.
        do i = 1, self%count
            if (fortbo_dominates(self%objectives(i, :), objectives)) then
                call status_set(status, FORTNUM_OK, "")
                return
            end if
            if (all(self%objectives(i, :) == objectives)) then
                call status_set(status, FORTNUM_OK, "")
                return
            end if
        end do

        allocate (kept_objectives(self%count + 1, self%n_objectives))
        allocate (kept_inputs(self%count + 1, self%n_inputs))
        kept = 0
        do i = 1, self%count
            if (fortbo_dominates(objectives, self%objectives(i, :))) cycle
            kept = kept + 1
            kept_objectives(kept, :) = self%objectives(i, :)
            kept_inputs(kept, :) = self%inputs(i, :)
        end do
        kept = kept + 1
        kept_objectives(kept, :) = objectives
        kept_inputs(kept, :) = input

        if (size(self%objectives, 1) < kept) then
            deallocate (self%objectives, self%inputs)
            allocate (self%objectives(2*kept, self%n_objectives))
            allocate (self%inputs(2*kept, self%n_inputs))
        end if
        self%objectives(1:kept, :) = kept_objectives(1:kept, :)
        self%inputs(1:kept, :) = kept_inputs(1:kept, :)
        self%count = kept
        accepted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine archive_insert

    pure logical function archive_contains_dominator(self, objectives) result(found)
        class(fortbo_pareto_archive_t), intent(in) :: self
        real(dp), intent(in) :: objectives(:)
        integer :: i

        found = .false.
        do i = 1, self%count
            if (fortbo_dominates(self%objectives(i, :), objectives)) then
                found = .true.
                return
            end if
        end do
    end function archive_contains_dominator

    subroutine archive_front(self, objectives, status)
        class(fortbo_pareto_archive_t), intent(in) :: self
        real(dp), intent(out), allocatable :: objectives(:, :)
        type(fortnum_status_t), intent(out) :: status

        allocate (objectives(self%count, self%n_objectives))
        if (self%count > 0) objectives = self%objectives(1:self%count, :)
        call status_set(status, FORTNUM_OK, "")
    end subroutine archive_front

    subroutine archive_hypervolume(self, reference, volume, status)
        class(fortbo_pareto_archive_t), intent(in) :: self
        real(dp), intent(in) :: reference(:)
        real(dp), intent(out) :: volume
        type(fortnum_status_t), intent(out) :: status

        call fortbo_hypervolume(self%objectives(1:self%count, :), reference, volume, &
            status)
    end subroutine archive_hypervolume

    !! Exact hypervolume of the region dominated by `points` and bounded above
    !! by `reference`. Points not strictly below the reference in every
    !! objective contribute nothing and are dropped.
    subroutine fortbo_hypervolume(points, reference, volume, status)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: reference(:)
        real(dp), intent(out) :: volume
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: usable(:, :)
        integer :: i, n, d, kept

        volume = 0.0_dp
        d = size(reference)
        if (d < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: reference point is empty")
            return
        end if
        if (size(points, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: point width does not match the reference")
            return
        end if

        n = size(points, 1)
        allocate (usable(n, d))
        kept = 0
        do i = 1, n
            if (any(points(i, :) >= reference)) cycle
            kept = kept + 1
            usable(kept, :) = points(i, :)
        end do
        if (kept == 0) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        volume = sweep_hypervolume(usable(1:kept, :), reference)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_hypervolume

    !! Recursive dimension sweep. In one dimension the dominated region is the
    !! interval from the best point to the reference.
    recursive function sweep_hypervolume(points, reference) result(volume)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: reference(:)
        real(dp) :: volume
        real(dp), allocatable :: sorted(:, :), projected(:, :), reduced(:, :)
        real(dp) :: depth, upper
        integer :: n, d, i, k, kept

        n = size(points, 1)
        d = size(reference)
        volume = 0.0_dp
        if (n == 0) return

        if (d == 1) then
            volume = reference(1) - minval(points(:, 1))
            volume = max(volume, 0.0_dp)
            return
        end if

        allocate (sorted(n, d))
        sorted = points
        call sort_by_column(sorted, d)

        do k = 1, n
            if (k < n) then
                upper = sorted(k + 1, d)
            else
                upper = reference(d)
            end if
            depth = upper - sorted(k, d)
            if (depth <= 0.0_dp) cycle

            ! Everything entered so far, projected out of the sweep dimension.
            allocate (projected(k, d - 1))
            do i = 1, k
                projected(i, :) = sorted(i, 1:d - 1)
            end do
            call keep_nondominated(projected, reduced, kept)
            if (kept > 0) then
                volume = volume + depth &
                    *sweep_hypervolume(reduced(1:kept, :), reference(1:d - 1))
            end if
            deallocate (projected)
            if (allocated(reduced)) deallocate (reduced)
        end do
    end function sweep_hypervolume

    !! Insertion sort on one column. Fronts in Bayesian optimization are small,
    !! and a stable simple sort keeps the sweep deterministic on ties.
    pure subroutine sort_by_column(points, column)
        real(dp), intent(inout) :: points(:, :)
        integer, intent(in) :: column
        real(dp), allocatable :: row(:)
        integer :: i, j

        allocate (row(size(points, 2)))
        do i = 2, size(points, 1)
            row = points(i, :)
            j = i - 1
            do while (j >= 1)
                if (points(j, column) <= row(column)) exit
                points(j + 1, :) = points(j, :)
                j = j - 1
            end do
            points(j + 1, :) = row
        end do
    end subroutine sort_by_column

    pure subroutine keep_nondominated(points, kept_points, kept)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out), allocatable :: kept_points(:, :)
        integer, intent(out) :: kept
        integer :: i, j
        logical :: dominated

        allocate (kept_points(size(points, 1), size(points, 2)))
        kept = 0
        do i = 1, size(points, 1)
            dominated = .false.
            do j = 1, size(points, 1)
                if (i == j) cycle
                if (fortbo_dominates(points(j, :), points(i, :))) then
                    dominated = .true.
                    exit
                end if
                ! Keep only the first of a set of identical points.
                if (j < i .and. all(points(j, :) == points(i, :))) then
                    dominated = .true.
                    exit
                end if
            end do
            if (dominated) cycle
            kept = kept + 1
            kept_points(kept, :) = points(i, :)
        end do
    end subroutine keep_nondominated

    !! Increase in hypervolume from adding `candidate`. Zero when the candidate
    !! is already dominated, which is exactly the behavior an acquisition wants:
    !! no reward for a point that cannot improve the front.
    subroutine fortbo_hypervolume_improvement(points, candidate, reference, &
            improvement, status)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: candidate(:)
        real(dp), intent(in) :: reference(:)
        real(dp), intent(out) :: improvement
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: extended(:, :)
        real(dp) :: before, after
        integer :: n, d

        improvement = 0.0_dp
        d = size(reference)
        n = size(points, 1)
        if (size(candidate) /= d .or. size(points, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo pareto: candidate width does not match")
            return
        end if

        call fortbo_hypervolume(points, reference, before, status)
        if (status%code /= FORTNUM_OK) return

        allocate (extended(n + 1, d))
        if (n > 0) extended(1:n, :) = points
        extended(n + 1, :) = candidate
        call fortbo_hypervolume(extended, reference, after, status)
        if (status%code /= FORTNUM_OK) return

        improvement = max(after - before, 0.0_dp)
    end subroutine fortbo_hypervolume_improvement

    pure real(dp) function fortbo_weighted_sum(objectives, weights) result(value)
        real(dp), intent(in) :: objectives(:)
        real(dp), intent(in) :: weights(:)

        value = sum(weights*objectives)
    end function fortbo_weighted_sum

    !! Augmented Chebyshev scalarization. The augmentation term is what keeps
    !! the scalarization from returning weakly dominated points: without it a
    !! point that ties on the worst objective but is worse on every other scores
    !! identically to the one that dominates it.
    pure real(dp) function fortbo_chebyshev(objectives, weights, ideal, rho) &
            result(value)
        real(dp), intent(in) :: objectives(:)
        real(dp), intent(in) :: weights(:)
        real(dp), intent(in) :: ideal(:)
        real(dp), intent(in) :: rho

        value = maxval(weights*(objectives - ideal)) &
            + rho*sum(weights*(objectives - ideal))
    end function fortbo_chebyshev

end module fortbo_pareto
