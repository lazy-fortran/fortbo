module fortbo_trust_region
    !! TuRBO trust regions (ROADMAP BO3T).
    !!
    !! A trust region is a hyperrectangle in the unit cube, centered on the
    !! incumbent, whose per-dimension side lengths are the surrogate's ARD
    !! lengthscales rescaled to a fixed total volume:
    !!
    !!     L_i = lambda_i * L / (prod_j lambda_j)^(1/d)
    !!
    !! The rescaling is the point of the construction. A single base length `L`
    !! controls how much of the space the region covers, while the lengthscales
    !! decide how that volume is *shaped* — long in the directions the surrogate
    !! says are smooth, short where it says the objective varies quickly. The
    !! product form keeps the volume exactly `L^d` regardless of the
    !! lengthscales, so `L` remains a meaningful, comparable scalar across
    !! iterations and across regions.
    !!
    !! Adaptation is the classical counter rule: `tau_succ` consecutive
    !! successful batches double the region up to `length_max`, `tau_fail`
    !! consecutive failures halve it, and both counters reset whenever the
    !! region resizes. A region that shrinks past `length_min` has concluded
    !! that its basin is exhausted; it is discarded and restarted elsewhere
    !! rather than allowed to collapse to a point.
    !!
    !! Defaults follow Eriksson et al. (NeurIPS 2019) and the authors'
    !! implementation. They are named constants with that provenance attached,
    !! not literals scattered through a loop.
    !!
    !! This module owns the state machine only. Candidate generation and
    !! Thompson selection sit above it, and the surrogate sits beside it, so the
    !! same region logic serves value-only TuRBO and derivative-informed DTuRBO
    !! without modification.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    !! Reference constants from `uber-research/TuRBO`.
    real(dp), parameter, public :: FORTBO_TR_LENGTH_INIT = 0.8_dp
    real(dp), parameter, public :: FORTBO_TR_LENGTH_MIN = 0.5_dp**7
    real(dp), parameter, public :: FORTBO_TR_LENGTH_MAX = 1.6_dp
    integer, parameter, public :: FORTBO_TR_SUCCESS_TOLERANCE = 3

    ! Outcome of one batch, recorded for the evidence trace.
    integer, parameter, public :: FORTBO_TR_UNCHANGED = 0
    integer, parameter, public :: FORTBO_TR_EXPANDED = 1
    integer, parameter, public :: FORTBO_TR_SHRANK = 2
    integer, parameter, public :: FORTBO_TR_EXHAUSTED = 3

    public :: fortbo_trust_region_t

    type :: fortbo_trust_region_t
        integer :: n_inputs = 0
        !! Base side length. The region covers `length**n_inputs` of the cube.
        real(dp) :: length = FORTBO_TR_LENGTH_INIT
        real(dp) :: length_init = FORTBO_TR_LENGTH_INIT
        real(dp) :: length_min = FORTBO_TR_LENGTH_MIN
        real(dp) :: length_max = FORTBO_TR_LENGTH_MAX
        integer :: success_tolerance = FORTBO_TR_SUCCESS_TOLERANCE
        integer :: failure_tolerance = 1
        integer :: success_counter = 0
        integer :: failure_counter = 0
        !! Relative improvement below which a batch counts as a failure. Zero
        !! means any strict improvement counts, which is the paper's rule.
        real(dp) :: improvement_tolerance = 0.0_dp
        real(dp), allocatable :: center(:)
        real(dp) :: center_value = 0.0_dp
        logical :: active = .false.
        !! True once the region has been placed at least once. Distinguishes the
        !! initial placement from a genuine restart, including the canonical
        !! case of restarting a region that has just been exhausted.
        logical :: placed = .false.
        integer :: restarts = 0
        integer :: batches = 0
        integer :: last_event = FORTBO_TR_UNCHANGED
    contains
        procedure, public :: initialize => region_initialize
        procedure, public :: restart => region_restart
        procedure, public :: failure_tolerance_for => region_failure_tolerance_for
        procedure, public :: side_lengths => region_side_lengths
        procedure, public :: bounds => region_bounds
        procedure, public :: contains_point => region_contains_point
        procedure, public :: observe_batch => region_observe_batch
        procedure, public :: volume_fraction => region_volume_fraction
    end type fortbo_trust_region_t

contains

    !! Prepare a region for a `n_inputs`-dimensional problem evaluated in
    !! batches of `batch_size`. The failure tolerance follows the reference
    !! implementation: `ceil(max(4, d) / q)`, so a wide batch is given fewer
    !! chances before the region shrinks, which keeps the number of *evaluations*
    !! spent on a failing region roughly constant as the batch grows.
    subroutine region_initialize(self, n_inputs, batch_size, status)
        class(fortbo_trust_region_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        integer, intent(in) :: batch_size
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: n_inputs must be positive")
            return
        end if
        if (batch_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: batch size must be positive")
            return
        end if
        self%n_inputs = n_inputs
        self%length = FORTBO_TR_LENGTH_INIT
        self%length_init = FORTBO_TR_LENGTH_INIT
        self%length_min = FORTBO_TR_LENGTH_MIN
        self%length_max = FORTBO_TR_LENGTH_MAX
        self%success_tolerance = FORTBO_TR_SUCCESS_TOLERANCE
        self%failure_tolerance = region_failure_tolerance_for(self, n_inputs, &
            batch_size)
        self%success_counter = 0
        self%failure_counter = 0
        allocate (self%center(n_inputs))
        self%center = 0.5_dp
        self%center_value = huge(1.0_dp)
        self%active = .false.
        self%placed = .false.
        self%restarts = 0
        self%batches = 0
        self%last_event = FORTBO_TR_UNCHANGED
        call status_set(status, FORTNUM_OK, "")
    end subroutine region_initialize

    pure integer function region_failure_tolerance_for(self, n_inputs, batch_size) &
            result(tolerance)
        class(fortbo_trust_region_t), intent(in) :: self
        integer, intent(in) :: n_inputs
        integer, intent(in) :: batch_size

        tolerance = self%success_tolerance
        if (batch_size < 1) return
        tolerance = (max(4, n_inputs) + batch_size - 1)/batch_size
        tolerance = max(tolerance, 1)
    end function region_failure_tolerance_for

    !! Place the region at `center` with incumbent value `value` and reset it to
    !! its initial size. Restarting is not a failure mode; it is how a local
    !! method covers a multimodal space.
    subroutine region_restart(self, center, value, status)
        class(fortbo_trust_region_t), intent(inout) :: self
        real(dp), intent(in) :: center(:)
        real(dp), intent(in) :: value
        type(fortnum_status_t), intent(out) :: status

        if (self%n_inputs == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: region is not initialized")
            return
        end if
        if (size(center) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: center width does not match")
            return
        end if
        if (any(center < 0.0_dp) .or. any(center > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: center must lie in the unit cube")
            return
        end if
        if (self%placed) self%restarts = self%restarts + 1
        self%center = center
        self%center_value = value
        self%length = self%length_init
        self%success_counter = 0
        self%failure_counter = 0
        self%active = .true.
        self%placed = .true.
        self%last_event = FORTBO_TR_UNCHANGED
        call status_set(status, FORTNUM_OK, "")
    end subroutine region_restart

    !! Per-dimension side lengths from the surrogate's ARD lengthscales. The
    !! geometric-mean normalization is what preserves the volume; a plain
    !! normalization by the sum or the maximum would let the region silently
    !! change size whenever the surrogate refitted.
    subroutine region_side_lengths(self, lengthscales, lengths, status)
        class(fortbo_trust_region_t), intent(in) :: self
        real(dp), intent(in) :: lengthscales(:)
        real(dp), intent(out) :: lengths(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_mean

        if (size(lengthscales) /= self%n_inputs .or. size(lengths) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: lengthscale width does not match")
            return
        end if
        if (any(lengthscales <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: lengthscales must be positive")
            return
        end if
        ! Work in logs: the direct product underflows for a few hundred
        ! dimensions, which is exactly where TuRBO is meant to be used.
        log_mean = sum(log(lengthscales))/real(self%n_inputs, dp)
        lengths = exp(log(lengthscales) - log_mean)*self%length
        call status_set(status, FORTNUM_OK, "")
    end subroutine region_side_lengths

    !! Region bounds, clipped to the unit cube. Clipping is deliberate and is
    !! why the realized volume can fall below `length**d` near a face.
    subroutine region_bounds(self, lengthscales, lower, upper, status)
        class(fortbo_trust_region_t), intent(in) :: self
        real(dp), intent(in) :: lengthscales(:)
        real(dp), intent(out) :: lower(:)
        real(dp), intent(out) :: upper(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: lengths(:)

        if (size(lower) /= self%n_inputs .or. size(upper) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: bound width does not match")
            return
        end if
        allocate (lengths(self%n_inputs))
        call region_side_lengths(self, lengthscales, lengths, status)
        if (status%code /= FORTNUM_OK) return
        lower = max(self%center - 0.5_dp*lengths, 0.0_dp)
        upper = min(self%center + 0.5_dp*lengths, 1.0_dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine region_bounds

    pure logical function region_contains_point(self, point, lengthscales) result(inside)
        class(fortbo_trust_region_t), intent(in) :: self
        real(dp), intent(in) :: point(:)
        real(dp), intent(in) :: lengthscales(:)
        real(dp) :: log_mean, half_width
        integer :: j

        inside = .false.
        if (size(point) /= self%n_inputs) return
        if (size(lengthscales) /= self%n_inputs) return
        if (any(lengthscales <= 0.0_dp)) return
        log_mean = sum(log(lengthscales))/real(self%n_inputs, dp)
        do j = 1, self%n_inputs
            half_width = 0.5_dp*exp(log(lengthscales(j)) - log_mean)*self%length
            if (point(j) < self%center(j) - half_width) return
            if (point(j) > self%center(j) + half_width) return
        end do
        inside = .true.
    end function region_contains_point

    !! Record the outcome of one evaluated batch and adapt.
    !!
    !! A batch succeeds when its best value improves on the region's incumbent.
    !! The improvement must be strict — a batch that merely reproduces the
    !! incumbent has learned nothing about where to go next, and counting it as
    !! progress lets a stalled region expand forever.
    subroutine region_observe_batch(self, inputs, values, status)
        class(fortbo_trust_region_t), intent(inout) :: self
        real(dp), intent(in) :: inputs(:, :)
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: best_value, required
        integer :: best_index, i
        logical :: improved

        if (.not. self%active) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: region is not active")
            return
        end if
        if (size(values) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: batch is empty")
            return
        end if
        if (size(inputs, 1) /= size(values) .or. size(inputs, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo trust region: batch shape does not match")
            return
        end if

        best_index = 1
        best_value = values(1)
        do i = 2, size(values)
            if (values(i) < best_value) then
                best_value = values(i)
                best_index = i
            end if
        end do

        required = self%improvement_tolerance*abs(self%center_value)
        improved = best_value < self%center_value - required
        self%batches = self%batches + 1

        if (improved) then
            self%center = inputs(best_index, :)
            self%center_value = best_value
            self%success_counter = self%success_counter + 1
            self%failure_counter = 0
        else
            self%failure_counter = self%failure_counter + 1
            self%success_counter = 0
        end if

        self%last_event = FORTBO_TR_UNCHANGED
        if (self%success_counter >= self%success_tolerance) then
            self%length = min(self%length_max, 2.0_dp*self%length)
            self%success_counter = 0
            self%failure_counter = 0
            self%last_event = FORTBO_TR_EXPANDED
        else if (self%failure_counter >= self%failure_tolerance) then
            self%length = 0.5_dp*self%length
            self%success_counter = 0
            self%failure_counter = 0
            self%last_event = FORTBO_TR_SHRANK
        end if

        if (self%length < self%length_min) then
            self%active = .false.
            self%last_event = FORTBO_TR_EXHAUSTED
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine region_observe_batch

    !! Fraction of the unit cube the region covers, computed in logs so it stays
    !! meaningful in the high dimensions this algorithm exists for.
    pure real(dp) function region_volume_fraction(self) result(fraction)
        class(fortbo_trust_region_t), intent(in) :: self

        fraction = 0.0_dp
        if (self%n_inputs < 1) return
        fraction = exp(real(self%n_inputs, dp)*log(min(self%length, 1.0_dp)))
    end function region_volume_fraction

end module fortbo_trust_region
