module fortbo_turbo_driver
    !! TuRBO-1 and TuRBO-`m` as a running policy (ROADMAP BO3T).
    !!
    !! The trust-region bookkeeping, candidate generation, and Thompson
    !! selection each already stand alone and are tested on their own. This
    !! module is what makes them a *method*: it owns `m` simultaneous regions,
    !! fits one local surrogate per region, pools their Thompson realizations to
    !! form a batch, feeds the outcome back to the right region, and restarts a
    !! region whose trust region has collapsed.
    !!
    !! The interface is ask/tell rather than a callback loop. FortBO does not
    !! own the objective: a run may evaluate on a cluster, in a queue, or by
    !! hand, and the roadmap's asynchronous-worker item needs exactly this shape
    !! anyway. `ask` returns a batch and says which region each point came from;
    !! `tell` takes the observed values back.
    !!
    !! **The bandit across regions is implicit.** There is no explicit arm
    !! selection, no UCB over regions, no counter deciding who gets the next
    !! evaluation. Every region proposes candidates, every candidate carries one
    !! posterior realization from its *own* local surrogate, and the batch takes
    !! the `q` smallest realizations across the pooled set. A region whose
    !! posterior is genuinely promising therefore wins more of the batch, and
    !! one whose local model has grown pessimistic quietly stops being sampled
    !! without ever being switched off. That is the mechanism in the TuRBO paper
    !! (Eriksson et al., NeurIPS 2019), and it is worth stating plainly because
    !! an explicit bandit is the obvious thing to write instead and would behave
    !! differently: it would allocate on estimated regret rather than on the
    !! posterior's own belief about which point is best.
    !!
    !! Realizations are comparable across regions only because they are drawn
    !! from posteriors over the *same* objective in the same units. They are not
    !! standardized per region, and standardizing them would break the bandit by
    !! erasing the very difference in level that says one region is better.
    !!
    !! Derivative observations flow through untouched: the per-region surrogate
    !! comes from `fortbo_fit_from_history`, which chooses the derivative model
    !! when the history carries gradients. TuRBO with gradient observations is
    !! DTuRBO mode 1 and needs no separate code path here.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_history, only: fortbo_history_t
    use fortbo_trust_region, only: fortbo_trust_region_t
    use fortbo_turbo, only: fortbo_turbo_candidates, fortbo_candidate_count, &
        fortbo_thompson_select
    use fortbo_fortml, only: fortbo_fit_from_history
    implicit none
    private

    public :: fortbo_turbo_config_t
    public :: fortbo_turbo_driver_t

    !! How the run is configured. Defaults describe TuRBO-1 with a batch of one.
    type :: fortbo_turbo_config_t
        !! `m` in TuRBO-`m`. One region is TuRBO-1.
        integer :: n_regions = 1
        !! `q`, the number of points returned by each `ask`.
        integer :: batch_size = 1
        !! Points drawn per region before its surrogate is trusted. Zero means
        !! `2*d`, the paper's rule.
        integer :: n_initial = 0
        real(dp) :: lengthscale = 0.2_dp
        real(dp) :: noise_variance = 1.0e-6_dp
        !! Use measured gradients when the history carries them. Left to the
        !! caller rather than inferred, because ignoring gradients is a
        !! legitimate choice and should be a visible one.
        logical :: use_gradients = .false.
        !! Draw candidates from a Sobol sequence rather than the generator.
        logical :: quasi_random = .true.
    end type fortbo_turbo_config_t

    type :: fortbo_turbo_driver_t
        type(fortbo_turbo_config_t) :: config
        integer :: n_inputs = 0
        type(fortbo_trust_region_t), allocatable :: regions(:)
        !! One history per region. Regions must not share data: a region's
        !! surrogate is a *local* model, and pooling would defeat the point.
        type(fortbo_history_t), allocatable :: histories(:)
        type(rng_t) :: generator
        integer :: evaluations = 0
        integer :: restarts = 0
        logical :: started = .false.
        !! Best value seen anywhere, across all regions and all restarts.
        real(dp) :: best_value = huge(1.0_dp)
        real(dp), allocatable :: best_point(:)
        !! Which region each point of the last `ask` came from.
        integer, allocatable :: pending_region(:)
        real(dp), allocatable :: pending_points(:, :)
    contains
        procedure, public :: initialize => driver_initialize
        procedure, public :: ask => driver_ask
        procedure, public :: tell => driver_tell
        procedure, public :: active_regions => driver_active_regions
    end type fortbo_turbo_driver_t

contains

    subroutine driver_initialize(self, n_inputs, config, seed, status)
        class(fortbo_turbo_driver_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortbo_turbo_config_t), intent(in) :: config
        integer, intent(in) :: seed
        type(fortnum_status_t), intent(out) :: status
        integer :: k

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: n_inputs must be positive")
            return
        end if
        if (config%n_regions < 1 .or. config%batch_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: region and batch counts must be positive")
            return
        end if
        if (config%n_initial < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: initial count must not be negative")
            return
        end if

        self%config = config
        self%n_inputs = n_inputs
        allocate (self%regions(config%n_regions))
        allocate (self%histories(config%n_regions))
        allocate (self%best_point(n_inputs))
        self%best_point = 0.5_dp
        do k = 1, config%n_regions
            call self%regions(k)%initialize(n_inputs, config%batch_size, status)
            if (status%code /= FORTNUM_OK) return
            call self%histories(k)%initialize(n_inputs, 0, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call rng_seed(self%generator, int(seed, kind(1_8)), status)
        if (status%code /= FORTNUM_OK) return
        self%started = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine driver_initialize

    pure integer function driver_active_regions(self) result(count)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer :: k

        count = 0
        if (.not. allocated(self%regions)) return
        do k = 1, size(self%regions)
            if (self%regions(k)%active) count = count + 1
        end do
    end function driver_active_regions

    !! Propose the next batch.
    !!
    !! A region still filling its initial design proposes uniform points and
    !! bypasses the bandit entirely — it has no surrogate yet, so a realization
    !! from it would be a draw from the prior and would win or lose the batch
    !! for reasons unrelated to the objective. Once every region has a
    !! surrogate, the pooled Thompson selection decides.
    subroutine driver_ask(self, points, regions, status)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(out) :: points(:, :)
        integer, intent(out) :: regions(:)
        type(fortnum_status_t), intent(out) :: status
        class(fortbo_posterior_t), allocatable :: posterior
        type(fortbo_history_t) :: scratch
        real(dp) :: shift, scale
        real(dp), allocatable :: candidates(:, :), lengthscales(:)
        real(dp), allocatable :: pooled(:, :), realizations(:, :)
        real(dp), allocatable :: mean(:), variance(:), draw(:)
        integer, allocatable :: pooled_region(:), chosen(:), assigned(:)
        real(dp) :: uniform
        integer :: k, i, j, per_region, total, filled, required, offset
        integer :: needed, slot

        points = 0.0_dp
        regions = 0
        if (.not. self%started) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: driver is not initialized")
            return
        end if
        if (size(points, 1) /= self%config%batch_size .or. &
            size(points, 2) /= self%n_inputs .or. &
            size(regions) /= self%config%batch_size) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: batch shape does not match the configuration")
            return
        end if
        required = self%config%n_initial
        if (required == 0) required = 2*self%n_inputs

        ! A region short of its initial design proposes uniform points. Regions
        ! are served round-robin rather than one at a time, so that with several
        ! regions the designs advance together and the bandit starts with
        ! comparable evidence about each.
        filled = 0
        allocate (draw(self%n_inputs))
        allocate (assigned(size(self%regions)))
        assigned = 0
        do while (filled < self%config%batch_size)
            k = next_region_needing_design(self, required, assigned)
            if (k == 0) exit
            do j = 1, self%n_inputs
                call rng_uniform(self%generator, uniform)
                draw(j) = uniform
            end do
            filled = filled + 1
            points(filled, :) = draw
            regions(filled) = k
            assigned(k) = assigned(k) + 1
        end do
        if (filled == self%config%batch_size) then
            call remember_pending(self, points, regions)
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        ! Every remaining slot goes through the pooled Thompson bandit.
        per_region = fortbo_candidate_count(self%n_inputs)
        allocate (lengthscales(self%n_inputs))
        lengthscales = self%config%lengthscale
        allocate (candidates(per_region, self%n_inputs))
        total = per_region*size(self%regions)
        allocate (pooled(total, self%n_inputs), pooled_region(total))
        ! One independent realization column per remaining batch slot. Each
        ! slot picks its own arg-min, which is what makes a TuRBO batch diverse
        ! rather than q copies of the same greedy point.
        needed = self%config%batch_size - filled
        allocate (realizations(total, needed))
        allocate (mean(per_region), variance(per_region))

        offset = 0
        do k = 1, size(self%regions)
            if (self%histories(k)%count < required) then
                ! Still in its initial design; contributes nothing to the pool.
                cycle
            end if
            ! A region whose design has just completed is placed now, centered
            ! on the best point it found. Placement is what makes it active;
            ! before it, there is nothing for a trust region to be centered on.
            if (.not. self%regions(k)%active) then
                call place_region(self, k, status)
                if (status%code /= FORTNUM_OK) return
            end if
            call fortbo_turbo_candidates(self%regions(k), lengthscales, &
                self%generator, candidates, status, &
                quasi_random=self%config%quasi_random)
            if (status%code /= FORTNUM_OK) return
            ! Standardize the region's observations before fitting, then map
            ! the moments back. This is not cosmetic. A GP with a zero mean
            ! function reverts to zero away from its data, so a region whose
            ! observations are all *bad* would extrapolate to a prior claiming
            ! that zero is typical — better than anything it has actually seen —
            ! and its realizations would undercut a region sitting on genuinely
            ! good values. Measured on a two-region run with one design at the
            ! optimum and one in the far corner, the unstandardized driver sent
            ! 31 of 50 proposals to the *worse* region. Standardizing per region
            ! makes the prior say "typical for here", which is what the trust
            ! region means, and the comparison across regions is restored by
            ! mapping back into the objective's own units.
            call standardized_copy(self%histories(k), scratch, shift, scale, status)
            if (status%code /= FORTNUM_OK) return
            call fortbo_fit_from_history(scratch, posterior, status, &
                lengthscale=self%config%lengthscale, &
                noise_variance=self%config%noise_variance, &
                use_gradients=self%config%use_gradients)
            if (status%code /= FORTNUM_OK) return
            call posterior%moments(candidates, mean, variance, status)
            if (status%code /= FORTNUM_OK) return
            mean = shift + scale*mean
            variance = variance*scale*scale
            do i = 1, per_region
                offset = offset + 1
                pooled(offset, :) = candidates(i, :)
                pooled_region(offset) = k
                ! Posterior realizations, deliberately left in the objective's
                ! own units so that a region believing in better values wins
                ! more of the batch.
                do slot = 1, needed
                    call rng_uniform(self%generator, uniform)
                    realizations(offset, slot) = mean(i) &
                        + sqrt(max(variance(i), 0.0_dp))*inverse_normal(uniform)
                end do
            end do
        end do

        if (offset == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: no region has a surrogate yet")
            return
        end if

        allocate (chosen(needed))
        call fortbo_thompson_select(realizations(:offset, :), chosen, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(chosen)
            filled = filled + 1
            points(filled, :) = pooled(chosen(i), :)
            regions(filled) = pooled_region(chosen(i))
        end do
        call remember_pending(self, points, regions)
        call status_set(status, FORTNUM_OK, "")
    end subroutine driver_ask

    !! A copy of `source` with its objectives standardized to zero mean and unit
    !! standard deviation, together with the shift and scale needed to undo it.
    !!
    !! Gradients are divided by the same scale, since a gradient of the
    !! standardized objective is the original gradient over that scale. A
    !! constant history has no spread; a scale of one is used there, which
    !! leaves the values centered and changes nothing else.
    subroutine standardized_copy(source, copy, shift, scale, status)
        type(fortbo_history_t), intent(in) :: source
        type(fortbo_history_t), intent(out) :: copy
        real(dp), intent(out) :: shift
        real(dp), intent(out) :: scale
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: spread
        integer :: n, i

        n = source%count
        shift = 0.0_dp
        scale = 1.0_dp
        call copy%initialize(source%n_inputs, 0, status)
        if (status%code /= FORTNUM_OK) return
        if (n < 1) return

        shift = sum(source%objectives(:n))/real(n, dp)
        if (n > 1) then
            spread = sum((source%objectives(:n) - shift)**2)/real(n - 1, dp)
            if (spread > 0.0_dp) scale = sqrt(spread)
        end if

        do i = 1, n
            if (source%has_gradient(i)) then
                call copy%add(source%inputs(i, :), status, &
                    objective=(source%objectives(i) - shift)/scale, &
                    gradient=source%gradients(i, :)/scale)
            else
                call copy%add(source%inputs(i, :), status, &
                    objective=(source%objectives(i) - shift)/scale)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine standardized_copy

    !! The next region still short of its initial design, counting points
    !! already assigned in this batch. Returns zero when every design is
    !! complete. Round-robin, so no region monopolizes a batch.
    pure integer function next_region_needing_design(self, required, assigned) &
            result(choice)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: required
        integer, intent(in) :: assigned(:)
        integer :: k, have, fewest

        choice = 0
        fewest = huge(1)
        do k = 1, size(self%regions)
            have = self%histories(k)%count + assigned(k)
            if (have >= required) cycle
            if (have < fewest) then
                fewest = have
                choice = k
            end if
        end do
    end function next_region_needing_design

    !! Center a region on the best point of its initial design.
    subroutine place_region(self, k, status)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        integer, intent(in) :: k
        type(fortnum_status_t), intent(out) :: status
        integer :: best

        best = self%histories(k)%best_index()
        if (best < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: region has no usable design point")
            return
        end if
        call self%regions(k)%restart(self%histories(k)%inputs(best, :), &
            self%histories(k)%objectives(best), status)
    end subroutine place_region

    subroutine remember_pending(self, points, regions)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: regions(:)

        if (allocated(self%pending_points)) deallocate (self%pending_points)
        if (allocated(self%pending_region)) deallocate (self%pending_region)
        allocate (self%pending_points, source=points)
        allocate (self%pending_region, source=regions)
    end subroutine remember_pending

    !! Record the outcome of the last `ask`.
    !!
    !! Each observation goes to the history and the trust region of the region
    !! that proposed it. A region whose trust region has collapsed is restarted
    !! from a fresh center with an empty history: restarting while keeping the
    !! old data would rebuild the same collapsed model and collapse again.
    subroutine driver_tell(self, points, regions, values, status, gradients)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: regions(:)
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: gradients(:, :)
        real(dp), allocatable :: batch_points(:, :), batch_values(:)
        integer :: k, i, n, count

        n = size(values)
        if (.not. self%started) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: driver is not initialized")
            return
        end if
        if (size(points, 1) /= n .or. size(regions) /= n .or. &
            size(points, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: observation shapes disagree")
            return
        end if
        if (any(regions < 1) .or. any(regions > size(self%regions))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: region index out of range")
            return
        end if
        if (present(gradients)) then
            if (size(gradients, 1) /= n .or. size(gradients, 2) /= self%n_inputs) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: gradient shape does not match the batch")
                return
            end if
        end if

        do i = 1, n
            if (present(gradients)) then
                call self%histories(regions(i))%add(points(i, :), status, &
                    objective=values(i), gradient=gradients(i, :))
            else
                call self%histories(regions(i))%add(points(i, :), status, &
                    objective=values(i))
            end if
            if (status%code /= FORTNUM_OK) return
            if (values(i) < self%best_value) then
                self%best_value = values(i)
                self%best_point = points(i, :)
            end if
        end do
        self%evaluations = self%evaluations + n

        ! Feed each region the part of the batch that belongs to it, in one
        ! call, because the success and failure counters are defined per batch
        ! rather than per point.
        do k = 1, size(self%regions)
            count = count_for_region(regions, k)
            if (count == 0) cycle
            ! A region still filling its initial design has no center yet, so
            ! there is nothing for the success and failure counters to be
            ! measured against. Those points are data, not a batch outcome.
            if (.not. self%regions(k)%active) cycle
            allocate (batch_points(count, self%n_inputs), batch_values(count))
            call gather_for_region(points, values, regions, k, batch_points, &
                batch_values)
            call self%regions(k)%observe_batch(batch_points, batch_values, status)
            deallocate (batch_points, batch_values)
            if (status%code /= FORTNUM_OK) return
            if (.not. self%regions(k)%active) then
                call restart_region(self, k, status)
                if (status%code /= FORTNUM_OK) return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine driver_tell

    !! A collapsed region restarts at a uniformly drawn center with no data.
    !! Keeping the old history would rebuild the model that just collapsed.
    subroutine restart_region(self, k, status)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        integer, intent(in) :: k
        type(fortnum_status_t), intent(out) :: status

        ! Clear the data and leave the region unplaced. It will draw a fresh
        ! initial design and be re-centered on the best point of that design.
        ! Restarting onto the old history would rebuild the model that just
        ! collapsed, and restarting onto an arbitrary center would throw away
        ! the design the run is about to pay for anyway.
        call self%histories(k)%initialize(self%n_inputs, 0, status)
        if (status%code /= FORTNUM_OK) return
        self%restarts = self%restarts + 1
    end subroutine restart_region

    pure integer function count_for_region(regions, k) result(matches)
        integer, intent(in) :: regions(:)
        integer, intent(in) :: k

        matches = count(regions == k)
    end function count_for_region

    pure subroutine gather_for_region(points, values, regions, k, batch_points, &
            batch_values)
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(in) :: values(:)
        integer, intent(in) :: regions(:)
        integer, intent(in) :: k
        real(dp), intent(out) :: batch_points(:, :)
        real(dp), intent(out) :: batch_values(:)
        integer :: i, filled

        filled = 0
        do i = 1, size(values)
            if (regions(i) /= k) cycle
            filled = filled + 1
            batch_points(filled, :) = points(i, :)
            batch_values(filled) = values(i)
        end do
    end subroutine gather_for_region

    !! Inverse standard normal CDF by Acklam's rational approximation, accurate
    !! to about 1.15e-9 relative across the whole range. Used only to turn a
    !! uniform draw into a posterior realization, where that accuracy is far
    !! beyond what the selection can resolve.
    pure real(dp) function inverse_normal(p) result(z)
        real(dp), intent(in) :: p
        real(dp), parameter :: a(6) = [-3.969683028665376e+01_dp, &
            2.209460984245205e+02_dp, -2.759285104469687e+02_dp, &
            1.383577518672690e+02_dp, -3.066479806614716e+01_dp, &
            2.506628277459239e+00_dp]
        real(dp), parameter :: b(5) = [-5.447609879822406e+01_dp, &
            1.615858368580409e+02_dp, -1.556989798598866e+02_dp, &
            6.680131188771972e+01_dp, -1.328068155288572e+01_dp]
        real(dp), parameter :: c(6) = [-7.784894002430293e-03_dp, &
            -3.223964580411365e-01_dp, -2.400758277161838e+00_dp, &
            -2.549732539343734e+00_dp, 4.374664141464968e+00_dp, &
            2.938163982698783e+00_dp]
        real(dp), parameter :: d(4) = [7.784695709041462e-03_dp, &
            3.224671290700398e-01_dp, 2.445134137142996e+00_dp, &
            3.754408661907416e+00_dp]
        real(dp), parameter :: split_low = 0.02425_dp
        real(dp) :: q, r, clamped

        clamped = min(max(p, 1.0e-300_dp), 1.0_dp - 1.0e-16_dp)
        if (clamped < split_low) then
            q = sqrt(-2.0_dp*log(clamped))
            z = (((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6))/ &
                ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0_dp)
        else if (clamped <= 1.0_dp - split_low) then
            q = clamped - 0.5_dp
            r = q*q
            z = (((((a(1)*r + a(2))*r + a(3))*r + a(4))*r + a(5))*r + a(6))*q/ &
                (((((b(1)*r + b(2))*r + b(3))*r + b(4))*r + b(5))*r + 1.0_dp)
        else
            q = sqrt(-2.0_dp*log(1.0_dp - clamped))
            z = -(((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6))/ &
                ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0_dp)
        end if
    end function inverse_normal

end module fortbo_turbo_driver
