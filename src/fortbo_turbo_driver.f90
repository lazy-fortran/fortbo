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
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortnum_sobol, only: sobol_t, sobol_initialize, sobol_next, &
        SOBOL_MAX_DIMENSION
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_history, only: fortbo_history_t
    use fortbo_trust_region, only: fortbo_trust_region_t
    use fortbo_acquisition, only: fortbo_expected_improvement
    use fortbo_turbo, only: fortbo_turbo_candidates, fortbo_candidate_count, &
        fortbo_thompson_select, fortbo_qei_select
    use fortbo_fortml, only: fortbo_fit_from_history
    use fortbo_normal, only: fortbo_inverse_normal
    implicit none
    private

    public :: fortbo_turbo_config_t
    public :: fortbo_turbo_driver_t

    integer, parameter, public :: FORTBO_TURBO_ACQUISITION_TS = 0
    integer, parameter, public :: FORTBO_TURBO_ACQUISITION_EI = 1

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
        !! Optional ARD lengthscales for the explicit Landreman parity lane.
        !! When absent, the established isotropic FortML adapter is used.
        real(dp), allocatable :: lengthscales(:)
        logical :: use_ard = .false.
        real(dp) :: noise_variance = 1.0e-6_dp
        !! Use measured gradients when the history carries them. Left to the
        !! caller rather than inferred, because ignoring gradients is a
        !! legitimate choice and should be a visible one.
        logical :: use_gradients = .false.
        !! Thompson sampling is the historical default. EI is analytic at q=1
        !! and greedy Monte Carlo qEI at larger batch sizes.
        integer :: acquisition = FORTBO_TURBO_ACQUISITION_TS
        !! Trust-state tolerances. A zero failure tolerance selects the
        !! dimension/batch rule; nonzero values pin a replay configuration.
        integer :: success_tolerance = 3
        integer :: failure_tolerance = 0
        !! Relative improvement threshold used by the upstream Turbo state.
        real(dp) :: improvement_tolerance = 0.0_dp
        !! Draw initial-design and generated candidates from Sobol sequences
        !! rather than the generator. The initial design uses a seeded digital
        !! shift; exact upstream LMS/Owen scramble matching remains a separate
        !! replay requirement.
        logical :: quasi_random = .true.
        !! Optional caller-owned initial design for exact cross-language replay.
        !! When present, these rows take precedence over `quasi_random` and are
        !! consumed in ask order, including across multiple regions.
        real(dp), allocatable :: frozen_initial_design(:, :)
        !! Optional caller-owned candidate pool for cross-language replay. The
        !! rows are unit-cube candidates and are scored by the normal posterior
        !! and acquisition path; supplying them does not bypass selection.
        real(dp), allocatable :: frozen_candidates(:, :)
    end type fortbo_turbo_config_t

    type :: fortbo_turbo_driver_t
        type(fortbo_turbo_config_t) :: config
        integer :: n_inputs = 0
        type(fortbo_trust_region_t), allocatable :: regions(:)
        !! One history per region. Regions must not share data: a region's
        !! surrogate is a *local* model, and pooling would defeat the point.
        type(fortbo_history_t), allocatable :: histories(:)
        type(rng_t) :: generator
        type(sobol_t) :: initial_sequence
        real(dp), allocatable :: initial_shift(:)
        logical :: initial_quasi_ready = .false.
        integer :: initial_frozen_index = 0
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
        integer :: k, required

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
        required = config%n_initial
        if (required == 0) required = 2*n_inputs
        if (allocated(config%frozen_initial_design)) then
            if (size(config%frozen_initial_design, 1) < required*config%n_regions .or. &
                    size(config%frozen_initial_design, 2) /= n_inputs .or. &
                    any(config%frozen_initial_design < 0.0_dp) .or. &
                    any(config%frozen_initial_design > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: frozen initial design is invalid")
                return
            end if
        end if
        if (config%use_ard) then
            if (.not. allocated(config%lengthscales) .or. &
                    size(config%lengthscales) /= n_inputs .or. &
                    any(config%lengthscales <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: ARD lengthscales must match inputs")
                return
            end if
        end if
        if (config%quasi_random .and. n_inputs > SOBOL_MAX_DIMENSION) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo turbo driver: seeded Sobol initial design exceeds "// &
                "the FortNum dimension limit")
            return
        end if
        if (allocated(config%frozen_candidates)) then
            if (size(config%frozen_candidates, 1) < config%batch_size .or. &
                    size(config%frozen_candidates, 2) /= n_inputs .or. &
                    any(config%frozen_candidates < 0.0_dp) .or. &
                    any(config%frozen_candidates > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: frozen candidate pool is invalid")
                return
            end if
        end if
        if (config%acquisition < FORTBO_TURBO_ACQUISITION_TS .or. &
                config%acquisition > FORTBO_TURBO_ACQUISITION_EI) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: unknown acquisition mode")
            return
        end if
        if (config%acquisition == FORTBO_TURBO_ACQUISITION_EI .and. &
                config%batch_size > 1 .and. .not. config%use_ard) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo turbo driver: batch qEI requires an ARD joint-sample posterior")
            return
        end if
        if (config%success_tolerance < 1 .or. config%failure_tolerance < 0 .or. &
                config%improvement_tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: invalid trust-state tolerance")
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
            self%regions(k)%success_tolerance = config%success_tolerance
            if (config%failure_tolerance > 0) then
                self%regions(k)%failure_tolerance = config%failure_tolerance
            end if
            self%regions(k)%improvement_tolerance = config%improvement_tolerance
            call self%histories(k)%initialize(n_inputs, 0, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call rng_seed(self%generator, int(seed, kind(1_8)), status)
        if (status%code /= FORTNUM_OK) return
        if (config%quasi_random .and. .not. &
                allocated(config%frozen_initial_design)) then
            call sobol_initialize(self%initial_sequence, n_inputs, status)
            if (status%code /= FORTNUM_OK) return
            allocate (self%initial_shift(n_inputs))
            block
                type(rng_t) :: scramble_generator
                integer :: j

                call rng_seed(scramble_generator, int(seed, kind(1_8)), status)
                if (status%code /= FORTNUM_OK) return
                do j = 1, n_inputs
                    call rng_uniform(scramble_generator, self%initial_shift(j))
                end do
            end block
            self%initial_quasi_ready = .true.
        end if
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
        real(dp), allocatable :: joint_samples(:, :)
        real(dp), allocatable :: mean(:), variance(:), draw(:)
        integer, allocatable :: pooled_region(:), chosen(:), assigned(:)
        real(dp) :: uniform, acquisition_value, incumbent
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

        ! A region short of its initial design proposes seeded digital-shifted
        ! Sobol points when quasi-random mode is enabled. Regions are served
        ! round-robin rather than one at a time, so that with several
        ! regions the designs advance together and the bandit starts with
        ! comparable evidence about each.
        filled = 0
        allocate (draw(self%n_inputs))
        allocate (assigned(size(self%regions)))
        assigned = 0
        do while (filled < self%config%batch_size)
            k = next_region_needing_design(self, required, assigned)
            if (k == 0) exit
            if (allocated(self%config%frozen_initial_design)) then
                if (self%initial_frozen_index >= &
                        size(self%config%frozen_initial_design, 1)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "fortbo turbo driver: frozen initial design is exhausted")
                    return
                end if
                self%initial_frozen_index = self%initial_frozen_index + 1
                draw = self%config%frozen_initial_design(&
                    self%initial_frozen_index, :)
            else if (self%initial_quasi_ready) then
                call sobol_next(self%initial_sequence, draw, status)
                if (status%code /= FORTNUM_OK) return
                draw = modulo(draw + self%initial_shift, 1.0_dp)
            else
                do j = 1, self%n_inputs
                    call rng_uniform(self%generator, uniform)
                    draw(j) = uniform
                end do
            end if
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
        if (allocated(self%config%frozen_candidates)) then
            per_region = size(self%config%frozen_candidates, 1)
        end if
        allocate (lengthscales(self%n_inputs))
        if (self%config%use_ard) then
            lengthscales = self%config%lengthscales
        else
            lengthscales = self%config%lengthscale
        end if
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
            if (allocated(self%config%frozen_candidates)) then
                candidates = self%config%frozen_candidates
                do i = 1, per_region
                    if (.not. self%regions(k)%contains_point(candidates(i, :), &
                            lengthscales)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo turbo driver: frozen candidate lies outside trust region")
                        return
                    end if
                end do
            else
                call fortbo_turbo_candidates(self%regions(k), lengthscales, &
                    self%generator, candidates, status, &
                    quasi_random=self%config%quasi_random)
            end if
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
            if (self%config%use_ard) then
                call fortbo_fit_from_history(scratch, posterior, status, &
                    noise_variance=self%config%noise_variance, &
                    use_gradients=self%config%use_gradients, &
                    lengthscales=lengthscales)
            else
                call fortbo_fit_from_history(scratch, posterior, status, &
                    lengthscale=self%config%lengthscale, &
                    noise_variance=self%config%noise_variance, &
                    use_gradients=self%config%use_gradients)
            end if
            if (status%code /= FORTNUM_OK) return
            call posterior%moments(candidates, mean, variance, status)
            if (status%code /= FORTNUM_OK) return
            mean = shift + scale*mean
            variance = variance*scale*scale
            if (allocated(joint_samples)) deallocate (joint_samples)
            if (self%config%use_ard .and. &
                    (self%config%acquisition == FORTBO_TURBO_ACQUISITION_TS .or. &
                    (self%config%acquisition == FORTBO_TURBO_ACQUISITION_EI .and. &
                    self%config%batch_size > 1))) then
                allocate (joint_samples(per_region, needed))
                call posterior%joint_sample(candidates, self%generator, &
                    joint_samples, status)
                if (status%code /= FORTNUM_OK) return
            end if
            do i = 1, per_region
                offset = offset + 1
                pooled(offset, :) = candidates(i, :)
                pooled_region(offset) = k
                if (self%config%acquisition == FORTBO_TURBO_ACQUISITION_EI .and. &
                        self%config%batch_size == 1) then
                    ! Landreman's production q=1 path is analytic EI. The
                    ! pooled reduction is a no-replacement arg-max of EI,
                    ! represented as a negated score for the common selector.
                    incumbent = self%histories(k)%objectives(&
                        self%histories(k)%best_index())
                    call fortbo_expected_improvement(mean(i), &
                        sqrt(max(variance(i), 0.0_dp)), incumbent, 0.0_dp, &
                        acquisition_value)
                    realizations(offset, :) = -acquisition_value
                else
                    ! Posterior realizations, deliberately left in the
                    ! objective's own units so that a region believing in
                    ! better values wins more of the batch.
                    do slot = 1, needed
                        if (self%config%use_ard) then
                            realizations(offset, slot) = shift + scale* &
                                joint_samples(i, slot)
                        else
                            call rng_uniform(self%generator, uniform)
                            realizations(offset, slot) = mean(i) &
                                + sqrt(max(variance(i), 0.0_dp))* &
                                fortbo_inverse_normal(uniform)
                        end if
                    end do
                end if
            end do
        end do

        if (offset == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: no region has a surrogate yet")
            return
        end if

        allocate (chosen(needed))
        if (self%config%acquisition == FORTBO_TURBO_ACQUISITION_EI) then
            call fortbo_qei_select(realizations(:offset, :), self%best_value, &
                chosen, status)
        else
            call fortbo_thompson_select(realizations(:offset, :), chosen, status)
        end if
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

end module fortbo_turbo_driver
