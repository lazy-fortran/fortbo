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
    use fortbo_history, only: fortbo_history_t, FORTBO_MISSING_IMPUTE_WORST, &
        FORTBO_OUTCOME_FAILED
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
    real(dp), parameter :: COMPLETION_PENDING_TOLERANCE = 1.0e-12_dp

    !! How the run is configured. Defaults describe TuRBO-1 with a batch of one.
    type :: fortbo_turbo_config_t
        !! `m` in TuRBO-`m`. One region is TuRBO-1.
        integer :: n_regions = 1
        !! `q`, the number of points returned by each `ask`.
        integer :: batch_size = 1
        !! Switch to one-point completion-driven ask/tell. In this mode
        !! `batch_size` must be one and `max_pending` bounds the in-flight
        !! evaluations; pending points are fantasized and excluded from the
        !! next candidate.
        logical :: completion_driven = .false.
        integer :: max_pending = 0
        !! Points drawn per region before its surrogate is trusted. Zero means
        !! `2*d`, the paper's rule.
        integer :: n_initial = 0
        real(dp) :: lengthscale = 0.2_dp
        !! Optional ARD lengthscales for the explicit Landreman parity lane.
        !! When absent, the established isotropic FortML adapter is used.
        real(dp), allocatable :: lengthscales(:)
        logical :: use_ard = .false.
        !! Select the fixed-hyperparameter inducing derivative adapter. This
        !! requires `use_ard` and caller-supplied unit-coordinate inducing
        !! points; absent this switch the established dense adapter remains the
        !! default.
        logical :: use_variational_derivative = .false.
        real(dp), allocatable :: inducing_points(:, :)
        real(dp) :: noise_variance = 1.0e-6_dp
        !! Use measured gradients when the history carries them. Left to the
        !! caller rather than forced, because requesting gradients from an
        !! empty history is an error.
        logical :: use_gradients = .false.
        !! Explicitly keep a value-only surrogate when histories contain
        !! gradients. The default is automatic derivative-observation use.
        logical :: ignore_gradients = .false.
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
        !! Optional caller-owned candidate pools for cross-language replay.
        !! Rows are unit-cube candidates and are scored by the normal posterior
        !! and acquisition path; supplying them does not bypass selection. For
        !! TuRBO-m, concatenate one equal-sized pool per region in region order.
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
        !! Number of points currently in flight in completion-driven mode.
        integer :: pending_count = 0
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
        if (config%completion_driven) then
            if (config%batch_size /= 1 .or. config%max_pending < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: completion-driven mode requires "// &
                    "batch_size=1 and max_pending>=1")
                return
            end if
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
        if (config%use_variational_derivative) then
            if (.not. config%use_ard .or. .not. allocated(config%inducing_points) .or. &
                    size(config%inducing_points, 1) < 1 .or. &
                    size(config%inducing_points, 2) /= n_inputs .or. &
                    any(config%inducing_points < 0.0_dp) .or. &
                    any(config%inducing_points > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: variational derivative mode requires "// &
                    "unit-coordinate inducing points and ARD lengthscales")
                return
            end if
        end if
        if (config%use_gradients .and. config%ignore_gradients) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: gradient-use and gradient-ignore modes conflict")
            return
        end if
        if (config%quasi_random .and. n_inputs > SOBOL_MAX_DIMENSION) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo turbo driver: seeded Sobol initial design exceeds "// &
                "the FortNum dimension limit")
            return
        end if
        if (allocated(config%frozen_candidates)) then
            if (size(config%frozen_candidates, 1) < config%batch_size*config%n_regions .or. &
                    mod(size(config%frozen_candidates, 1), config%n_regions) /= 0 .or. &
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
        self%pending_count = 0
        if (config%completion_driven) then
            allocate (self%pending_points(config%max_pending, n_inputs), &
                self%pending_region(config%max_pending))
            self%pending_points = 0.0_dp
            self%pending_region = 0
        end if
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
            ! The driver can receive failed evaluations through the optional
            ! `successful` tell argument. Store those rows as imputed worst
            ! cases so they remain in the checkpoint and trust trace, while
            ! `usable_count` keeps them out of surrogate training.
            self%histories(k)%missing_policy = FORTBO_MISSING_IMPUTE_WORST
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
        if (self%config%completion_driven .and. &
                self%pending_count >= self%config%max_pending) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: all completion-driven workers are busy")
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
            if (self%config%completion_driven) then
                k = next_completion_region_needing_design(self, required)
            else
                k = next_region_needing_design(self, required, assigned)
            end if
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
            if (self%config%completion_driven) then
                call register_completion_pending(self, points, regions, status)
                if (status%code /= FORTNUM_OK) return
            else
                call remember_pending(self, points, regions)
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        ! Every remaining slot goes through the pooled Thompson bandit.
        per_region = fortbo_candidate_count(self%n_inputs)
        if (allocated(self%config%frozen_candidates)) then
            per_region = size(self%config%frozen_candidates, 1)/size(self%regions)
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
                candidates = self%config%frozen_candidates(&
                    (k - 1)*per_region + 1:k*per_region, :)
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
            call fit_driver_posterior(self, scratch, lengthscales, posterior, status)
            if (status%code /= FORTNUM_OK) return
            if (self%config%completion_driven .and. &
                    completion_pending_count_for_region(self, k) > 0) then
                call add_pending_fantasies(self, k, scratch, posterior, status)
                if (status%code /= FORTNUM_OK) return
                call fit_driver_posterior(self, scratch, lengthscales, posterior, status)
                if (status%code /= FORTNUM_OK) return
            end if
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
                if (self%config%completion_driven .and. &
                        is_completion_pending(self, k, candidates(i, :))) then
                    realizations(offset, :) = huge(1.0_dp)
                    cycle
                end if
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
        if (self%config%completion_driven) then
            call register_completion_pending(self, points, regions, status)
            if (status%code /= FORTNUM_OK) return
        else
            call remember_pending(self, points, regions)
        end if
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
        integer :: n, i, row

        n = source%usable_count()
        shift = 0.0_dp
        scale = 1.0_dp
        call copy%initialize(source%n_inputs, 0, status)
        if (status%code /= FORTNUM_OK) return
        if (n < 1) return

        shift = 0.0_dp
        do row = 1, source%count
            if (.not. source%is_usable(row)) cycle
            shift = shift + source%objectives(row)
        end do
        shift = shift/real(n, dp)
        if (n > 1) then
            spread = 0.0_dp
            do row = 1, source%count
                if (.not. source%is_usable(row)) cycle
                spread = spread + (source%objectives(row) - shift)**2
            end do
            spread = spread/real(n - 1, dp)
            if (spread > 0.0_dp) scale = sqrt(spread)
        end if

        do i = 1, source%count
            if (.not. source%is_usable(i)) cycle
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

    !! Fit the configured posterior adapter. Keeping this branch in one place
    !! lets completion-driven mode refit after adding posterior-mean fantasies
    !! without giving the two ask paths different derivative semantics.
    subroutine fit_driver_posterior(self, history, lengthscales, posterior, status)
        class(fortbo_turbo_driver_t), intent(in) :: self
        type(fortbo_history_t), intent(in) :: history
        real(dp), intent(in) :: lengthscales(:)
        class(fortbo_posterior_t), allocatable, intent(out) :: posterior
        type(fortnum_status_t), intent(out) :: status

        if (self%config%use_variational_derivative) then
            if (self%config%use_gradients .or. self%config%ignore_gradients) then
                call fortbo_fit_from_history(history, posterior, status, &
                    noise_variance=self%config%noise_variance, &
                    use_gradients=self%config%use_gradients, &
                    lengthscales=lengthscales, &
                    inducing_points=self%config%inducing_points)
            else
                call fortbo_fit_from_history(history, posterior, status, &
                    noise_variance=self%config%noise_variance, &
                    lengthscales=lengthscales, &
                    inducing_points=self%config%inducing_points)
            end if
        else if (self%config%use_ard) then
            if (self%config%use_gradients .or. self%config%ignore_gradients) then
                call fortbo_fit_from_history(history, posterior, status, &
                    noise_variance=self%config%noise_variance, &
                    use_gradients=self%config%use_gradients, &
                    lengthscales=lengthscales)
            else
                call fortbo_fit_from_history(history, posterior, status, &
                    noise_variance=self%config%noise_variance, &
                    lengthscales=lengthscales)
            end if
        else
            if (self%config%use_gradients .or. self%config%ignore_gradients) then
                call fortbo_fit_from_history(history, posterior, status, &
                    lengthscale=self%config%lengthscale, &
                    noise_variance=self%config%noise_variance, &
                    use_gradients=self%config%use_gradients)
            else
                call fortbo_fit_from_history(history, posterior, status, &
                    lengthscale=self%config%lengthscale, &
                    noise_variance=self%config%noise_variance)
            end if
        end if
    end subroutine fit_driver_posterior

    !! Add posterior-mean fantasies only to the temporary fitting history. A
    !! pending evaluation remains absent from the durable history until tell;
    !! the fantasy merely prevents an asynchronous ask from seeing full
    !! uncertainty at a point already in flight.
    subroutine add_pending_fantasies(self, region, history, posterior, status)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: region
        type(fortbo_history_t), intent(inout) :: history
        class(fortbo_posterior_t), allocatable, intent(inout) :: posterior
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: points(:, :), mean(:), variance(:)
        integer :: n_pending, n_found, i

        n_pending = completion_pending_count_for_region(self, region)
        if (n_pending < 1) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        allocate (points(n_pending, self%n_inputs), mean(n_pending), &
            variance(n_pending))
        call completion_pending_points_for_region(self, region, points, n_found)
        if (n_found /= n_pending) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: pending-region count changed")
            return
        end if
        call posterior%moments(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n_pending
            call history%add(points(i, :), status, objective=mean(i))
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine add_pending_fantasies

    pure integer function completion_pending_count_for_region(self, region) &
            result(n)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: region
        integer :: i

        n = 0
        if (.not. allocated(self%pending_region)) return
        do i = 1, self%pending_count
            if (self%pending_region(i) == region) n = n + 1
        end do
    end function completion_pending_count_for_region

    subroutine completion_pending_points_for_region(self, region, points, n_found)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: region
        real(dp), intent(out) :: points(:, :)
        integer, intent(out) :: n_found
        integer :: i

        points = 0.0_dp
        n_found = 0
        do i = 1, self%pending_count
            if (self%pending_region(i) /= region) cycle
            n_found = n_found + 1
            points(n_found, :) = self%pending_points(i, :)
        end do
    end subroutine completion_pending_points_for_region

    pure logical function is_completion_pending(self, region, point) result(found)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: region
        real(dp), intent(in) :: point(:)
        integer :: i

        found = .false.
        if (.not. allocated(self%pending_region)) return
        if (size(point) /= self%n_inputs) return
        do i = 1, self%pending_count
            if (self%pending_region(i) /= region) cycle
            if (maxval(abs(self%pending_points(i, :) - point)) <= &
                    COMPLETION_PENDING_TOLERANCE) then
                found = .true.
                return
            end if
        end do
    end function is_completion_pending

    subroutine register_completion_pending(self, points, regions, status)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: regions(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n, i

        n = size(points, 1)
        if (n /= 1 .or. size(points, 2) /= self%n_inputs .or. &
                size(regions) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: completion asks must contain one point")
            return
        end if
        if (self%pending_count + n > self%config%max_pending) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: completion pending capacity exceeded")
            return
        end if
        do i = 1, n
            self%pending_count = self%pending_count + 1
            self%pending_points(self%pending_count, :) = points(i, :)
            self%pending_region(self%pending_count) = regions(i)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine register_completion_pending

    subroutine complete_completion_pending(self, points, regions, status)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: regions(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, match

        if (size(points, 1) /= 1 .or. size(points, 2) /= self%n_inputs .or. &
                size(regions) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: completion tells must contain one point")
            return
        end if
        match = 0
        do i = 1, self%pending_count
            if (self%pending_region(i) /= regions(1)) cycle
            if (maxval(abs(self%pending_points(i, :) - points(1, :))) <= &
                    COMPLETION_PENDING_TOLERANCE) then
                match = i
                exit
            end if
        end do
        if (match == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo driver: completion does not match a pending point")
            return
        end if
        do j = match, self%pending_count - 1
            self%pending_points(j, :) = self%pending_points(j + 1, :)
            self%pending_region(j) = self%pending_region(j + 1)
        end do
        self%pending_count = self%pending_count - 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine complete_completion_pending

    !! Completion-driven initial design counts points already in flight, so an
    !! asynchronous caller cannot overfill one region before its peers have
    !! received their initial observations.
    integer function next_completion_region_needing_design(self, required) &
            result(choice)
        class(fortbo_turbo_driver_t), intent(in) :: self
        integer, intent(in) :: required
        integer :: k, have, fewest

        choice = 0
        fewest = huge(1)
        do k = 1, size(self%regions)
            have = self%histories(k)%count + &
                completion_pending_count_for_region(self, k)
            if (have >= required) cycle
            if (have < fewest) then
                fewest = have
                choice = k
            end if
        end do
    end function next_completion_region_needing_design

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
    subroutine driver_tell(self, points, regions, values, status, gradients, &
            successful)
        class(fortbo_turbo_driver_t), intent(inout) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: regions(:)
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: gradients(:, :)
        logical, intent(in), optional :: successful(:)
        real(dp), allocatable :: batch_points(:, :), batch_values(:)
        integer :: k, i, n, count
        logical :: observation_ok

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
        if (present(successful)) then
            if (size(successful) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo driver: success mask does not match the batch")
                return
            end if
        end if
        if (self%config%completion_driven) then
            call complete_completion_pending(self, points, regions, status)
            if (status%code /= FORTNUM_OK) return
        end if

        do i = 1, n
            observation_ok = .true.
            if (present(successful)) observation_ok = successful(i)
            if (.not. observation_ok) then
                call self%histories(regions(i))%add(points(i, :), status, &
                    outcome=FORTBO_OUTCOME_FAILED)
            else if (present(gradients)) then
                call self%histories(regions(i))%add(points(i, :), status, &
                    objective=values(i), gradient=gradients(i, :))
            else
                call self%histories(regions(i))%add(points(i, :), status, &
                    objective=values(i))
            end if
            if (status%code /= FORTNUM_OK) return
            if (observation_ok .and. values(i) < self%best_value) then
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
