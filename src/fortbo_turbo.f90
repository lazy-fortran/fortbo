module fortbo_turbo
    !! TuRBO candidate generation and Thompson selection (ROADMAP BO3T).
    !!
    !! Candidates. Within a trust region, TuRBO does not sample the region
    !! uniformly. It starts from the region center and perturbs each coordinate
    !! only with probability `p = min(1, 20/d)`, forcing at least one coordinate
    !! to move. In twenty dimensions or fewer that is an ordinary uniform draw
    !! inside the region; above twenty it becomes a sparse perturbation, and
    !! that sparsity is the whole reason the method survives in hundreds of
    !! dimensions. A dense draw in 200 dimensions lands, with overwhelming
    !! probability, in a corner of the region far from the center and from every
    !! other candidate; the surrogate has no information there and the batch is
    !! wasted. Perturbing about twenty coordinates keeps candidates close enough
    !! to the incumbent for the local model to mean something.
    !!
    !! The perturbation magnitudes come from a Sobol sequence, so the candidate
    !! cloud covers the region evenly instead of clumping. The perturbation
    !! *mask* comes from the pseudorandom stream, because which coordinates move
    !! is a Bernoulli draw and has no low-discrepancy structure to exploit.
    !!
    !! Selection. One posterior realization is drawn per region, the
    !! realizations are concatenated across all regions, and the batch takes the
    !! minimizers. That single rule is simultaneously the within-region
    !! acquisition and the across-region bandit: a region whose posterior looks
    !! promising wins more of the batch, without any explicit bandit
    !! bookkeeping. Splitting it into two heuristics would lose exactly the
    !! coupling that makes it work.
    !!
    !! Selection is over a discrete candidate set, so it has no derivative. That
    !! is stated as a typed refusal rather than left for a caller to discover.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_uniform
    use fortnum_sobol, only: sobol_t, sobol_initialize, sobol_next, &
        SOBOL_MAX_DIMENSION
    use fortbo_trust_region, only: fortbo_trust_region_t
    implicit none
    private

    public :: fortbo_candidate_count
    public :: fortbo_perturbation_probability
    public :: fortbo_turbo_candidates
    public :: fortbo_thompson_select
    public :: fortbo_qei_select
    public :: fortbo_thompson_gradient_refusal

    !! Coordinates expected to move per candidate, from the reference
    !! implementation's `min(20/d, 1)`.
    integer, parameter, public :: FORTBO_TURBO_PERTURBED_COORDINATES = 20

contains

    !! `n_cand = min(5000, max(2000, 200 d))` for the Landreman replay.
    pure integer function fortbo_candidate_count(n_inputs) result(count)
        integer, intent(in) :: n_inputs

        count = min(5000, max(2000, 200*max(n_inputs, 1)))
    end function fortbo_candidate_count

    !! `p = min(1, 20/d)`.
    pure real(dp) function fortbo_perturbation_probability(n_inputs) result(p)
        integer, intent(in) :: n_inputs

        p = 1.0_dp
        if (n_inputs > FORTBO_TURBO_PERTURBED_COORDINATES) then
            p = real(FORTBO_TURBO_PERTURBED_COORDINATES, dp)/real(n_inputs, dp)
        end if
    end function fortbo_perturbation_probability

    !! Fill `candidates(n_cand, d)` with perturbations of the region center.
    !!
    !! `quasi_random` selects the source of the perturbation magnitudes. The
    !! default is the Sobol sequence, which is what the reference implementation
    !! uses, and it now reaches the dimensions this method exists for: FortNum
    !! computes its primitive polynomials rather than tabulating them, so
    !! dimensions into the hundreds are covered. Past the enumeration limit this
    !! still refuses by name instead of quietly substituting pseudorandom
    !! points, because substituting would change the sampling properties
    !! invisibly in exactly the regime that matters.
    subroutine fortbo_turbo_candidates(region, lengthscales, generator, candidates, &
            status, quasi_random, sequence)
        type(fortbo_trust_region_t), intent(in) :: region
        real(dp), intent(in) :: lengthscales(:)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: candidates(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: quasi_random
        type(sobol_t), intent(inout), optional :: sequence
        type(sobol_t) :: local_sequence
        real(dp), allocatable :: lower(:), upper(:), draw(:)
        real(dp) :: probability, uniform
        integer :: n_inputs, n_candidates, i, j, forced
        logical :: use_quasi, any_perturbed

        n_inputs = region%n_inputs
        n_candidates = size(candidates, 1)
        if (n_inputs < 1 .or. .not. region%active) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: region is not active")
            return
        end if
        if (size(candidates, 2) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: candidate width does not match the region")
            return
        end if
        if (n_candidates < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: candidate count must be positive")
            return
        end if

        use_quasi = .true.
        if (present(quasi_random)) use_quasi = quasi_random
        if (use_quasi .and. n_inputs > SOBOL_MAX_DIMENSION) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo turbo: fortnum_sobol enumerates primitive polynomials "// &
                "only to degree 13; pass quasi_random=.false. explicitly to "// &
                "accept pseudorandom perturbations")
            return
        end if

        allocate (lower(n_inputs), upper(n_inputs), draw(n_inputs))
        call region%bounds(lengthscales, lower, upper, status)
        if (status%code /= FORTNUM_OK) return

        if (use_quasi .and. .not. present(sequence)) then
            call sobol_initialize(local_sequence, n_inputs, status)
            if (status%code /= FORTNUM_OK) return
        end if

        probability = fortbo_perturbation_probability(n_inputs)
        do i = 1, n_candidates
            if (use_quasi) then
                if (present(sequence)) then
                    call sobol_next(sequence, draw, status)
                else
                    call sobol_next(local_sequence, draw, status)
                end if
                if (status%code /= FORTNUM_OK) return
            else
                do j = 1, n_inputs
                    call rng_uniform(generator, draw(j))
                end do
            end if

            candidates(i, :) = region%center
            any_perturbed = .false.
            do j = 1, n_inputs
                call rng_uniform(generator, uniform)
                if (uniform >= probability) cycle
                candidates(i, j) = lower(j) + draw(j)*(upper(j) - lower(j))
                any_perturbed = .true.
            end do

            ! At least one coordinate must move, or the candidate is the center
            ! again and the batch wastes an evaluation re-measuring a known
            ! point. The forced coordinate is chosen from the same stream so the
            ! whole construction stays replayable.
            if (.not. any_perturbed) then
                call rng_uniform(generator, uniform)
                forced = min(int(uniform*real(n_inputs, dp)) + 1, n_inputs)
                candidates(i, forced) = lower(forced) &
                    + draw(forced)*(upper(forced) - lower(forced))
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_turbo_candidates

    !! Select a batch of `q` candidates from posterior realizations.
    !!
    !! `samples(c, s)` is realization `s` evaluated at candidate `c`, with the
    !! candidates of every region concatenated along the first axis. For each
    !! realization the smallest value wins, and an already-selected candidate is
    !! skipped so the batch holds distinct points. Ties go to the lowest index,
    !! which is what makes a replayed run select the same batch.
    subroutine fortbo_thompson_select(samples, selected, status)
        real(dp), intent(in) :: samples(:, :)
        integer, intent(out) :: selected(:)
        type(fortnum_status_t), intent(out) :: status
        logical, allocatable :: taken(:)
        real(dp) :: best_value
        integer :: n_candidates, batch_size, s, c, best

        n_candidates = size(samples, 1)
        batch_size = size(selected)
        selected = 0
        if (n_candidates < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: no candidates to select from")
            return
        end if
        if (batch_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: batch size must be positive")
            return
        end if
        if (size(samples, 2) < batch_size) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: need one posterior realization per batch member")
            return
        end if
        if (batch_size > n_candidates) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: batch is larger than the candidate set")
            return
        end if

        allocate (taken(n_candidates))
        taken = .false.
        do s = 1, batch_size
            best = 0
            best_value = huge(1.0_dp)
            do c = 1, n_candidates
                if (taken(c)) cycle
                if (samples(c, s) < best_value) then
                    best_value = samples(c, s)
                    best = c
                end if
            end do
            if (best == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo: candidate set exhausted")
                return
            end if
            selected(s) = best
            taken(best) = .true.
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_thompson_select

    !! Greedy Monte Carlo qEI selection from a frozen joint draw matrix.
    !!
    !! `samples(c, s)` is one objective-space draw at candidate `c`. At each
    !! slot, add the candidate with the largest expected improvement of the
    !! set already selected. This is the replayable discrete reduction used
    !! by the driver for q>1; q=1 remains the analytic EI path there.
    subroutine fortbo_qei_select(samples, threshold, selected, status)
        real(dp), intent(in) :: samples(:, :)
        real(dp), intent(in) :: threshold
        integer, intent(out) :: selected(:)
        type(fortnum_status_t), intent(out) :: status
        logical, allocatable :: taken(:)
        real(dp) :: score, best_score, improvement, best_value
        integer :: n_candidates, n_samples, batch_size, slot, c, s, prior, best

        n_candidates = size(samples, 1)
        n_samples = size(samples, 2)
        batch_size = size(selected)
        selected = 0
        if (n_candidates < 1 .or. n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: qEI needs at least one candidate and draw")
            return
        end if
        if (batch_size < 1 .or. batch_size > n_candidates) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo turbo: qEI batch does not fit the candidate set")
            return
        end if

        allocate (taken(n_candidates))
        taken = .false.
        do slot = 1, batch_size
            best = 0
            best_score = -huge(1.0_dp)
            do c = 1, n_candidates
                if (taken(c)) cycle
                score = 0.0_dp
                do s = 1, n_samples
                    best_value = 0.0_dp
                    do prior = 1, slot - 1
                        best_value = max(best_value, threshold - &
                            samples(selected(prior), s))
                    end do
                    improvement = max(best_value, threshold - samples(c, s))
                    score = score + improvement
                end do
                score = score/real(n_samples, dp)
                if (score > best_score) then
                    best_score = score
                    best = c
                end if
            end do
            if (best == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo turbo: qEI candidate set exhausted")
                return
            end if
            selected(slot) = best
            taken(best) = .true.
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_qei_select

    !! Thompson selection is an arg-min over a discrete set. It has no
    !! derivative with respect to the candidates, and any value a caller
    !! computed by perturbing the candidates and re-selecting would be an
    !! artifact of which candidate happened to win, not a derivative.
    subroutine fortbo_thompson_gradient_refusal(status)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "fortbo turbo: thompson_select is a discrete arg-min and has no "// &
            "input derivative")
    end subroutine fortbo_thompson_gradient_refusal

end module fortbo_turbo
