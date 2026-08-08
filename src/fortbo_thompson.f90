module fortbo_thompson
    !! Thompson sampling as a standalone batch rule (ROADMAP BO1).
    !!
    !! The selection itself already lives in `fortbo_turbo`, because that is
    !! where it was first needed. It is not trust-region-specific: given `q`
    !! posterior realizations over a candidate set, each batch slot takes the
    !! arg-min of its own realization. This module supplies the missing half —
    !! drawing those realizations from an arbitrary posterior — so the rule can
    !! be used with any policy, and against any provider, including one that is
    !! not a Gaussian process.
    !!
    !! **The realizations must be joint.** Drawing each candidate independently
    !! from its marginal would let two near-identical candidates take two batch
    !! slots, because nothing would tell the estimator they are the same
    !! question asked twice. Under a joint draw they rise and fall together and
    !! the second slot moves elsewhere on its own. This is the same reason
    !! `fortbo_batch` insists on joint samples, and it is why a posterior that
    !! offers only marginal moments is refused rather than accommodated.
    !!
    !! **Selection is without replacement.** A batch that evaluated one point
    !! `q` times would be a correct arg-min per slot and a useless batch, so a
    !! candidate already taken is skipped and the slot falls to the next best
    !! under its own realization. That is a deliberate departure from the
    !! textbook rule, which is silent on the case because it describes one draw
    !! at a time.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t
    use fortbo_posterior, only: fortbo_posterior_t
    use fortbo_batch, only: fortbo_batch_samples_t
    use fortbo_turbo, only: fortbo_thompson_select
    implicit none
    private

    public :: fortbo_thompson_batch

contains

    !! Draw `q` joint realizations over `candidates` and return the batch.
    !!
    !! `selected` holds indices into `candidates`, so a caller keeps whatever
    !! else it knows about those rows rather than being handed coordinates it
    !! then has to match back up.
    subroutine fortbo_thompson_batch(posterior, candidates, generator, selected, &
            status, points)
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: candidates(:, :)
        type(rng_t), intent(inout) :: generator
        integer, intent(out) :: selected(:)
        type(fortnum_status_t), intent(out) :: status
        !! Optionally, the selected rows themselves.
        real(dp), intent(out), optional :: points(:, :)
        type(fortbo_batch_samples_t) :: samples
        integer :: batch_size, i

        selected = 0
        batch_size = size(selected)
        if (batch_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo thompson: batch size must be positive")
            return
        end if
        if (size(candidates, 1) < batch_size) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo thompson: batch is larger than the candidate set")
            return
        end if
        if (present(points)) then
            if (size(points, 1) /= batch_size .or. &
                size(points, 2) /= size(candidates, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo thompson: point buffer does not match the batch")
                return
            end if
            points = 0.0_dp
        end if

        ! One independent realization per batch slot, all jointly drawn over the
        ! candidate set. `generate` refuses a posterior without joint sampling,
        ! which is the refusal that keeps this rule honest.
        call samples%generate(posterior, candidates, batch_size, generator, status)
        if (status%code /= FORTNUM_OK) return

        call fortbo_thompson_select(samples%draws, selected, status)
        if (status%code /= FORTNUM_OK) return

        if (present(points)) then
            do i = 1, batch_size
                points(i, :) = candidates(selected(i), :)
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_thompson_batch

end module fortbo_thompson
