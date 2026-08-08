module fortbo_constrained
    !! Constrained and cost-aware acquisitions (ROADMAP BO1).
    !!
    !! Both are weightings of a base acquisition, and both are wrong in an
    !! instructive way if the weighting is applied naively.
    !!
    !! **Constrained.** With independent constraint surrogates `c_k`, each
    !! required to satisfy `c_k(x) <= 0`, the probability of feasibility is the
    !! product of per-constraint tail probabilities and the acquisition becomes
    !!
    !!     cEI(x) = EI(x) * prod_k P(c_k(x) <= 0).
    !!
    !! The multiplication is only valid because `EI` is non-negative. Applying
    !! the same weight to an acquisition that can go negative — UCB in a
    !! minimization convention, for instance — would make an *infeasible* point
    !! score better than a feasible one whenever the base value is negative,
    !! since multiplying by a small probability moves a negative number *up*.
    !! This module therefore refuses to weight a negative base value rather than
    !! producing a number whose ordering is silently inverted.
    !!
    !! When nothing feasible has been observed, `EI` has no incumbent to improve
    !! on. Falling back to maximizing feasibility alone is the standard remedy
    !! and is what `fortbo_feasibility_probability` is for; the caller chooses,
    !! because "we have not found a feasible point yet" is a fact about the run
    !! that the acquisition should not silently paper over.
    !!
    !! **Cost-aware.** Dividing by cost gives improvement per unit cost:
    !!
    !!     cost_aware(x) = EI(x) / cost(x)^alpha.
    !!
    !! `alpha` exists because the raw ratio is badly behaved: as cost goes to
    !! zero the ratio diverges, so a cheap and nearly useless point beats an
    !! expensive and excellent one. `alpha` in `[0, 1]` interpolates between
    !! ignoring cost entirely and full per-unit-cost accounting, and the cost
    !! is floored rather than allowed to reach zero.
    !!
    !! Costs are *positive* by definition. A non-positive cost is refused, not
    !! clamped, because it almost always means the cost model was fitted in log
    !! space and someone forgot to exponentiate.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_feasibility_probability
    public :: fortbo_constrained_acquisition
    public :: fortbo_cost_aware_acquisition

    !! Smallest cost that may appear in a denominator. Below this the ratio is
    !! dominated by the cost model's own error rather than by the objective.
    real(dp), parameter, public :: FORTBO_COST_FLOOR = 1.0e-12_dp

contains

    !! `prod_k P(c_k(x) <= threshold_k)` under independent normal beliefs.
    !!
    !! Independence across constraints is assumed and stated. It holds for
    !! separately fitted constraint surrogates, which is the usual arrangement;
    !! it does not hold for a multi-output model, and this routine is the wrong
    !! one to use there.
    subroutine fortbo_feasibility_probability(mean, sd, threshold, probability, &
            status)
        !! `mean(constraint, point)` and `sd(constraint, point)`.
        real(dp), intent(in) :: mean(:, :)
        real(dp), intent(in) :: sd(:, :)
        real(dp), intent(in) :: threshold(:)
        real(dp), intent(out) :: probability(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: z
        integer :: n_constraints, n_points, i, k

        probability = 0.0_dp
        n_constraints = size(mean, 1)
        n_points = size(mean, 2)
        if (n_constraints < 1 .or. n_points < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: at least one constraint and point are required")
            return
        end if
        if (size(sd, 1) /= n_constraints .or. size(sd, 2) /= n_points .or. &
            size(threshold) /= n_constraints .or. size(probability) /= n_points) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: constraint array shapes disagree")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: standard deviations must not be negative")
            return
        end if

        do i = 1, n_points
            probability(i) = 1.0_dp
            do k = 1, n_constraints
                if (sd(k, i) > 0.0_dp) then
                    z = (threshold(k) - mean(k, i))/sd(k, i)
                    probability(i) = probability(i)* &
                        0.5_dp*(1.0_dp + erf(z/sqrt(2.0_dp)))
                else
                    ! A constraint known exactly is satisfied or it is not.
                    ! Reporting one half at the boundary keeps the limit
                    ! symmetric, as elsewhere in FortBO.
                    if (mean(k, i) < threshold(k)) then
                        probability(i) = probability(i)*1.0_dp
                    else if (mean(k, i) > threshold(k)) then
                        probability(i) = 0.0_dp
                    else
                        probability(i) = probability(i)*0.5_dp
                    end if
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_feasibility_probability

    !! Weight a non-negative base acquisition by the probability of feasibility.
    subroutine fortbo_constrained_acquisition(base, probability, value, status)
        real(dp), intent(in) :: base(:)
        real(dp), intent(in) :: probability(:)
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (size(base) /= size(probability) .or. size(value) /= size(base)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: acquisition shapes disagree")
            return
        end if
        if (any(base < 0.0_dp)) then
            ! Multiplying a negative value by a probability moves it *up*, so a
            ! point that is almost certainly infeasible would outrank a feasible
            ! one. The ordering would be inverted with nothing in the output
            ! saying so.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: base acquisition must be non-negative")
            return
        end if
        if (any(probability < 0.0_dp) .or. any(probability > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo constrained: feasibility must be a probability")
            return
        end if

        value = base*probability
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_constrained_acquisition

    !! `base / cost**alpha`, improvement per unit cost.
    subroutine fortbo_cost_aware_acquisition(base, cost, alpha, value, status)
        real(dp), intent(in) :: base(:)
        real(dp), intent(in) :: cost(:)
        real(dp), intent(in) :: alpha
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        value = 0.0_dp
        if (size(base) /= size(cost) .or. size(value) /= size(base)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo cost aware: acquisition shapes disagree")
            return
        end if
        if (alpha < 0.0_dp .or. alpha > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo cost aware: alpha must lie in [0, 1]")
            return
        end if
        if (any(cost <= 0.0_dp)) then
            ! Almost always a cost model fitted in log space and never
            ! exponentiated. Clamping would hide that.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo cost aware: costs must be positive")
            return
        end if
        if (any(base < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo cost aware: base acquisition must be non-negative")
            return
        end if

        do i = 1, size(base)
            value(i) = base(i)/max(cost(i), FORTBO_COST_FLOOR)**alpha
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_cost_aware_acquisition

end module fortbo_constrained
