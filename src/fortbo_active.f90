module fortbo_active
    !! Active learning, level-set estimation, and design of experiments
    !! (ROADMAP BO4).
    !!
    !! These are the goals that are *not* optimization. A run may want to learn
    !! the whole surface, find where it crosses a threshold, or lay out a design
    !! before any model exists. Using an optimization acquisition for any of
    !! them concentrates evaluations near the optimum, which is precisely the
    !! wrong place when the answer lives elsewhere.
    !!
    !!   * **Active learning (ALM).** Sample where the posterior is least
    !!     certain. The score is the posterior variance, and the point of
    !!     naming it rather than inlining `maxval(variance)` is that it makes
    !!     the *goal* explicit in a run's record.
    !!   * **Level-set estimation (straddle).** To find where `f(x) = t`, score
    !!     `kappa sigma - |mu - t|`. The two terms pull in different directions:
    !!     the first wants uncertainty, the second wants proximity to the
    !!     threshold, and their difference selects points that are both close to
    !!     the contour and still in doubt. Maximizing variance alone would chase
    !!     the least-known corner of the space, which usually is not near the
    !!     contour at all.
    !!   * **Feasibility search** is level-set estimation with `t = 0` on a
    !!     constraint, so it needs no separate routine — noted here because
    !!     adding one would suggest otherwise.
    !!   * **Maximin design.** Before a model exists, spread points by
    !!     maximizing the smallest pairwise distance. This is a property of the
    !!     design alone, so it is scored, not sampled, and it gives a
    !!     model-free starting design and a baseline any adaptive method must
    !!     beat.
    !!
    !! Every score here follows the FortBO convention that *lower is better*, so
    !! the same optimizers apply unchanged. That is why the straddle score is
    !! negated relative to the form usually published, and why the maximin score
    !! is the negative of the minimum distance.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_active_learning_score
    public :: fortbo_straddle_score
    public :: fortbo_level_set_probability
    public :: fortbo_minimum_pairwise_distance
    public :: fortbo_maximin_score

contains

    !! Negative posterior variance: lower is better, so the least certain point
    !! wins.
    subroutine fortbo_active_learning_score(variance, score, status)
        real(dp), intent(in) :: variance(:)
        real(dp), intent(out) :: score(:)
        type(fortnum_status_t), intent(out) :: status

        score = 0.0_dp
        if (size(score) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: score and variance shapes disagree")
            return
        end if
        if (any(variance < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: variance must not be negative")
            return
        end if

        score = -variance
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_active_learning_score

    !! `|mu - t| - kappa sigma`, lower is better.
    !!
    !! `kappa` sets how much uncertainty is worth relative to distance from the
    !! threshold. At zero the rule collapses to "sample nearest the contour",
    !! which stops learning as soon as one crossing is found; as `kappa` grows it
    !! degenerates to pure variance sampling and forgets the contour. Neither
    !! limit is useful, and both are reachable, so the caller chooses.
    subroutine fortbo_straddle_score(mean, sd, threshold, kappa, score, status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: threshold
        real(dp), intent(in) :: kappa
        real(dp), intent(out) :: score(:)
        type(fortnum_status_t), intent(out) :: status

        score = 0.0_dp
        if (size(sd) /= size(mean) .or. size(score) /= size(mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: straddle shapes disagree")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: standard deviation must not be negative")
            return
        end if
        if (kappa < 0.0_dp) then
            ! A negative kappa would prefer *certainty* near the contour, which
            ! stops the search dead once one crossing is known.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: kappa must not be negative")
            return
        end if

        score = abs(mean - threshold) - kappa*sd
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_straddle_score

    !! `P(f(x) <= threshold)` under the marginal posterior — the membership
    !! probability of the sub-level set.
    subroutine fortbo_level_set_probability(mean, sd, threshold, probability, &
            status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        real(dp), intent(in) :: threshold
        real(dp), intent(out) :: probability(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        probability = 0.0_dp
        if (size(sd) /= size(mean) .or. size(probability) /= size(mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: level-set shapes disagree")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: standard deviation must not be negative")
            return
        end if

        do i = 1, size(mean)
            if (sd(i) > 0.0_dp) then
                probability(i) = 0.5_dp*(1.0_dp + &
                    erf((threshold - mean(i))/(sd(i)*sqrt(2.0_dp))))
            else if (mean(i) < threshold) then
                probability(i) = 1.0_dp
            else if (mean(i) > threshold) then
                probability(i) = 0.0_dp
            else
                probability(i) = 0.5_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_level_set_probability

    !! Smallest Euclidean distance between any two distinct design points.
    subroutine fortbo_minimum_pairwise_distance(design, distance, status)
        !! `design(point, coordinate)`.
        real(dp), intent(in) :: design(:, :)
        real(dp), intent(out) :: distance
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: candidate
        integer :: n, i, j

        distance = 0.0_dp
        n = size(design, 1)
        if (n < 2 .or. size(design, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo active: a design needs at least two points")
            return
        end if

        distance = huge(1.0_dp)
        do i = 1, n - 1
            do j = i + 1, n
                candidate = sqrt(sum((design(i, :) - design(j, :))**2))
                distance = min(distance, candidate)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_minimum_pairwise_distance

    !! Negative minimum pairwise distance, so a better-spread design scores
    !! lower and the ordinary optimizers apply.
    subroutine fortbo_maximin_score(design, score, status)
        real(dp), intent(in) :: design(:, :)
        real(dp), intent(out) :: score
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance

        score = 0.0_dp
        call fortbo_minimum_pairwise_distance(design, distance, status)
        if (status%code /= FORTNUM_OK) return
        score = -distance
    end subroutine fortbo_maximin_score

end module fortbo_active
