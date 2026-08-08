module fortbo_entropy
    !! Max-value entropy search (ROADMAP BO1).
    !!
    !! Improvement-based acquisitions ask how much better the next observation
    !! will be. Entropy search asks a different question: how much will the next
    !! observation *tell us* about where the optimum is? The two disagree
    !! whenever an evaluation is informative without being good — a point that
    !! will almost certainly turn out mediocre, but whose value would rule out a
    !! whole region, is worth an evaluation and expected improvement cannot say
    !! so.
    !!
    !! Full entropy search targets the distribution over the *location* of the
    !! optimum, which lives in the input space and needs an expensive nested
    !! approximation. Max-value entropy search targets the distribution over its
    !! *value* instead, which is one-dimensional whatever the input dimension
    !! is, and the resulting acquisition is available in closed form:
    !!
    !!     MES(x) = H(f(x)) - E_{y*}[ H(f(x) | f(x) >= y*) ]
    !!            = mean over sampled y* of  -log c - gamma phi(gamma) / (2 c)
    !!
    !! with `gamma = (y* - mu)/sigma` and `c = 1 - Phi(gamma)`. FortBO
    !! minimizes, so `y*` is the *minimum* value and conditioning truncates the
    !! posterior from below; the maximization form found in the literature has
    !! the opposite truncation and the opposite sign on the second term, and
    !! transcribing it unchanged is the obvious way to get this wrong.
    !!
    !! The samples of `y*` are supplied by the caller, as the reference set is
    !! for knowledge gradient. They come from Thompson sampling over a candidate
    !! set, or from a Gumbel fit to the posterior maximum; which one is a
    !! decision about the problem, and hiding it inside the acquisition would
    !! make two runs incomparable without either of them saying why.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_truncated_normal_entropy
    public :: fortbo_max_value_entropy_search

    !! Below this tail mass the conditioning event is numerically impossible and
    !! its logarithm is meaningless. A candidate the sampled optimum has already
    !! ruled out contributes nothing rather than an enormous number: the
    !! information is real but unusable, and reporting it would let one degenerate
    !! sample dominate the average.
    real(dp), parameter, public :: FORTBO_MES_TAIL_FLOOR = 1.0e-12_dp

contains

    !! Differential entropy of a normal truncated to `[lower, infinity)`.
    !!
    !!     H = log(sqrt(2 pi e) sigma Z) + alpha phi(alpha) / (2 Z)
    !!
    !! with `alpha = (lower - mu)/sigma` and `Z = 1 - Phi(alpha)`. Returned in
    !! nats, as everything else here is.
    subroutine fortbo_truncated_normal_entropy(mean, sd, lower, entropy, status)
        real(dp), intent(in) :: mean
        real(dp), intent(in) :: sd
        real(dp), intent(in) :: lower
        real(dp), intent(out) :: entropy
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: alpha, tail, density

        entropy = 0.0_dp
        if (sd <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo entropy: standard deviation must be positive")
            return
        end if

        alpha = (lower - mean)/sd
        tail = 0.5_dp*(1.0_dp - erf(alpha/sqrt(2.0_dp)))
        if (tail <= FORTBO_MES_TAIL_FLOOR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo entropy: truncation leaves no usable mass")
            return
        end if
        density = exp(-0.5_dp*alpha*alpha)/sqrt(8.0_dp*atan(1.0_dp))

        entropy = log(sqrt(2.0_dp*exp(1.0_dp)*4.0_dp*atan(1.0_dp))*sd*tail) &
            + alpha*density/(2.0_dp*tail)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_truncated_normal_entropy

    !! MES at each candidate, averaged over the supplied samples of the optimum.
    subroutine fortbo_max_value_entropy_search(mean, sd, optimum_samples, value, &
            status)
        real(dp), intent(in) :: mean(:)
        real(dp), intent(in) :: sd(:)
        !! Samples of the objective's minimum value.
        real(dp), intent(in) :: optimum_samples(:)
        real(dp), intent(out) :: value(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gamma, tail, density, contribution
        integer :: i, k, used

        value = 0.0_dp
        if (size(sd) /= size(mean) .or. size(value) /= size(mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo entropy: moment shapes disagree")
            return
        end if
        if (size(optimum_samples) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo entropy: at least one optimum sample is required")
            return
        end if
        if (any(sd < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo entropy: standard deviations must not be negative")
            return
        end if

        do i = 1, size(mean)
            if (sd(i) <= 0.0_dp) then
                ! A point the posterior already knows exactly cannot be
                ! informative about anything, whatever the sampled optimum is.
                value(i) = 0.0_dp
                cycle
            end if
            used = 0
            do k = 1, size(optimum_samples)
                gamma = (optimum_samples(k) - mean(i))/sd(i)
                tail = 0.5_dp*(1.0_dp - erf(gamma/sqrt(2.0_dp)))
                if (tail <= FORTBO_MES_TAIL_FLOOR) cycle
                density = exp(-0.5_dp*gamma*gamma)/sqrt(8.0_dp*atan(1.0_dp))
                contribution = -log(tail) - gamma*density/(2.0_dp*tail)
                ! Mutual information is non-negative; a negative value here is
                ! rounding in the tail, not a finding.
                value(i) = value(i) + max(contribution, 0.0_dp)
                used = used + 1
            end do
            if (used > 0) then
                value(i) = value(i)/real(used, dp)
            else
                value(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_max_value_entropy_search

end module fortbo_entropy
