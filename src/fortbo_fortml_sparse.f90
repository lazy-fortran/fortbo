module fortbo_fortml_sparse
    !! Sparse/variational and multi-output FortML surrogates behind the posterior
    !! contract (ROADMAP BO2).
    !!
    !! Both adapters exist to make the same point the Bayesian linear provider
    !! makes, with models a run would actually use: an acquisition does not know
    !! what produced its moments. What is worth stating is where each one's
    !! honesty boundary sits, because that is what a capability bit is for.
    !!
    !! **Sparse (variational) GP.** Its predictive marginals are those of the
    !! variational posterior `q(f_*)`, not of the exact posterior. That is a
    !! genuinely different distribution — variational inference systematically
    !! *underestimates* posterior variance — so an acquisition run against it
    !! explores less than the same acquisition against an exact GP. The adapter
    !! does not correct for that and could not: the correction is not known. It
    !! declares moments, and the caller who chose a sparse model chose that
    !! trade.
    !!
    !! **Multi-output GP.** FortBO's posterior contract is single-output by
    !! construction — an acquisition ranks scalars. The adapter therefore
    !! projects onto one selected output and presents that. It declares joint
    !! covariance and sampling, because the underlying joint covariance is real
    !! and restricting it to one output's rows is exact, not an approximation.
    !! What it does *not* do is pretend the other outputs are absent: they still
    !! condition the selected one, which is the entire reason to fit a
    !! multi-output model rather than several independent ones.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_normal
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_sparse_gp, only: sparse_gp_t
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortml_student_t_process, only: student_t_process_t
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENTS, &
        FORTBO_CAP_NOISY_MOMENTS, FORTBO_CAP_COVARIANCE, FORTBO_CAP_JOINT_SAMPLE
    implicit none
    private

    public :: fortbo_sparse_gp_posterior_t
    public :: fortbo_multi_output_posterior_t
    public :: fortbo_student_t_posterior_t

    !! Jitter added before factorizing a joint covariance for sampling. A
    !! multi-output covariance restricted to one output is often near-singular
    !! at closely spaced queries, and refusing there would make the adapter
    !! useless exactly where a batch policy wants it.
    real(dp), parameter, public :: FORTBO_JOINT_JITTER = 1.0e-10_dp

    type, extends(fortbo_posterior_t) :: fortbo_sparse_gp_posterior_t
        type(sparse_gp_t) :: model
        integer :: dimension = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => sparse_n_inputs
        procedure, public :: capabilities => sparse_capabilities
        procedure, public :: moments => sparse_moments
    end type fortbo_sparse_gp_posterior_t

    !! Student-t process behind the contract.
    !!
    !! Its predictive *variance* is what the contract asks for and what this
    !! reports. What the contract cannot carry is the rest of the distribution:
    !! a TP's marginals are Student-t, with heavier tails than a normal at the
    !! same variance, and every acquisition in FortBO integrates against a
    !! Gaussian. An acquisition run here is therefore using the right first two
    !! moments and the wrong tail, which understates the value of exploring
    !! where the model is least certain. That is a real approximation, it is
    !! stated rather than hidden, and it shrinks as the posterior degrees of
    !! freedom grow — which they do with every observation.
    type, extends(fortbo_posterior_t) :: fortbo_student_t_posterior_t
        type(student_t_process_t) :: model
        integer :: dimension = 0
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => student_t_n_inputs
        procedure, public :: capabilities => student_t_capabilities
        procedure, public :: moments => student_t_moments
    end type fortbo_student_t_posterior_t

    type, extends(fortbo_posterior_t) :: fortbo_multi_output_posterior_t
        type(multi_output_gp_t) :: model
        integer :: dimension = 0
        !! Which output the acquisition ranks. The others still condition it.
        integer :: output = 1
        integer :: n_outputs = 1
        logical :: fitted = .false.
    contains
        procedure, public :: n_inputs => multi_n_inputs
        procedure, public :: capabilities => multi_capabilities
        procedure, public :: moments => multi_moments
        procedure, public :: covariance => multi_covariance
        procedure, public :: joint_sample => multi_joint_sample
    end type fortbo_multi_output_posterior_t

contains

    pure integer function sparse_n_inputs(self) result(n)
        class(fortbo_sparse_gp_posterior_t), intent(in) :: self

        n = self%dimension
    end function sparse_n_inputs

    !! Marginal moments only. The sparse model has no joint-covariance route in
    !! FortML, so batch acquisitions refuse against it rather than being handed
    !! independent marginals — which for a batch is worse than no answer.
    pure integer function sparse_capabilities(self) result(caps)
        class(fortbo_sparse_gp_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS
    end function sparse_capabilities

    subroutine sparse_moments(self, points, mean, variance, status)
        class(fortbo_sparse_gp_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo sparse: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension .or. size(mean) /= size(points, 1) &
            .or. size(variance) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo sparse: query shape does not match the surrogate")
            return
        end if
        call self%model%predict(points, mean, variance, status)
    end subroutine sparse_moments

    pure integer function student_t_n_inputs(self) result(n)
        class(fortbo_student_t_posterior_t), intent(in) :: self

        n = self%dimension
    end function student_t_n_inputs

    pure integer function student_t_capabilities(self) result(caps)
        class(fortbo_student_t_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS
    end function student_t_capabilities

    subroutine student_t_moments(self, points, mean, variance, status)
        class(fortbo_student_t_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo student-t: surrogate has not been fitted")
            return
        end if
        if (size(points, 2) /= self%dimension .or. size(mean) /= size(points, 1) &
            .or. size(variance) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo student-t: query shape does not match the surrogate")
            return
        end if
        call self%model%predict(points, mean, variance, status)
    end subroutine student_t_moments

    pure integer function multi_n_inputs(self) result(n)
        class(fortbo_multi_output_posterior_t), intent(in) :: self

        n = self%dimension
    end function multi_n_inputs

    pure integer function multi_capabilities(self) result(caps)
        class(fortbo_multi_output_posterior_t), intent(in) :: self

        caps = 0
        if (self%fitted) then
            caps = FORTBO_CAP_MOMENTS + FORTBO_CAP_NOISY_MOMENTS &
                + FORTBO_CAP_COVARIANCE + FORTBO_CAP_JOINT_SAMPLE
        end if
    end function multi_capabilities

    !! Marginals of the selected output, read off the diagonal of the joint
    !! covariance restricted to that output's rows.
    subroutine multi_moments(self, points, mean, variance, status)
        class(fortbo_multi_output_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: full_mean(:, :), joint(:, :)
        integer :: n, i

        mean = 0.0_dp
        variance = 0.0_dp
        call check_multi(self, points, size(mean), status)
        if (status%code /= FORTNUM_OK) return
        if (size(variance) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-output: variance width does not match the query")
            return
        end if

        n = size(points, 1)
        allocate (full_mean(n, self%n_outputs))
        call self%model%predict(points, full_mean, status)
        if (status%code /= FORTNUM_OK) return
        mean = full_mean(:, self%output)

        allocate (joint(n, n))
        call selected_covariance(self, points, joint, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            variance(i) = max(joint(i, i), 0.0_dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_moments

    subroutine multi_covariance(self, points, covariance, status)
        class(fortbo_multi_output_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        covariance = 0.0_dp
        call check_multi(self, points, size(covariance, 1), status)
        if (status%code /= FORTNUM_OK) return
        if (size(covariance, 2) /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-output: covariance must be square in the query")
            return
        end if
        call selected_covariance(self, points, covariance, status)
    end subroutine multi_covariance

    !! The joint covariance of the *selected* output over the query set.
    !!
    !! FortML stacks with outputs varying *slowest*, so the selected output's
    !! block is a contiguous square rather than a strided one. Getting that
    !! wrong yields a plausible symmetric positive-definite matrix belonging to
    !! another output, which is why the test compares the diagonal against the
    !! marginals rather than only checking symmetry — and why the fixture has to
    !! give the two outputs different marginal scales for the check to bite.
    subroutine selected_covariance(self, points, covariance, status)
        class(fortbo_multi_output_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: joint(:, :)
        integer :: n, p, i, j, row, column

        n = size(points, 1)
        p = self%n_outputs
        allocate (joint(n*p, n*p))
        ! The *posterior* covariance, not `joint_covariance`, which is the prior
        ! `B (x) K`. Using the prior would report uncertainty that never shrinks
        ! with data — plausible on a plot and simply wrong. FortML had no
        ! posterior route until this adapter needed one.
        call self%model%predict_covariance(points, joint, status)
        if (status%code /= FORTNUM_OK) return

        ! Outputs vary *slowest* in FortML's stacking, so the selected output's
        ! block is contiguous rows rather than a stride of `p`.
        do i = 1, n
            row = (self%output - 1)*n + i
            do j = 1, n
                column = (self%output - 1)*n + j
                covariance(i, j) = joint(row, column)
            end do
        end do
    end subroutine selected_covariance

    !! Joint draws of the selected output, by Cholesky of its sub-block.
    subroutine multi_joint_sample(self, points, generator, samples, status)
        class(fortbo_multi_output_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: samples(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: covariance(:, :), mean(:), variance(:), draw(:)
        integer :: n, i, s

        samples = 0.0_dp
        n = size(points, 1)
        call check_multi(self, points, size(samples, 1), status)
        if (status%code /= FORTNUM_OK) return

        allocate (mean(n), variance(n), covariance(n, n), draw(n))
        call self%moments(points, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call selected_covariance(self, points, covariance, status)
        if (status%code /= FORTNUM_OK) return
        ! Closely spaced queries make this near-singular; the jitter is what
        ! keeps a batch policy usable rather than refusing where it is needed.
        do i = 1, n
            covariance(i, i) = covariance(i, i) + FORTBO_JOINT_JITTER
        end do

        call factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, size(samples, 2)
            do i = 1, n
                call rng_normal(generator, draw(i))
            end do
            samples(:, s) = mean + matmul(factorization%lower, draw)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_joint_sample

    pure subroutine check_multi(self, points, n_out, status)
        class(fortbo_multi_output_posterior_t), intent(in) :: self
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: n_out
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-output: surrogate has not been fitted")
            return
        end if
        if (self%output < 1 .or. self%output > self%n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-output: selected output is out of range")
            return
        end if
        if (size(points, 2) /= self%dimension .or. n_out /= size(points, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo multi-output: query shape does not match the surrogate")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_multi

end module fortbo_fortml_sparse
