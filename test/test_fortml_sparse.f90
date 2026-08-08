program test_fortml_sparse
    !! BO2: sparse, multi-output, and Student-t surrogates behind the contract.
    !!
    !! The claim is the contract's, not the models': acquisitions run unchanged.
    !! What the tests pin beyond that is where each adapter's honesty boundary
    !! sits, since a capability bit is only worth having if it is accurate.
    !!
    !!   * the multi-output adapter's covariance must be the *selected* output's
    !!     sub-block, checked by requiring its diagonal to equal the marginals.
    !!     Reading the wrong stride would produce a plausible symmetric matrix
    !!     belonging to another output, which symmetry alone would not catch;
    !!   * its joint samples must reproduce that covariance, which is what makes
    !!     a batch acquisition on top of it valid;
    !!   * the Student-t adapter reports the TP's variance, and the test
    !!     confirms it differs from a GP's on the same inputs — otherwise the
    !!     adapter would be a GP wearing another name.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortbo_posterior, only: FORTBO_CAP_MOMENTS, FORTBO_CAP_COVARIANCE, &
        FORTBO_CAP_JOINT_SAMPLE
    use fortbo_fortml_sparse, only: fortbo_multi_output_posterior_t, &
        fortbo_student_t_posterior_t, fortbo_heteroskedastic_posterior_t, &
        fortbo_classification_posterior_t
    use fortbo_acquisition, only: fortbo_ei_t
    use fortbo_batch, only: fortbo_batch_samples_t, fortbo_qei
    implicit none

    integer :: failures

    failures = 0
    call check_multi_output_covariance_is_the_right_block(failures)
    call check_multi_output_samples_match_its_covariance(failures)
    call check_student_t_differs_from_a_gaussian(failures)
    call check_acquisitions_run_unchanged(failures)
    call check_heteroskedastic_separates_its_two_uncertainties(failures)
    call check_classification_presents_the_latent(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_fortml_sparse: PASS"
    else
        print *, "test_fortml_sparse: FAIL", failures
        error stop 1
    end if

contains

    subroutine build_multi(model, output, status)
        type(fortbo_multi_output_posterior_t), intent(out) :: model
        integer, intent(in) :: output
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel
        real(dp) :: x(6, 1), y(6, 2)
        integer :: k

        do k = 1, 6
            x(k, 1) = -1.0_dp + 0.4_dp*real(k - 1, dp)
            y(k, 1) = sin(2.0_dp*x(k, 1))
            y(k, 2) = cos(1.5_dp*x(k, 1)) + 0.3_dp*x(k, 1)
        end do
        kernel = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        if (status%code /= FORTNUM_OK) return
        ! A linear-coregionalization weighting with two latent factors, so the
        ! outputs are genuinely coupled rather than independent fits.
        !
        ! The rows must give different diagonals of `B = W W'`. An earlier
        ! version used rows [1, 0] and [0.6, 0.8], whose diagonals are both one,
        ! so the two outputs had identical covariance blocks and the test could
        ! not tell a correct stride from a wrong one — it looked like a stride
        ! bug and was a fixture bug.
        call model%model%initialize(kernel, &
            reshape([1.0_dp, 0.6_dp, 0.0_dp, 1.2_dp], [2, 2]), &
            [0.05_dp, 0.09_dp], 0.02_dp, status)
        if (status%code /= FORTNUM_OK) return
        call model%model%fit(x, y, status)
        if (status%code /= FORTNUM_OK) return
        model%dimension = 1
        model%n_outputs = 2
        model%output = output
        model%fitted = .true.
    end subroutine build_multi

    !! The diagonal of the reported covariance must be the reported marginals.
    !! A wrong stride into the joint gives another output's block, which is
    !! symmetric and positive definite and completely wrong.
    subroutine check_multi_output_covariance_is_the_right_block(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_output_posterior_t) :: first, second
        type(fortnum_status_t) :: status
        real(dp) :: points(4, 1)
        real(dp) :: mean_a(4), var_a(4), cov_a(4, 4)
        real(dp) :: mean_b(4), var_b(4), cov_b(4, 4)
        integer :: k
        logical :: consistent

        do k = 1, 4
            points(k, 1) = -0.8_dp + 0.5_dp*real(k - 1, dp)
        end do

        call build_multi(first, 1, status)
        call expect(status%code == FORTNUM_OK, "the multi-output model fits", &
            failures)
        call expect(first%supports(FORTBO_CAP_MOMENTS) .and. &
            first%supports(FORTBO_CAP_COVARIANCE) .and. &
            first%supports(FORTBO_CAP_JOINT_SAMPLE), &
            "the multi-output adapter declares its capabilities", failures)

        call first%moments(points, mean_a, var_a, status)
        call first%covariance(points, cov_a, status)
        call expect(status%code == FORTNUM_OK, "the covariance evaluates", failures)

        consistent = .true.
        do k = 1, 4
            if (abs(cov_a(k, k) - var_a(k)) > 1.0e-12_dp) consistent = .false.
        end do
        call expect(consistent, &
            "the covariance diagonal is the reported marginal variance", failures)
        call expect(maxval(abs(cov_a - transpose(cov_a))) < 1.0e-12_dp, &
            "the covariance is symmetric", failures)

        ! Selecting the other output must actually give a different model.
        call build_multi(second, 2, status)
        call second%moments(points, mean_b, var_b, status)
        call second%covariance(points, cov_b, status)
        call expect(maxval(abs(mean_a - mean_b)) > 1.0e-8_dp, &
            "the two outputs have different posterior means", failures)
        call expect(maxval(abs(cov_a - cov_b)) > 1.0e-12_dp, &
            "selecting an output selects its covariance block", failures)
    end subroutine check_multi_output_covariance_is_the_right_block

    subroutine check_multi_output_samples_match_its_covariance(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_output_posterior_t) :: model
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n = 3, n_samples = 200000
        real(dp) :: points(n, 1), covariance(n, n), samples(n, 1)
        real(dp) :: totals(n), cross(n, n), empirical, standard_error
        integer :: k, s, i, j
        logical :: matches

        do k = 1, n
            points(k, 1) = -0.6_dp + 0.6_dp*real(k - 1, dp)
        end do
        call build_multi(model, 1, status)
        call model%covariance(points, covariance, status)

        call rng_seed(generator, int(636363, int64), status)
        totals = 0.0_dp
        cross = 0.0_dp
        do s = 1, n_samples
            call model%joint_sample(points, generator, samples, status)
            if (status%code /= FORTNUM_OK) exit
            totals = totals + samples(:, 1)
            do i = 1, n
                do j = 1, n
                    cross(i, j) = cross(i, j) + samples(i, 1)*samples(j, 1)
                end do
            end do
        end do
        totals = totals/real(n_samples, dp)

        matches = .true.
        do i = 1, n
            do j = 1, n
                empirical = cross(i, j)/real(n_samples, dp) - totals(i)*totals(j)
                standard_error = sqrt(abs(covariance(i, i)*covariance(j, j))) &
                    /sqrt(real(n_samples, dp))
                if (abs(empirical - covariance(i, j)) > 6.0_dp*standard_error) &
                    matches = .false.
            end do
        end do
        call expect(matches, &
            "joint samples reproduce the selected output's covariance", failures)
    end subroutine check_multi_output_samples_match_its_covariance

    !! If the adapter reported a GP's variance it would be a GP wearing another
    !! name, so the contrast is the test.
    subroutine check_student_t_differs_from_a_gaussian(failures)
        integer, intent(inout) :: failures
        type(fortbo_student_t_posterior_t) :: tame, wild
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), calm(6), loud(6), points(3, 1)
        real(dp) :: mean_a(3), var_a(3), mean_b(3), var_b(3)
        integer :: k

        do k = 1, 6
            x(k, 1) = -1.0_dp + 0.4_dp*real(k - 1, dp)
            calm(k) = 0.02_dp*sin(2.0_dp*x(k, 1))
            loud(k) = 5.0_dp*sin(2.0_dp*x(k, 1))
        end do
        do k = 1, 3
            points(k, 1) = -0.7_dp + 0.6_dp*real(k - 1, dp)
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        call tame%model%fit(x, calm, kernel, 4.0_dp, 0.02_dp, status)
        tame%dimension = 1
        tame%fitted = .true.
        kernel = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        call wild%model%fit(x, loud, kernel, 4.0_dp, 0.02_dp, status)
        wild%dimension = 1
        wild%fitted = .true.

        call tame%moments(points, mean_a, var_a, status)
        call expect(status%code == FORTNUM_OK, "the Student-t adapter evaluates", &
            failures)
        call wild%moments(points, mean_b, var_b, status)

        ! Same inputs, same kernel, same noise: only the observations differ.
        ! A GP would report identical variances here.
        call expect(all(var_b > var_a), &
            "the Student-t variance responds to the observed values", failures)
        call expect(tame%supports(FORTBO_CAP_MOMENTS), &
            "the Student-t adapter declares moments", failures)
        call expect(.not. tame%supports(FORTBO_CAP_JOINT_SAMPLE), &
            "it does not claim joint sampling it cannot do", failures)
    end subroutine check_student_t_differs_from_a_gaussian

    subroutine check_acquisitions_run_unchanged(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_output_posterior_t) :: multi
        type(fortbo_student_t_posterior_t) :: process
        type(fortbo_batch_samples_t) :: batch
        type(fortbo_ei_t) :: ei
        type(kernel_t) :: kernel
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), points(4, 1), values(4), qei
        integer :: k

        do k = 1, 6
            x(k, 1) = -1.0_dp + 0.4_dp*real(k - 1, dp)
            y(k) = sin(2.0_dp*x(k, 1))
        end do
        do k = 1, 4
            points(k, 1) = -0.8_dp + 0.5_dp*real(k - 1, dp)
        end do

        ei%best = 1.0_dp
        ei%xi = 0.0_dp

        call build_multi(multi, 1, status)
        call ei%value(multi, points, values, status)
        call expect(status%code == FORTNUM_OK .and. all(values >= 0.0_dp), &
            "expected improvement runs against the multi-output adapter", failures)

        call rng_seed(generator, int(1234, int64), status)
        call batch%generate(multi, points, 200, generator, status)
        call fortbo_qei(batch, 1.0_dp, 0.0_dp, qei, status)
        call expect(status%code == FORTNUM_OK .and. qei >= 0.0_dp, &
            "qEI runs against the multi-output adapter", failures)

        kernel = make_rbf_kernel(1, 1.0_dp, 0.7_dp, status)
        call process%model%fit(x, y, kernel, 5.0_dp, 0.02_dp, status)
        process%dimension = 1
        process%fitted = .true.
        call ei%value(process, points, values, status)
        call expect(status%code == FORTNUM_OK .and. all(values >= 0.0_dp), &
            "expected improvement runs against the Student-t adapter", failures)
    end subroutine check_acquisitions_run_unchanged

    !! The distinction a heteroskedastic surrogate exists to make: uncertainty
    !! about the *signal* is not the same as uncertainty in the *measurement*.
    !! A point unmeasured deserves a first evaluation; a point measured badly
    !! deserves a repeat. A plain GP cannot tell a policy which it is looking at.
    subroutine check_heteroskedastic_separates_its_two_uncertainties(failures)
        integer, intent(inout) :: failures
        type(fortbo_heteroskedastic_posterior_t) :: model
        type(kernel_t) :: signal, noise_kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 1), y(10), variances(10)
        real(dp) :: probe(2, 1), mean(2), variance(2), noise(2)
        integer :: k

        ! Left half measured precisely, right half poorly.
        do k = 1, 10
            x(k, 1) = -2.0_dp + 0.4_dp*real(k - 1, dp)
            y(k) = 0.5_dp*x(k, 1)
            if (x(k, 1) < 0.0_dp) then
                variances(k) = 1.0e-4_dp
            else
                variances(k) = 1.0_dp
            end if
        end do

        signal = make_rbf_kernel(1, 1.0_dp, 0.6_dp, status)
        noise_kernel = make_rbf_kernel(1, 1.0_dp, 1.2_dp, status)
        call model%model%fit(x, y, variances, signal, noise_kernel, status)
        call expect(status%code == FORTNUM_OK, "the heteroskedastic model fits", &
            failures)
        model%dimension = 1
        model%fitted = .true.

        probe(1, 1) = -1.4_dp
        probe(2, 1) = 1.4_dp
        call model%moments(probe, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "the adapter reports moments", &
            failures)
        call model%observation_noise(probe, noise, status)
        call expect(status%code == FORTNUM_OK, "the adapter reports noise", failures)

        call expect(variance(1) < variance(2), &
            "signal uncertainty is lower where the data are precise", failures)
        call expect(noise(1) < noise(2), &
            "measurement noise is lower where the data are precise", failures)
        call expect(all(noise > 0.0_dp), &
            "reported noise is positive, as the log construction guarantees", &
            failures)
        call expect(model%supports(FORTBO_CAP_MOMENTS), &
            "the heteroskedastic adapter declares moments", failures)
    end subroutine check_heteroskedastic_separates_its_two_uncertainties

    !! The classification adapter presents the *latent* moments, not the class
    !! probability. The latent is Gaussian and unbounded, which is what every
    !! FortBO acquisition integrates against; a probability is neither.
    subroutine check_classification_presents_the_latent(failures)
        integer, intent(inout) :: failures
        type(fortbo_classification_posterior_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 1), probe(3, 1), mean(3), variance(3)
        integer :: labels(10), k

        ! A boundary at zero: negative inputs one class, positive the other.
        do k = 1, 10
            x(k, 1) = -2.0_dp + 0.4_dp*real(k - 1, dp)
            if (x(k, 1) < 0.0_dp) then
                labels(k) = 0
            else
                labels(k) = 1
            end if
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call model%model%fit(x, labels, kernel, status)
        call expect(status%code == FORTNUM_OK, "the classification model fits", &
            failures)
        model%dimension = 1
        model%fitted = .true.

        probe(1, 1) = -1.5_dp
        probe(2, 1) = 0.0_dp
        probe(3, 1) = 1.5_dp
        call model%moments(probe, mean, variance, status)
        call expect(status%code == FORTNUM_OK, "the latent moments evaluate", &
            failures)

        ! A latent, not a probability: it must be signed and may leave [0, 1].
        call expect(mean(1) < 0.0_dp .and. mean(3) > 0.0_dp, &
            "the latent is signed, separating the two classes", failures)
        call expect(abs(mean(2)) < abs(mean(1)) .and. abs(mean(2)) < abs(mean(3)), &
            "the latent is nearest zero at the decision boundary", failures)
        call expect(all(variance >= 0.0_dp), "the latent variance is a variance", &
            failures)

        ! Only marginal moments: the contract must not claim a joint route the
        ! Laplace approximation does not supply here.
        call expect(.not. model%supports(FORTBO_CAP_JOINT_SAMPLE), &
            "the classification adapter claims no joint sampling", failures)
    end subroutine check_classification_presents_the_latent

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_multi_output_posterior_t) :: model, unfitted
        type(fortbo_student_t_posterior_t) :: process
        type(fortbo_batch_samples_t) :: batch
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        real(dp) :: points(3, 1), mean(3), variance(3), wide(3, 2)

        points(:, 1) = [-0.5_dp, 0.0_dp, 0.5_dp]
        call rng_seed(generator, int(1, int64), status)

        call unfitted%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unfitted multi-output adapter is refused", failures)

        call build_multi(model, 1, status)
        model%output = 5
        call model%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an out-of-range output selection is refused", failures)

        model%output = 1
        call model%moments(wide, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a query of the wrong width is refused", failures)

        call process%moments(points, mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unfitted Student-t adapter is refused", failures)

        ! The Student-t adapter does not offer joint sampling, so a batch
        ! acquisition must refuse rather than assume independence.
        call batch%generate(process, points, 10, generator, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a batch refuses against a marginal-only adapter", failures)
    end subroutine check_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_fortml_sparse
