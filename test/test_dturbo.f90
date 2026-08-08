program test_dturbo
    !! BO3T: DTuRBO mode 2, posterior-derivative local models.
    !!
    !! Oracles:
    !!
    !!   * the local model is checked against the moments it is built from,
    !!     recomputed independently in the test — a `lambda` applied to the
    !!     wrong term, or to the mean instead of the standard deviation, fails;
    !!   * `lambda`'s distribution is checked against the truncated normal's own
    !!     definition by comparing the empirical CDF with `2*Phi(x) - 1`, with
    !!     the tolerance from the sampling error rather than tuned;
    !!   * the step must minimize the model over the region, checked against a
    !!     dense grid, and must lie inside the region and the unit cube;
    !!   * the ratio test is driven to both sides of both thresholds, and the
    !!     case that distinguishes it from the counter rule is checked directly:
    !!     a step that improves the objective while the model predicted far more
    !!     must *shrink* the region, where a success counter would have grown it;
    !!   * mode 2 must actually optimize. On a quadratic bowl with an exact
    !!     posterior, repeated steps must converge to the minimum.

    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed
    use fortbo_trust_region, only: fortbo_trust_region_t, FORTBO_TR_UNCHANGED, &
        FORTBO_TR_EXPANDED, FORTBO_TR_SHRANK, FORTBO_TR_EXHAUSTED
    use fortbo_quadratic, only: fortbo_quadratic_value
    use fortbo_normal, only: fortbo_inverse_normal, &
        fortbo_symmetric_truncated_normal
    use fortbo_dturbo, only: fortbo_dturbo_lambda, fortbo_dturbo_local_model, &
        fortbo_dturbo_step, fortbo_dturbo_ratio_update, FORTBO_DTURBO_ETA_0, &
        FORTBO_DTURBO_ETA_1
    use fortbo_test_posteriors, only: curved_posterior_t, &
        moments_only_posterior_t
    implicit none

    integer :: failures

    failures = 0
    call check_quantile_function(failures)
    call check_lambda_distribution(failures)
    call check_local_model_composition(failures)
    call check_step_minimizes_the_model(failures)
    call check_ratio_test_thresholds(failures)
    call check_ratio_beats_the_counter_rule(failures)
    call check_mode_two_converges(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_dturbo: PASS"
    else
        print *, "test_dturbo: FAIL", failures
        error stop 1
    end if

contains

    !! The quantile function against the CDF it inverts. `erf` is the
    !! independent route: Phi(z) = (1 + erf(z/sqrt(2)))/2.
    subroutine check_quantile_function(failures)
        integer, intent(inout) :: failures
        real(dp) :: p, z, recovered
        integer :: k
        logical :: inverts

        inverts = .true.
        do k = 1, 199
            p = real(k, dp)/200.0_dp
            z = fortbo_inverse_normal(p)
            recovered = 0.5_dp*(1.0_dp + erf(z/sqrt(2.0_dp)))
            if (abs(recovered - p) > 1.0e-8_dp) inverts = .false.
        end do
        call expect(inverts, "the quantile function inverts the normal CDF", &
            failures)
        call expect(abs(fortbo_inverse_normal(0.5_dp)) < 1.0e-12_dp, &
            "the median is zero", failures)

        ! The symmetric truncation stays inside its bound and is symmetric
        ! about zero, which the half-normal used previously was not.
        call expect(abs(fortbo_symmetric_truncated_normal(0.5_dp, 1.0_dp)) &
            < 1.0e-12_dp, "the truncated normal's median is zero", failures)
        call expect(fortbo_symmetric_truncated_normal(0.0_dp, 1.0_dp) &
            >= -1.0_dp .and. fortbo_symmetric_truncated_normal(1.0_dp, 1.0_dp) &
            <= 1.0_dp, "the truncated normal respects its bound", failures)
        call expect(abs(fortbo_symmetric_truncated_normal(0.25_dp, 1.0_dp) &
            + fortbo_symmetric_truncated_normal(0.75_dp, 1.0_dp)) < 1.0e-12_dp, &
            "the truncated normal is symmetric about zero", failures)
    end subroutine check_quantile_function

    !! Compare the empirical CDF of sampled `lambda` against the truncated
    !! normal's own CDF on `(-1, 1)`, which is
    !! `(Phi(x) - Phi(-1)) / (Phi(1) - Phi(-1))`.
    !!
    !! Negative draws must occur. An earlier version truncated to the
    !! non-negative half line on the reasoning that a negative `lambda` steers
    !! away from uncertainty; the paper truncates to `(-1, 1)` explicitly to
    !! ensure local convergence, and the bound limits deviation from the mean's
    !! Newton model in *either* direction.
    subroutine check_lambda_distribution(failures)
        integer, intent(inout) :: failures
        type(rng_t) :: generator
        type(fortnum_status_t) :: status
        integer, parameter :: n_samples = 200000
        real(dp) :: lambda, exact, empirical, standard_error, threshold
        real(dp) :: low_mass, high_mass
        integer :: k, below, outside, negatives, grid
        logical :: matches

        low_mass = 0.5_dp*(1.0_dp + erf(-1.0_dp/sqrt(2.0_dp)))
        high_mass = 0.5_dp*(1.0_dp + erf(1.0_dp/sqrt(2.0_dp)))

        matches = .true.
        outside = 0
        negatives = 0
        do grid = 1, 4
            threshold = -0.75_dp + 0.5_dp*real(grid - 1, dp)
            call rng_seed(generator, int(8675309, int64), status)
            below = 0
            do k = 1, n_samples
                call fortbo_dturbo_lambda(generator, lambda, status)
                if (abs(lambda) > 1.0_dp) outside = outside + 1
                if (lambda < 0.0_dp) negatives = negatives + 1
                if (lambda <= threshold) below = below + 1
            end do
            empirical = real(below, dp)/real(n_samples, dp)
            exact = (0.5_dp*(1.0_dp + erf(threshold/sqrt(2.0_dp))) - low_mass) &
                /(high_mass - low_mass)
            standard_error = sqrt(exact*(1.0_dp - exact)/real(n_samples, dp))
            if (abs(empirical - exact) > 5.0_dp*standard_error) matches = .false.
        end do
        call expect(matches, &
            "lambda follows the truncated normal on (-1, 1) it is specified as", &
            failures)
        call expect(outside == 0, "lambda never leaves its truncation bound", &
            failures)
        call expect(negatives > 0, &
            "negative lambda occurs, as the two-sided truncation requires", &
            failures)
    end subroutine check_lambda_distribution

    !! The model must be exactly `mean + lambda*sd` in both derivatives, with
    !! the moments taken from the posterior independently here.
    subroutine check_local_model_composition(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortnum_status_t) :: status
        real(dp) :: point(2), gradient(2), hessian(2, 2)
        real(dp) :: zero_gradient(2), zero_hessian(2, 2)
        real(dp) :: query(1, 2), mean_gradient(1, 2), sd_gradient(1, 2)
        real(dp) :: mean_hessian(2, 2), sd_hessian(2, 2)
        ! Inside the (-1, 1) truncation the paper specifies.
        real(dp), parameter :: lambda = 0.75_dp

        posterior%dimension = 2
        point = [0.3_dp, -0.4_dp]
        query(1, :) = point

        call fortbo_dturbo_local_model(posterior, point, lambda, gradient, hessian, &
            status)
        call expect(status%code == FORTNUM_OK, "the local model builds", failures)

        call posterior%moment_gradient(query, mean_gradient, sd_gradient, status)
        call posterior%moment_hessian(point, mean_hessian, sd_hessian, status)
        call expect(maxval(abs(gradient - (mean_gradient(1, :) &
            + lambda*sd_gradient(1, :)))) < 1.0e-14_dp, &
            "the model gradient is the mean's plus lambda times the sd's", &
            failures)
        call expect(maxval(abs(hessian - (mean_hessian + lambda*sd_hessian))) &
            < 1.0e-14_dp, &
            "the model Hessian is the mean's plus lambda times the sd's", failures)

        ! Zero lambda must give the pure mean model, which is the property that
        ! makes lambda an exploration weight rather than an arbitrary mixture.
        call fortbo_dturbo_local_model(posterior, point, 0.0_dp, zero_gradient, &
            zero_hessian, status)
        call expect(maxval(abs(zero_gradient - mean_gradient(1, :))) < 1.0e-14_dp &
            .and. maxval(abs(zero_hessian - mean_hessian)) < 1.0e-14_dp, &
            "lambda zero recovers a Newton step on the posterior mean", failures)

        call expect(maxval(abs(gradient - zero_gradient)) > 1.0e-6_dp, &
            "a positive lambda changes the step", failures)
    end subroutine check_local_model_composition

    !! Against a dense grid over the region, in the step variable.
    subroutine check_step_minimizes_the_model(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(2), point(2), predicted
        real(dp) :: gradient(2), hessian(2, 2)
        real(dp) :: lower(2), upper(2), trial(2), best_value, step(2)
        real(dp), parameter :: lambda = 0.9_dp
        integer, parameter :: resolution = 400
        integer :: i, j
        logical :: indefinite

        posterior%dimension = 2
        lengthscales = 1.0_dp
        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        call expect(status%code == FORTNUM_OK, "the region places", failures)

        call fortbo_dturbo_step(posterior, region, lengthscales, lambda, point, &
            predicted, status, indefinite)
        call expect(status%code == FORTNUM_OK, "the mode-2 step solves", failures)

        call region%bounds(lengthscales, lower, upper, status)
        call expect(all(point >= lower - 1.0e-12_dp) .and. &
            all(point <= upper + 1.0e-12_dp), &
            "the step stays inside the region", failures)
        call expect(all(point >= 0.0_dp) .and. all(point <= 1.0_dp), &
            "the step stays inside the unit cube", failures)

        call fortbo_dturbo_local_model(posterior, region%center, lambda, gradient, &
            hessian, status)
        step = point - region%center
        best_value = huge(1.0_dp)
        do i = 0, resolution
            do j = 0, resolution
                trial(1) = (lower(1) - region%center(1)) &
                    + (upper(1) - lower(1))*real(i, dp)/real(resolution, dp)
                trial(2) = (lower(2) - region%center(2)) &
                    + (upper(2) - lower(2))*real(j, dp)/real(resolution, dp)
                best_value = min(best_value, &
                    fortbo_quadratic_value(gradient, hessian, trial))
            end do
        end do
        call expect(fortbo_quadratic_value(gradient, hessian, step) &
            <= best_value + 1.0e-8_dp, &
            "the step matches or beats a dense grid over the region", failures)

        call expect(predicted >= 0.0_dp, &
            "the predicted decrease is reported positive-is-good", failures)
        call expect(abs(predicted + fortbo_quadratic_value(gradient, hessian, step)) &
            < 1.0e-10_dp .or. predicted == 0.0_dp, &
            "the predicted decrease is the model's own gain", failures)
    end subroutine check_step_minimizes_the_model

    subroutine check_ratio_test_thresholds(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: before
        integer :: event, attempt

        ! Above eta_1: expand.
        call fresh_region(region, status)
        before = region%length
        call fortbo_dturbo_ratio_update(region, FORTBO_DTURBO_ETA_1*1.0_dp, 1.0_dp, &
            event, status)
        call expect(event == FORTBO_TR_EXPANDED .and. region%length > before, &
            "a ratio at eta_1 expands the region", failures)

        ! Between the thresholds: unchanged.
        call fresh_region(region, status)
        before = region%length
        call fortbo_dturbo_ratio_update(region, 0.5_dp, 1.0_dp, event, status)
        call expect(event == FORTBO_TR_UNCHANGED .and. region%length == before, &
            "a middling ratio leaves the radius alone", failures)

        ! Below eta_0: shrink.
        call fresh_region(region, status)
        before = region%length
        call fortbo_dturbo_ratio_update(region, FORTBO_DTURBO_ETA_0*0.5_dp, 1.0_dp, &
            event, status)
        call expect(event == FORTBO_TR_SHRANK .and. region%length < before, &
            "a poor ratio shrinks the region", failures)

        ! A prediction too small to divide by carries no information.
        call fresh_region(region, status)
        before = region%length
        call fortbo_dturbo_ratio_update(region, 1.0_dp, 0.0_dp, event, status)
        call expect(event == FORTBO_TR_UNCHANGED .and. region%length == before, &
            "a vanishing prediction leaves the radius alone", failures)

        ! Repeated shrinking must exhaust the region rather than shrink forever.
        call fresh_region(region, status)
        do attempt = 1, 40
            call fortbo_dturbo_ratio_update(region, 0.0_dp, 1.0_dp, event, status)
            if (.not. region%active) exit
        end do
        call expect(.not. region%active, &
            "repeated poor ratios exhaust the region", failures)
    end subroutine check_ratio_test_thresholds

    !! The case that separates the ratio test from the success/failure counter.
    !! The step improves the objective, so a counter rule would score a success
    !! and eventually expand. The model predicted ten times that improvement, so
    !! the model is badly wrong and the region must shrink instead.
    subroutine check_ratio_beats_the_counter_rule(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: before
        integer :: event

        call fresh_region(region, status)
        before = region%length
        call fortbo_dturbo_ratio_update(region, 0.05_dp, 1.0_dp, event, status)
        call expect(event == FORTBO_TR_SHRANK, &
            "an improving step with a badly wrong model still shrinks", failures)
        call expect(region%length < before, &
            "the radius follows the model's quality, not the outcome alone", &
            failures)
    end subroutine check_ratio_beats_the_counter_rule

    !! Mode 2 must optimize, not merely produce well-formed steps. The curved
    !! test posterior has a known minimizer; repeated steps with a small lambda
    !! must approach it.
    subroutine check_mode_two_converges(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(2), point(2), predicted
        real(dp) :: start_distance, end_distance
        real(dp) :: minimizer(2)
        integer :: step

        posterior%dimension = 2
        lengthscales = 1.0_dp
        call region%initialize(2, 1, status)
        call region%restart([0.9_dp, 0.1_dp], 1.0_dp, status)

        minimizer = posterior%minimizer()
        start_distance = sqrt(sum((region%center - minimizer)**2))

        do step = 1, 30
            if (.not. region%active) exit
            call fortbo_dturbo_step(posterior, region, lengthscales, 0.0_dp, point, &
                predicted, status)
            if (status%code /= FORTNUM_OK) exit
            call region%restart(point, 0.0_dp, status)
            if (status%code /= FORTNUM_OK) exit
        end do

        end_distance = sqrt(sum((region%center - minimizer)**2))
        call expect(end_distance < 0.1_dp*start_distance, &
            "mode 2 converges toward the posterior mean's minimizer", failures)
    end subroutine check_mode_two_converges

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(curved_posterior_t) :: posterior
        type(moments_only_posterior_t) :: plain
        type(fortbo_trust_region_t) :: region, unplaced
        type(fortnum_status_t) :: status
        real(dp) :: gradient(2), hessian(2, 2), point(2), predicted
        real(dp) :: lengthscales(2)
        integer :: event

        posterior%dimension = 2
        plain%dimension = 2
        lengthscales = 1.0_dp

        ! A negative lambda inside the bound is legitimate.
        call fortbo_dturbo_local_model(posterior, [0.1_dp, 0.2_dp], -0.5_dp, &
            gradient, hessian, status)
        call expect(status%code == FORTNUM_OK, &
            "a negative lambda inside the bound is accepted", failures)

        call fortbo_dturbo_local_model(posterior, [0.1_dp, 0.2_dp], 1.5_dp, &
            gradient, hessian, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a lambda outside the truncation bound is refused", failures)

        ! A posterior with moments but no derivatives cannot build this model,
        ! and must say so rather than silently dropping the lambda term.
        call fortbo_dturbo_local_model(plain, [0.1_dp, 0.2_dp], 1.0_dp, gradient, &
            hessian, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "a posterior without moment derivatives refuses by capability", &
            failures)

        call region%initialize(2, 1, status)
        call fortbo_dturbo_step(posterior, region, lengthscales, 1.0_dp, point, &
            predicted, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unplaced region is refused", failures)

        call fresh_region(region, status)
        call fortbo_dturbo_ratio_update(region, 1.0_dp, -1.0_dp, event, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a negative predicted decrease is refused", failures)

        call fortbo_dturbo_ratio_update(unplaced, 1.0_dp, 1.0_dp, event, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "updating an inactive region is refused", failures)
    end subroutine check_refusals

    subroutine fresh_region(region, status)
        type(fortbo_trust_region_t), intent(out) :: region
        type(fortnum_status_t), intent(out) :: status

        call region%initialize(2, 1, status)
        if (status%code /= FORTNUM_OK) return
        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
    end subroutine fresh_region

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_dturbo
