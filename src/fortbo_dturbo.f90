module fortbo_dturbo
    !! DTuRBO mode 2: posterior-derivative local models (ROADMAP BO3T).
    !!
    !! Modes 1 and 3 need no code of their own. Mode 1 is a derivative-
    !! observation surrogate, which `fortbo_fit_from_history` already selects
    !! from the data; mode 3 is gradient-based acquisition optimization inside
    !! the region, which `fortbo_optimize_acquisition` already does. Mode 2 is
    !! the one that needs a policy, and this is it.
    !!
    !! Following Newton-BO (Enhancing Trust-Region Bayesian Optimization via
    !! Newton Methods, arXiv:2508.18423), the local model is built from the
    !! posterior's own derivatives rather than from sampled candidates:
    !!
    !!     g = grad mu + lambda grad sigma
    !!     H = hess mu + lambda hess sigma
    !!
    !! with `lambda` drawn from a standard normal truncated to the non-negative
    !! half line. The next point is the solution of the bound-constrained
    !! quadratic program over the trust region intersected with the unit cube.
    !!
    !! `lambda` is what keeps this an *optimization under uncertainty* rather
    !! than a Newton step on the posterior mean. It is a random exploration
    !! weight: a draw near zero gives a pure exploitation step, a large draw
    !! follows the uncertainty. Truncating to the non-negative side matters —
    !! a negative `lambda` would subtract the standard deviation's gradient and
    !! steer the step *away* from uncertain regions, which is the opposite of
    !! what an acquisition should do.
    !!
    !! Radius adaptation switches to the classical trust-region ratio test
    !!
    !!     rho = actual decrease / predicted decrease,
    !!
    !! expanding on `rho >= eta_1` and shrinking on `rho < eta_0`. This subsumes
    !! the success/failure counter rule: a counter asks only whether the point
    !! improved, while the ratio asks whether the *model* predicted the
    !! improvement. A region whose model is badly wrong but happens to improve
    !! should not be trusted with a larger radius, and the counter rule cannot
    !! tell the difference.
    !!
    !! An indefinite Hessian is not repaired. Negative curvature is information,
    !! and the bound-constrained solve follows it to the boundary; that is the
    !! documented behaviour of `fortbo_quadratic` and mode 2 relies on it rather
    !! than working around it.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_uniform
    use fortbo_posterior, only: fortbo_posterior_t, FORTBO_CAP_MOMENT_GRADIENT, &
        FORTBO_CAP_MOMENT_HESSIAN
    use fortbo_trust_region, only: fortbo_trust_region_t, FORTBO_TR_UNCHANGED, &
        FORTBO_TR_EXPANDED, FORTBO_TR_SHRANK, FORTBO_TR_EXHAUSTED
    use fortbo_quadratic, only: fortbo_solve_quadratic_subproblem, &
        fortbo_quadratic_value
    use fortbo_normal, only: fortbo_half_normal
    implicit none
    private

    public :: fortbo_dturbo_lambda
    public :: fortbo_dturbo_local_model
    public :: fortbo_dturbo_step
    public :: fortbo_dturbo_ratio_update

    !! Classical trust-region ratio thresholds. Shrink below `ETA_0`, expand at
    !! or above `ETA_1`, leave the radius alone in between.
    real(dp), parameter, public :: FORTBO_DTURBO_ETA_0 = 0.1_dp
    real(dp), parameter, public :: FORTBO_DTURBO_ETA_1 = 0.75_dp

    !! Below this predicted decrease the ratio is meaningless: dividing a tiny
    !! actual decrease by a tiny predicted one amplifies rounding into a verdict
    !! about the model. Such a step counts as no information and leaves the
    !! radius unchanged.
    real(dp), parameter, public :: FORTBO_DTURBO_PREDICTION_FLOOR = 1.0e-14_dp

    !! Restart when the posterior mean's gradient falls below this: the model
    !! believes it is at a stationary point, and shrinking further only refines
    !! a place it has already given up on.
    real(dp), parameter, public :: FORTBO_DTURBO_GRADIENT_TOLERANCE = 1.0e-8_dp

contains

    !! Draw the exploration weight. Kept separate from the step so a caller can
    !! replay a run from a recorded `lambda`, and so the truncation can be
    !! tested on its own.
    subroutine fortbo_dturbo_lambda(generator, lambda, status)
        type(rng_t), intent(inout) :: generator
        real(dp), intent(out) :: lambda
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: uniform

        call rng_uniform(generator, uniform)
        lambda = fortbo_half_normal(uniform)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_dturbo_lambda

    !! Assemble the local model at `point`.
    !!
    !! The posterior must supply both moment gradients and both moment Hessians.
    !! A surrogate that has only the mean's curvature cannot build this model,
    !! and says so by capability rather than by silently dropping the `lambda`
    !! term — which would turn an exploring policy into a Newton method on the
    !! mean without anything in the output saying so.
    subroutine fortbo_dturbo_local_model(posterior, point, lambda, gradient, &
            hessian, status)
        class(fortbo_posterior_t), intent(in) :: posterior
        real(dp), intent(in) :: point(:)
        real(dp), intent(in) :: lambda
        real(dp), intent(out) :: gradient(:)
        real(dp), intent(out) :: hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: query(:, :)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        real(dp), allocatable :: mean_hessian(:, :), sd_hessian(:, :)
        integer :: d

        gradient = 0.0_dp
        hessian = 0.0_dp
        d = size(point)
        if (d < 1 .or. size(gradient) /= d .or. size(hessian, 1) /= d .or. &
            size(hessian, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: local model shapes disagree")
            return
        end if
        if (lambda < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: lambda must not be negative")
            return
        end if
        if (.not. posterior%supports(FORTBO_CAP_MOMENT_GRADIENT) .or. &
            .not. posterior%supports(FORTBO_CAP_MOMENT_HESSIAN)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo dturbo: posterior lacks moment derivatives")
            return
        end if

        allocate (query(1, d))
        allocate (mean_gradient(1, d), sd_gradient(1, d))
        allocate (mean_hessian(d, d), sd_hessian(d, d))
        query(1, :) = point
        call posterior%moment_gradient(query, mean_gradient, sd_gradient, status)
        if (status%code /= FORTNUM_OK) return
        call posterior%moment_hessian(point, mean_hessian, sd_hessian, status)
        if (status%code /= FORTNUM_OK) return

        gradient = mean_gradient(1, :) + lambda*sd_gradient(1, :)
        hessian = mean_hessian + lambda*sd_hessian
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_dturbo_local_model

    !! One mode-2 proposal: build the local model at the region's center and
    !! minimize it over the region intersected with the unit cube.
    !!
    !! `predicted_decrease` is returned as a positive quantity — the amount the
    !! model says the step will gain — because that is what the ratio test
    !! divides by, and returning a signed model value would leave every caller
    !! to remember the convention.
    subroutine fortbo_dturbo_step(posterior, region, lengthscales, lambda, point, &
            predicted_decrease, status, indefinite, stationary)
        class(fortbo_posterior_t), intent(in) :: posterior
        type(fortbo_trust_region_t), intent(in) :: region
        real(dp), intent(in) :: lengthscales(:)
        real(dp), intent(in) :: lambda
        real(dp), intent(out) :: point(:)
        real(dp), intent(out) :: predicted_decrease
        type(fortnum_status_t), intent(out) :: status
        logical, intent(out), optional :: indefinite
        !! True when the posterior mean's gradient is below tolerance, which is
        !! one of the two restart conditions in the specification.
        logical, intent(out), optional :: stationary
        real(dp), allocatable :: gradient(:), hessian(:, :)
        real(dp), allocatable :: lower(:), upper(:), step(:), origin(:)
        real(dp), allocatable :: mean_gradient(:, :), sd_gradient(:, :)
        real(dp), allocatable :: query(:, :)
        real(dp) :: model_value
        integer :: d

        d = region%n_inputs
        point = 0.0_dp
        predicted_decrease = 0.0_dp
        if (present(indefinite)) indefinite = .false.
        if (present(stationary)) stationary = .false.

        if (d < 1 .or. .not. region%active) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: region is not active")
            return
        end if
        if (size(point) /= d .or. size(lengthscales) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: step shapes disagree with the region")
            return
        end if

        allocate (gradient(d), hessian(d, d))
        call fortbo_dturbo_local_model(posterior, region%center, lambda, gradient, &
            hessian, status)
        if (status%code /= FORTNUM_OK) return

        if (present(stationary)) then
            allocate (query(1, d), mean_gradient(1, d), sd_gradient(1, d))
            query(1, :) = region%center
            call posterior%moment_gradient(query, mean_gradient, sd_gradient, status)
            if (status%code /= FORTNUM_OK) return
            stationary = sqrt(sum(mean_gradient(1, :)**2)) < &
                FORTBO_DTURBO_GRADIENT_TOLERANCE
        end if

        ! The subproblem is posed in the step variable, so the box is the region
        ! bounds shifted to the center. Intersecting with the unit cube happens
        ! inside `region%bounds`, which already clips.
        allocate (lower(d), upper(d), step(d), origin(d))
        call region%bounds(lengthscales, lower, upper, status)
        if (status%code /= FORTNUM_OK) return
        lower = lower - region%center
        upper = upper - region%center
        origin = 0.0_dp

        call fortbo_solve_quadratic_subproblem(gradient, hessian, lower, upper, &
            origin, step, model_value, status, &
            indefinite)
        if (status%code /= FORTNUM_OK) return

        point = region%center + step
        ! The model is zero at the center by construction, so its value at the
        ! step *is* the negative of the predicted decrease. A model that does
        ! not predict a decrease predicts none, rather than a negative one.
        predicted_decrease = max(-model_value, 0.0_dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_dturbo_step

    !! Classical ratio-test radius adaptation.
    !!
    !! `actual_decrease` and `predicted_decrease` are both positive-is-good.
    !! When the prediction is too small to divide by, the step carries no
    !! information about the model's quality and the radius is left alone: the
    !! alternative is to let rounding decide whether a region grows.
    subroutine fortbo_dturbo_ratio_update(region, actual_decrease, &
            predicted_decrease, event, status)
        type(fortbo_trust_region_t), intent(inout) :: region
        real(dp), intent(in) :: actual_decrease
        real(dp), intent(in) :: predicted_decrease
        integer, intent(out) :: event
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: ratio

        event = FORTBO_TR_UNCHANGED
        if (.not. region%active) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: region is not active")
            return
        end if
        if (predicted_decrease < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo dturbo: predicted decrease must not be negative")
            return
        end if

        region%batches = region%batches + 1
        if (predicted_decrease <= FORTBO_DTURBO_PREDICTION_FLOOR) then
            region%last_event = FORTBO_TR_UNCHANGED
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        ratio = actual_decrease/predicted_decrease
        if (ratio >= FORTBO_DTURBO_ETA_1) then
            region%length = min(2.0_dp*region%length, region%length_max)
            event = FORTBO_TR_EXPANDED
        else if (ratio < FORTBO_DTURBO_ETA_0) then
            region%length = 0.5_dp*region%length
            event = FORTBO_TR_SHRANK
            if (region%length < region%length_min) then
                region%active = .false.
                event = FORTBO_TR_EXHAUSTED
            end if
        end if
        region%last_event = event
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortbo_dturbo_ratio_update

end module fortbo_dturbo
