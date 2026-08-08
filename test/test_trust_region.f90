program test_trust_region
    !! BO3T: the TuRBO trust-region state machine.
    !!
    !! Oracles:
    !!   * the volume invariant is checked directly — the product of the side
    !!     lengths must equal `L^d` for arbitrary lengthscales, which is the
    !!     property the geometric-mean normalization exists to guarantee and
    !!     which a normalization by sum or maximum would silently break;
    !!   * the adaptation rule is checked against an independent counter model
    !!     written in the test, driven by a scripted sequence of batch outcomes.
    !!     The test decides what should happen from the paper's rule and
    !!     compares; it does not read the implementation's counters back;
    !!   * the reference constants are asserted explicitly, so a future edit
    !!     that "tunes" them has to change a test that says where they came from.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortbo_trust_region, only: fortbo_trust_region_t, FORTBO_TR_LENGTH_INIT, &
        FORTBO_TR_LENGTH_MIN, FORTBO_TR_LENGTH_MAX, FORTBO_TR_SUCCESS_TOLERANCE, &
        FORTBO_TR_EXPANDED, FORTBO_TR_SHRANK, FORTBO_TR_EXHAUSTED, &
        FORTBO_TR_UNCHANGED
    implicit none

    interface
        pure subroutine fortbo_generated_trust_region_leaf(log_lengthscale, &
                log_mean, base_length, side_length, side_d_log_lengthscale, &
                side_d_log_mean, side_d_base_length)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: log_lengthscale, log_mean, base_length
            real(real64), intent(out) :: side_length, side_d_log_lengthscale
            real(real64), intent(out) :: side_d_log_mean, side_d_base_length
        end subroutine fortbo_generated_trust_region_leaf
    end interface

    integer :: failures

    failures = 0
    call check_reference_constants(failures)
    call check_failure_tolerance(failures)
    call check_volume_invariant(failures)
    call check_lengthscale_shaping(failures)
    call check_bounds_are_clipped(failures)
    call check_expansion_rule(failures)
    call check_shrink_and_exhaustion(failures)
    call check_counter_model(failures)
    call check_center_tracking(failures)
    call check_refusals(failures)
    call check_generated_leaf_derivatives(failures)

    if (failures == 0) then
        print *, "test_trust_region: PASS"
    else
        print *, "test_trust_region: FAIL", failures
        error stop 1
    end if

contains

    !! These values come from Eriksson et al. and the reference implementation.
    !! Pinning them here is the difference between reproducing TuRBO and
    !! reproducing something that resembles it.
    subroutine check_reference_constants(failures)
        integer, intent(inout) :: failures

        call expect(FORTBO_TR_LENGTH_INIT == 0.8_dp, "L_init is 0.8", failures)
        call expect(FORTBO_TR_LENGTH_MIN == 0.5_dp**7, "L_min is 2^-7", failures)
        call expect(FORTBO_TR_LENGTH_MAX == 1.6_dp, "L_max is 1.6", failures)
        call expect(FORTBO_TR_SUCCESS_TOLERANCE == 3, "tau_succ is 3", failures)
    end subroutine check_reference_constants

    !! tau_fail = ceil(max(4, d) / q).
    subroutine check_failure_tolerance(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        integer :: d, q, expected
        logical :: agrees

        agrees = .true.
        do d = 1, 60
            do q = 1, 20
                call region%initialize(d, q, status)
                expected = (max(4, d) + q - 1)/q
                if (region%failure_tolerance /= max(expected, 1)) agrees = .false.
            end do
        end do
        call expect(agrees, "tau_fail is ceil(max(4,d)/q) everywhere", failures)

        call region%initialize(20, 1, status)
        call expect(region%failure_tolerance == 20, &
            "a sequential run tolerates d failures", failures)
        call expect(status%code == FORTNUM_OK, "initialization succeeds", failures)
    end subroutine check_failure_tolerance

    !! The product of the side lengths must be exactly L^d whatever the
    !! lengthscales are. This is the invariant that makes L comparable.
    subroutine check_volume_invariant(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        integer, parameter :: d = 8
        real(dp) :: lengthscales(d), lengths(d)
        real(dp) :: log_product, expected
        integer :: trial, j
        logical :: invariant

        call region%initialize(d, 4, status)
        invariant = .true.
        do trial = 1, 20
            do j = 1, d
                lengthscales(j) = 0.05_dp + 3.0_dp &
                    *abs(sin(0.9_dp*real(trial, dp) + real(j, dp)))
            end do
            region%length = 0.1_dp + 0.05_dp*real(trial, dp)
            call region%side_lengths(lengthscales, lengths, status)
            if (status%code /= FORTNUM_OK) invariant = .false.
            log_product = sum(log(lengths))
            expected = real(d, dp)*log(region%length)
            if (abs(log_product - expected) > 1.0e-10_dp) invariant = .false.
        end do
        call expect(invariant, "the side lengths always enclose volume L^d", failures)
    end subroutine check_volume_invariant

    !! Equal lengthscales give a cube; a long lengthscale gives a
    !! proportionally longer side.
    subroutine check_lengthscale_shaping(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: equal(3), skewed(3), lengths(3)

        call region%initialize(3, 4, status)
        region%length = 0.5_dp

        equal = [2.0_dp, 2.0_dp, 2.0_dp]
        call region%side_lengths(equal, lengths, status)
        call expect(maxval(abs(lengths - 0.5_dp)) < 1.0e-12_dp, &
            "equal lengthscales give a cube of side L", failures)

        skewed = [4.0_dp, 1.0_dp, 1.0_dp]
        call region%side_lengths(skewed, lengths, status)
        call expect(abs(lengths(1)/lengths(2) - 4.0_dp) < 1.0e-12_dp, &
            "side lengths are proportional to the lengthscales", failures)
        call expect(abs(lengths(2) - lengths(3)) < 1.0e-12_dp, &
            "equal lengthscales keep equal sides", failures)

        ! High dimension: the direct product of lengthscales would underflow,
        ! so this is where a naive implementation returns zeros or NaNs.
        block
            type(fortbo_trust_region_t) :: wide
            real(dp) :: many(400), wide_lengths(400)
            integer :: j

            call wide%initialize(400, 100, status)
            do j = 1, 400
                many(j) = 1.0e-3_dp*real(j, dp)
            end do
            call wide%side_lengths(many, wide_lengths, status)
            call expect(status%code == FORTNUM_OK, &
                "four hundred dimensions still resolve", failures)
            call expect(all(wide_lengths > 0.0_dp), &
                "no side length underflows to zero", failures)
            call expect(all(wide_lengths == wide_lengths), &
                "no side length becomes NaN", failures)
        end block
    end subroutine check_lengthscale_shaping

    subroutine check_bounds_are_clipped(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(2), lower(2), upper(2)

        call region%initialize(2, 4, status)
        lengthscales = [1.0_dp, 1.0_dp]

        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        region%length = 0.4_dp
        call region%bounds(lengthscales, lower, upper, status)
        call expect(abs(lower(1) - 0.3_dp) < 1.0e-12_dp, "the lower bound is centered", &
            failures)
        call expect(abs(upper(1) - 0.7_dp) < 1.0e-12_dp, "the upper bound is centered", &
            failures)

        call region%restart([0.02_dp, 0.98_dp], 1.0_dp, status)
        region%length = 0.4_dp
        call region%bounds(lengthscales, lower, upper, status)
        call expect(lower(1) == 0.0_dp, "the region is clipped at the lower face", &
            failures)
        call expect(upper(2) == 1.0_dp, "the region is clipped at the upper face", &
            failures)
        call expect(all(lower >= 0.0_dp) .and. all(upper <= 1.0_dp), &
            "the region never leaves the unit cube", failures)
    end subroutine check_bounds_are_clipped

    !! tau_succ consecutive improvements double the length, capped at L_max.
    subroutine check_expansion_rule(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: inputs(1, 2), values(1)
        integer :: i

        call region%initialize(2, 1, status)
        call region%restart([0.5_dp, 0.5_dp], 10.0_dp, status)
        inputs = reshape([0.5_dp, 0.5_dp], [1, 2])

        do i = 1, FORTBO_TR_SUCCESS_TOLERANCE - 1
            values = [10.0_dp - real(i, dp)]
            call region%observe_batch(inputs, values, status)
            call expect(abs(region%length - FORTBO_TR_LENGTH_INIT) < 1.0e-14_dp, &
                "the length holds before the success tolerance", failures)
            call expect(region%last_event == FORTBO_TR_UNCHANGED, &
                "no resize event before the tolerance", failures)
        end do

        values = [10.0_dp - real(FORTBO_TR_SUCCESS_TOLERANCE, dp)]
        call region%observe_batch(inputs, values, status)
        call expect(abs(region%length - 2.0_dp*FORTBO_TR_LENGTH_INIT) < 1.0e-14_dp, &
            "the length doubles at the success tolerance", failures)
        call expect(region%last_event == FORTBO_TR_EXPANDED, &
            "the expansion is recorded", failures)

        ! Already at 1.6 = L_max; a further run of successes must not exceed it.
        do i = 1, 3*FORTBO_TR_SUCCESS_TOLERANCE
            values = [region%center_value - 1.0_dp]
            call region%observe_batch(inputs, values, status)
        end do
        call expect(region%length <= FORTBO_TR_LENGTH_MAX + 1.0e-14_dp, &
            "the length never exceeds L_max", failures)
    end subroutine check_expansion_rule

    subroutine check_shrink_and_exhaustion(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: inputs(1, 1), values(1)
        real(dp) :: previous
        integer :: batch
        logical :: exhausted

        call region%initialize(1, 4, status)
        call region%restart([0.5_dp], 0.0_dp, status)
        inputs = reshape([0.5_dp], [1, 1])
        values = [1.0_dp]

        call expect(region%failure_tolerance == 1, &
            "a batch of four in one dimension shrinks after one failure", &
            failures)

        previous = region%length
        call region%observe_batch(inputs, values, status)
        call expect(abs(region%length - 0.5_dp*previous) < 1.0e-14_dp, &
            "a failing batch halves the length", failures)
        call expect(region%last_event == FORTBO_TR_SHRANK, "the shrink is recorded", &
            failures)

        exhausted = .false.
        do batch = 1, 40
            call region%observe_batch(inputs, values, status)
            if (status%code /= FORTNUM_OK) exit
            if (.not. region%active) then
                exhausted = .true.
                exit
            end if
        end do
        call expect(exhausted, "repeated failure exhausts the region", failures)
        call expect(region%last_event == FORTBO_TR_EXHAUSTED, &
            "exhaustion is recorded", failures)
        call expect(region%length < FORTBO_TR_LENGTH_MIN, &
            "the region is exhausted only below L_min", failures)

        call region%observe_batch(inputs, values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an exhausted region refuses further batches", failures)

        call region%restart([0.25_dp], 5.0_dp, status)
        call expect(status%code == FORTNUM_OK, "an exhausted region can restart", &
            failures)
        call expect(abs(region%length - FORTBO_TR_LENGTH_INIT) < 1.0e-14_dp, &
            "a restart resets the length", failures)
        call expect(region%active, "a restart reactivates the region", failures)
        call expect(region%restarts == 1, "the restart is counted", failures)
    end subroutine check_shrink_and_exhaustion

    !! Independent counter model. The test tracks what the paper's rule says the
    !! length should be over a scripted outcome sequence and compares.
    subroutine check_counter_model(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: inputs(1, 1), values(1)
        real(dp) :: model_length, incumbent
        integer :: model_success, model_failure
        integer :: step
        logical :: improves, agrees

        call region%initialize(3, 1, status)
        call region%restart([0.5_dp], 100.0_dp, status)
        inputs = reshape([0.5_dp], [1, 1])

        model_length = FORTBO_TR_LENGTH_INIT
        model_success = 0
        model_failure = 0
        incumbent = 100.0_dp
        agrees = .true.

        do step = 1, 60
            ! A deterministic but irregular success pattern.
            improves = mod(step*step + step/3, 5) < 3
            if (improves) then
                values = [incumbent - 1.0_dp]
            else
                values = [incumbent + 1.0_dp]
            end if

            call region%observe_batch(inputs, values, status)
            if (status%code /= FORTNUM_OK) exit

            if (improves) then
                incumbent = incumbent - 1.0_dp
                model_success = model_success + 1
                model_failure = 0
            else
                model_failure = model_failure + 1
                model_success = 0
            end if
            if (model_success >= region%success_tolerance) then
                model_length = min(FORTBO_TR_LENGTH_MAX, 2.0_dp*model_length)
                model_success = 0
                model_failure = 0
            else if (model_failure >= region%failure_tolerance) then
                model_length = 0.5_dp*model_length
                model_success = 0
                model_failure = 0
            end if

            if (abs(region%length - model_length) > 1.0e-12_dp) agrees = .false.
            if (.not. region%active) exit
        end do

        call expect(agrees, "the length matches an independent counter model", failures)
    end subroutine check_counter_model

    !! The center must follow the best point of an improving batch, and must not
    !! move when the batch fails to improve.
    subroutine check_center_tracking(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: inputs(3, 2), values(3), before(2)

        call region%initialize(2, 3, status)
        call region%restart([0.5_dp, 0.5_dp], 10.0_dp, status)

        inputs = reshape([0.1_dp, 0.2_dp, 0.3_dp, 0.9_dp, 0.8_dp, 0.7_dp], [3, 2])
        values = [5.0_dp, 2.0_dp, 7.0_dp]
        call region%observe_batch(inputs, values, status)
        call expect(abs(region%center_value - 2.0_dp) < 1.0e-14_dp, &
            "the incumbent takes the batch minimum", failures)
        call expect(abs(region%center(1) - 0.2_dp) < 1.0e-14_dp, &
            "the center moves to the best point", failures)
        call expect(abs(region%center(2) - 0.8_dp) < 1.0e-14_dp, &
            "the center keeps the best point's second coordinate", failures)

        before = region%center
        values = [50.0_dp, 60.0_dp, 70.0_dp]
        call region%observe_batch(inputs, values, status)
        call expect(maxval(abs(region%center - before)) == 0.0_dp, &
            "a failing batch does not move the center", failures)
        call expect(abs(region%center_value - 2.0_dp) < 1.0e-14_dp, &
            "a failing batch does not change the incumbent", failures)

        ! Ties do not count as improvement.
        values = [2.0_dp, 2.0_dp, 2.0_dp]
        call region%observe_batch(inputs, values, status)
        call expect(abs(region%center_value - 2.0_dp) < 1.0e-14_dp, &
            "matching the incumbent is not an improvement", failures)
    end subroutine check_center_tracking

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_trust_region_t) :: region
        type(fortnum_status_t) :: status
        real(dp) :: lengthscales(2), lengths(2), inputs(2, 2), values(2)

        call region%initialize(0, 4, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a zero-dimensional region is refused", failures)
        call region%initialize(2, 0, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "an empty batch size is refused", failures)

        call region%initialize(2, 4, status)
        call region%restart([1.5_dp, 0.5_dp], 1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a center outside the unit cube is refused", failures)

        call region%restart([0.5_dp, 0.5_dp], 1.0_dp, status)
        lengthscales = [1.0_dp, -1.0_dp]
        call region%side_lengths(lengthscales, lengths, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-positive lengthscale is refused", failures)

        inputs = 0.5_dp
        values = [1.0_dp, 2.0_dp]
        call region%observe_batch(inputs(:, 1:1), values, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a batch of the wrong width is refused", failures)
    end subroutine check_refusals

    !! The rescaling's per-dimension relation is generated by FortSym, so its
    !! derivatives must be checked the same way every other generated leaf is:
    !! against central differences of its own value. `region_side_lengths` does
    !! not consume them today, but they are what lets a policy differentiate
    !! through the region geometry, and an untested generated derivative is
    !! exactly the kind of thing that is wrong the first time it is used.
    subroutine check_generated_leaf_derivatives(failures)
        integer, intent(inout) :: failures
        real(dp) :: log_lengthscale, log_mean, base_length
        real(dp) :: value, d_ell, d_mean, d_base
        real(dp) :: plus, minus, ignored_a, ignored_b, ignored_c
        real(dp), parameter :: h = 1.0e-6_dp
        integer :: case_index
        logical :: matches

        matches = .true.
        do case_index = 1, 3
            select case (case_index)
            case (1)
                log_lengthscale = 0.3_dp; log_mean = -0.2_dp; base_length = 0.8_dp
            case (2)
                log_lengthscale = -1.7_dp; log_mean = 0.9_dp; base_length = 1.6_dp
            case (3)
                log_lengthscale = 2.5_dp; log_mean = 2.5_dp; base_length = 0.05_dp
            end select

            call fortbo_generated_trust_region_leaf(log_lengthscale, log_mean, &
                base_length, value, d_ell, d_mean, d_base)

            call fortbo_generated_trust_region_leaf(log_lengthscale + h, log_mean, &
                base_length, plus, ignored_a, ignored_b, ignored_c)
            call fortbo_generated_trust_region_leaf(log_lengthscale - h, log_mean, &
                base_length, minus, ignored_a, ignored_b, ignored_c)
            if (abs((plus - minus)/(2.0_dp*h) - d_ell) > 1.0e-6_dp*max(1.0_dp, &
                abs(d_ell))) matches = .false.

            call fortbo_generated_trust_region_leaf(log_lengthscale, log_mean + h, &
                base_length, plus, ignored_a, ignored_b, ignored_c)
            call fortbo_generated_trust_region_leaf(log_lengthscale, log_mean - h, &
                base_length, minus, ignored_a, ignored_b, ignored_c)
            if (abs((plus - minus)/(2.0_dp*h) - d_mean) > 1.0e-6_dp*max(1.0_dp, &
                abs(d_mean))) matches = .false.

            call fortbo_generated_trust_region_leaf(log_lengthscale, log_mean, &
                base_length + h, plus, ignored_a, ignored_b, ignored_c)
            call fortbo_generated_trust_region_leaf(log_lengthscale, log_mean, &
                base_length - h, minus, ignored_a, ignored_b, ignored_c)
            if (abs((plus - minus)/(2.0_dp*h) - d_base) > 1.0e-6_dp*max(1.0_dp, &
                abs(d_base))) matches = .false.
        end do
        call expect(matches, &
            "the generated rescaling derivatives match central differences", &
            failures)

        ! At the mean the side length is exactly the base length: this is the
        ! statement that the normalization is the geometric mean and not some
        ! other average that happens to be close.
        call fortbo_generated_trust_region_leaf(1.3_dp, 1.3_dp, 0.7_dp, value, &
            d_ell, d_mean, d_base)
        call expect(abs(value - 0.7_dp) < 1.0e-15_dp, &
            "a lengthscale at the geometric mean gets exactly the base length", &
            failures)
    end subroutine check_generated_leaf_derivatives

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_trust_region
