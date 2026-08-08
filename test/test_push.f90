program test_push
    !! BO6: the 14D robot pushing fixture.
    !!
    !! A simulated objective cannot be checked against a formula, so the
    !! oracles here are conservation and structure arguments that hold for any
    !! correct rigid-body simulation and would fail for the plausible wrong
    !! ones:
    !!
    !!   * an object nobody touches does not move, so a hand parked far away
    !!     with no velocity scores exactly zero -- the value of doing nothing;
    !!   * bounded table friction means the objects stop, so extending the
    !!     simulation cannot keep changing the answer;
    !!   * the reward cannot exceed the initial total distance, since that is
    !!     what landing both objects exactly on their goals is worth;
    !!   * every parameter must matter. A fourteen-dimensional benchmark with
    !!     inert coordinates is a lower-dimensional benchmark wearing a
    !!     disguise, and the torque parameters are the ones most easily left
    !!     inert -- an impulse applied at the centre of mass rather than at the
    !!     contact point would make them do nothing while everything else still
    !!     looked right;
    !!   * the objective is deterministic, which the reference's version is not
    !!     until its observation noise is turned down, and which a benchmark
    !!     has to be if runs are to be replayed.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_push, only: fortbo_push_t, FORTBO_PUSH_DIMENSION
    implicit none

    integer :: failures

    failures = 0
    call check_doing_nothing_scores_nothing(failures)
    call check_a_push_moves_an_object(failures)
    call check_the_reward_is_bounded(failures)
    call check_every_parameter_matters(failures)
    call check_objects_come_to_rest(failures)
    call check_determinism(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_push: PASS"
    else
        print *, "test_push: FAIL", failures
        error stop 1
    end if

contains

    !! A configuration whose hands are parked in the far corners with zero
    !! velocity and zero torque. Nothing is touched, so nothing moves.
    subroutine idle_configuration(x)
        real(dp), intent(out) :: x(FORTBO_PUSH_DIMENSION)

        x = 0.0_dp
        x(1:2) = [-5.0_dp, -5.0_dp]
        x(3:4) = 0.0_dp
        x(5) = 2.0_dp
        x(6) = 0.0_dp
        x(7:8) = [5.0_dp, -5.0_dp]
        x(9:10) = 0.0_dp
        x(11) = 2.0_dp
        x(12) = 0.0_dp
        x(13:14) = 0.0_dp
    end subroutine idle_configuration

    subroutine check_doing_nothing_scores_nothing(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_PUSH_DIMENSION), value, first(2), second(2)

        call idle_configuration(x)
        call problem%value(x, value, status)
        call expect(status%code == FORTNUM_OK, "the idle configuration runs", &
            failures)

        call problem%final_positions(x, first, second, status)
        call expect(maxval(abs(first - problem%start_one)) < 1.0e-12_dp, &
            "an untouched square does not move", failures)
        call expect(maxval(abs(second - problem%start_two)) < 1.0e-12_dp, &
            "an untouched disc does not move", failures)
        ! Doing nothing is worth exactly zero: the objects end as far from
        ! their goals as they began.
        call expect(abs(value) < 1.0e-12_dp, &
            "doing nothing scores exactly zero", failures)
    end subroutine check_doing_nothing_scores_nothing

    !! A hand placed below the disc and driven upward must move it. If this
    !! fails the fixture is inert and every other check passes vacuously.
    subroutine check_a_push_moves_an_object(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_PUSH_DIMENSION), first(2), second(2), value

        call idle_configuration(x)
        ! Hand two starts below the disc at (0, -2) and pushes up toward the
        ! goal at (-4, 3.5).
        x(7:8) = [0.0_dp, -4.0_dp]
        x(9:10) = [-2.0_dp, 6.0_dp]
        x(11) = 20.0_dp
        x(12) = 0.0_dp

        call problem%final_positions(x, first, second, status)
        call expect(status%code == FORTNUM_OK, "the pushing configuration runs", &
            failures)
        call expect(norm2(second - problem%start_two) > 0.5_dp, &
            "a hand driven into the disc moves it", failures)
        call expect(maxval(abs(first - problem%start_one)) < 1.0e-9_dp, &
            "and leaves the other object alone", failures)

        call problem%value(x, value, status)
        call expect(value < 0.0_dp, &
            "pushing the disc toward its goal beats doing nothing", failures)
    end subroutine check_a_push_moves_an_object

    !! No configuration can score better than landing both objects exactly on
    !! their goals.
    subroutine check_the_reward_is_bounded(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_PUSH_DIMENSION), value, bound
        integer :: k
        logical :: within

        bound = problem%best_possible()
        call expect(bound > 0.0_dp, "the bound is positive", failures)

        within = .true.
        do k = 1, 40
            call idle_configuration(x)
            ! A spread of genuinely different pushes.
            x(1:2) = [-3.0_dp + 0.15_dp*real(k, dp), -3.0_dp + 0.1_dp*real(k, dp)]
            x(3:4) = [4.0_dp*sin(0.7_dp*real(k, dp)), 5.0_dp*cos(0.5_dp*real(k, dp))]
            x(5) = 2.0_dp + 0.6_dp*real(k, dp)
            x(6) = 0.15_dp*real(k, dp)
            x(7:8) = [3.0_dp - 0.12_dp*real(k, dp), -4.0_dp + 0.16_dp*real(k, dp)]
            x(9:10) = [-5.0_dp*cos(0.3_dp*real(k, dp)), 6.0_dp*sin(0.4_dp*real(k, dp))]
            x(11) = 2.0_dp + 0.5_dp*real(k, dp)
            x(12) = 0.2_dp*real(k, dp)
            x(13:14) = [2.0_dp*sin(real(k, dp)), -2.0_dp*cos(real(k, dp))]

            call problem%value(x, value, status)
            if (status%code /= FORTNUM_OK) within = .false.
            ! value is the negated reward, so it can never fall below -bound.
            if (value < -bound - 1.0e-9_dp) within = .false.
            ! And nothing is infinite or undefined, which a solver that let a
            ! contact blow up would produce.
            if (.not. (abs(value) < 1.0e6_dp)) within = .false.
        end do
        call expect(within, &
            "no configuration beats landing both objects on their goals", &
            failures)
    end subroutine check_the_reward_is_bounded

    !! Every one of the fourteen coordinates must be able to change the
    !! objective. This pins that the fixture is genuinely fourteen-dimensional
    !! and not a lower-dimensional problem in disguise.
    !!
    !! It does *not* pin how contact impulses are applied, and an earlier
    !! version of this comment claimed it did. Removing the contact-point
    !! torque entirely leaves this check passing, because the torque
    !! parameters also steer the hand's own orientation and so change where the
    !! hand touches. That was found by mutation testing, not by reasoning.
    !!
    !! No check replaces it, and that is stated rather than papered over: the
    !! reference's friction joint caps object torque at 2 against a rotational
    !! inertia near 0.08, so any spin a push imparts is removed within a step
    !! or two and the final rotation for a deliberately off-centre strike
    !! measures 1.6e-15. The application point is therefore not observable in
    !! the final state, and the alternative -- asserting my own solver's
    !! displacement numbers back to itself -- would be a check that cannot
    !! fail.
    subroutine check_every_parameter_matters(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: base(FORTBO_PUSH_DIMENSION), moved(FORTBO_PUSH_DIMENSION)
        real(dp) :: base_value, moved_value
        integer :: k, inert, probe
        logical :: responds
        real(dp), parameter :: DISPLACEMENTS(5) = &
            [0.7_dp, -0.7_dp, 2.0_dp, -2.0_dp, 4.0_dp]

        ! Both hands engaged with both objects.
        call idle_configuration(base)
        ! Hand one to the left of the square at (0, 2), driven right and
        ! slightly up toward that object's goal at (4, 3.5), with its long axis
        ! across the push so the flat face does the work. An earlier version
        ! placed it above and drove it down-right, where it passed the square
        ! without ever touching -- which made its push-duration coordinate
        ! inert for entirely correct reasons and cost a real debugging round.
        base(1:2) = [-2.0_dp, 1.9_dp]
        base(3:4) = [5.0_dp, 1.2_dp]
        base(5) = 12.0_dp
        base(6) = 1.5707963267948966_dp
        base(7:8) = [0.3_dp, -4.0_dp]
        base(9:10) = [-1.5_dp, 5.0_dp]
        base(11) = 14.0_dp
        base(12) = 0.5_dp
        base(13:14) = [1.0_dp, -1.0_dp]

        call problem%value(base, base_value, status)
        call expect(status%code == FORTNUM_OK, "the engaged configuration runs", &
            failures)
        call expect(abs(base_value) > 1.0e-6_dp, &
            "the engaged configuration actually touches something", failures)

        ! Several displacements per coordinate, not one. A coordinate can be
        ! legitimately inert at a particular point -- extending a push after
        ! contact has been lost changes nothing, which is real physics and not
        ! a defect -- so the claim being tested is that no coordinate is inert
        ! *everywhere*. A single probe conflated the two and failed on the
        ! push-duration coordinate for an entirely correct reason.
        inert = 0
        do k = 1, FORTBO_PUSH_DIMENSION
            responds = .false.
            do probe = 1, size(DISPLACEMENTS)
                moved = base
                ! Large enough to matter physically: the objective is
                ! discontinuous, so an infinitesimal probe would legitimately
                ! find nothing and prove nothing.
                moved(k) = base(k) + DISPLACEMENTS(probe)
                call problem%value(moved, moved_value, status)
                if (status%code /= FORTNUM_OK) cycle
                if (abs(moved_value - base_value) > 1.0e-9_dp) responds = .true.
            end do
            if (.not. responds) then
                inert = inert + 1
                print *, "    coordinate inert under every probe:", k
            end if
        end do
        call expect(inert == 0, &
            "no coordinate is inert under every probe", failures)
    end subroutine check_every_parameter_matters

    !! Bounded friction must bring everything to rest, so simulating past the
    !! settling window cannot keep moving the objects. A viscous drag would
    !! creep forever and make the objective depend on how long one waited.
    subroutine check_objects_come_to_rest(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: short_run(FORTBO_PUSH_DIMENSION)
        real(dp) :: long_run(FORTBO_PUSH_DIMENSION)
        real(dp) :: first_short(2), second_short(2)
        real(dp) :: first_long(2), second_long(2)

        call idle_configuration(short_run)
        short_run(7:8) = [0.0_dp, -4.0_dp]
        short_run(9:10) = [-2.0_dp, 6.0_dp]
        short_run(11) = 10.0_dp
        call problem%final_positions(short_run, first_short, second_short, status)
        call expect(status%code == FORTNUM_OK, "the short run completes", failures)

        ! Same push, but the *other* hand is asked to run far longer while
        ! parked out of reach. That extends the simulation without changing the
        ! physics, so the outcome must not move.
        long_run = short_run
        long_run(5) = 30.0_dp
        call problem%final_positions(long_run, first_long, second_long, status)
        call expect(status%code == FORTNUM_OK, "the long run completes", failures)

        call expect(maxval(abs(second_long - second_short)) < 1.0e-6_dp, &
            "the pushed object has come to rest before the run ends", failures)
        call expect(maxval(abs(first_long - first_short)) < 1.0e-6_dp, &
            "and so has the untouched one", failures)
    end subroutine check_objects_come_to_rest

    subroutine check_determinism(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: x(FORTBO_PUSH_DIMENSION), first, second

        call idle_configuration(x)
        x(1:2) = [0.5_dp, 3.6_dp]
        x(3:4) = [1.5_dp, -4.0_dp]
        x(5) = 12.0_dp
        x(13) = 1.3_dp

        call problem%value(x, first, status)
        call problem%value(x, second, status)
        ! Bit-identical, not close. The reference is stochastic until its
        ! observation noise is turned down; a benchmark that cannot be replayed
        ! exactly cannot support the reproducibility claims elsewhere in this
        ! package.
        call expect(first == second, &
            "the objective is bit-identical on repeat evaluation", failures)
    end subroutine check_determinism

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(fortbo_push_t) :: problem
        type(fortnum_status_t) :: status
        real(dp) :: lower(FORTBO_PUSH_DIMENSION), upper(FORTBO_PUSH_DIMENSION)
        real(dp) :: short_bounds(3), x(FORTBO_PUSH_DIMENSION)
        real(dp) :: gradient(FORTBO_PUSH_DIMENSION), value
        real(dp), parameter :: two_pi = 6.283185307179586_dp

        call problem%bounds(lower, upper, status)
        call expect(status%code == FORTNUM_OK, "the box is available", failures)
        call expect(problem%dimension() == 14, "the fixture is fourteen wide", &
            failures)
        ! The reference's box, spot-checked where a transcription slip would
        ! land: the duration floor is 2 and not 0, and the angle runs to 2 pi.
        call expect(abs(lower(5) - 2.0_dp) < 1.0e-12_dp .and. &
            abs(upper(5) - 30.0_dp) < 1.0e-12_dp, &
            "the duration bounds match the reference", failures)
        call expect(abs(upper(6) - two_pi) < 1.0e-12_dp, &
            "the angle runs to two pi", failures)
        call expect(abs(lower(13) + 5.0_dp) < 1.0e-12_dp .and. &
            abs(upper(14) - 5.0_dp) < 1.0e-12_dp, &
            "the torque bounds match the reference", failures)

        call problem%bounds(short_bounds, short_bounds, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "bounds of the wrong width are refused", failures)

        call problem%value(short_bounds, value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a configuration of the wrong width is refused", failures)

        call idle_configuration(x)
        call problem%gradient(x, gradient, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "the gradient is refused by name, not differenced", failures)
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

end program test_push
