module fortbo_push
    !! The 14-dimensional robot pushing fixture (ROADMAP BO6).
    !!
    !! Two robot hands push two objects across a frictional table toward goal
    !! locations. The reward is how much closer the objects end up than they
    !! started. Unlike every other benchmark in this package there is no closed
    !! form: the objective is the output of a rigid-body simulation, and that
    !! is the point of including it.
    !!
    !! **Provenance.** The TuRBO paper uses this problem but does not define
    !! it; its README points at `zi-w/Ensemble-Bayesian-Optimization` and lists
    !! the three changes it made (visualization off, the 0.01 observation noise
    !! reduced to 1e-6, and the sign flipped for minimization). That repository
    !! was fetched into `fortbo-bench` provenance and read. Everything numeric
    !! here comes from it and from nowhere else:
    !!
    !!   * the box, `[-5,5] [-5,5] [-10,10] [-10,10] [2,30] [0,2pi]` twice over
    !!     followed by `[-5,5]` twice for the two torques;
    !!   * objects starting at `(0, 2)` and `(0, -2)`, with goals at `(4, 3.5)`
    !!     and `(-4, 3.5)`;
    !!   * a 1x1 square and a radius-1 disc, densities 0.05, hand density 0.1;
    !!   * table friction as a joint capped at force 5 and torque 2 for the
    !!     objects and force 2 for the hands;
    !!   * the hand controller, which is proportional rather than a prescribed
    !!     trajectory: `F = m (v_desired - v) * 30` and
    !!     `tau = m (w_desired - w) * 30`;
    !!   * a 1/100 timestep, the pushing duration `int(10 x)` steps, and the
    !!     simulation continuing a further 100 steps after the last hand stops
    !!     so the objects coast to rest;
    !!   * the reward, `|g1 - s1| + |g2 - s2| - |g1 - e1| - |g2 - e2|`.
    !!
    !! None of those is derivable. Guessing them would have produced a
    !! plausible 14-dimensional function that is not this benchmark.
    !!
    !! **What is ours, stated plainly.** The reference runs Box2D, a specific
    !! sequential-impulse solver with its own iteration counts, slop and bias
    !! constants. This module implements its own simulation, so it does *not*
    !! reproduce Box2D's trajectories and cannot be compared against published
    !! numbers for this problem. Two approximations go beyond mere solver
    !! differences and are named rather than buried:
    !!
    !!   * contact shapes are a capsule for each hand and a disc for each
    !!     object. The capsule is a close stand-in for the 2 x 0.6 hand, which
    !!     matters because a long thin pusher is what makes contact geometry
    !!     interesting; the square object becomes a disc of its inscribed
    !!     radius, which loses its corners.
    !!   * masses and rotational inertias are computed from the *true* shapes
    !!     and densities, not from the collision proxies, so the dynamics carry
    !!     the right scales even where the geometry is simplified.
    !!
    !! What is faithfully reproduced is the structure that makes the problem
    !! hard, and that is what a fixture is for: fourteen interacting
    !! parameters, an objective that is **discontinuous** where a hand starts
    !! or stops touching an object, flat regions where a hand misses entirely
    !! and the reward does not vary at all, and two coupled sub-problems that
    !! cannot be solved independently because both hands share the table.
    !!
    !! The fixture refuses a gradient, for the same reason the rover does and
    !! more sharply: contact makes the objective genuinely discontinuous, not
    !! merely non-differentiable. A finite difference across a contact change
    !! reports a slope that describes nothing.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortbo_push_t

    integer, parameter, public :: FORTBO_PUSH_DIMENSION = 14

    !! Simulation constants, all from the reference.
    real(dp), parameter :: TIME_STEP = 0.01_dp
    integer, parameter :: SETTLE_STEPS = 100
    real(dp), parameter :: HAND_FORCE_UNIT = 30.0_dp
    real(dp), parameter :: HAND_TORQUE_UNIT = 30.0_dp
    real(dp), parameter :: OBJECT_MAX_FRICTION_FORCE = 5.0_dp
    real(dp), parameter :: OBJECT_MAX_FRICTION_TORQUE = 2.0_dp
    real(dp), parameter :: HAND_MAX_FRICTION_FORCE = 2.0_dp
    real(dp), parameter :: HAND_MAX_FRICTION_TORQUE = 2.0_dp
    real(dp), parameter :: OBJECT_DENSITY = 0.05_dp
    real(dp), parameter :: HAND_DENSITY = 0.1_dp

    !! Shapes. The half-extents are the reference's `box=` arguments.
    real(dp), parameter :: SQUARE_HALF = 0.5_dp
    real(dp), parameter :: DISC_RADIUS = 1.0_dp
    real(dp), parameter :: HAND_HALF_LENGTH = 1.0_dp
    real(dp), parameter :: HAND_HALF_WIDTH = 0.3_dp

    !! Restitution is zero: pushing, not bouncing.
    real(dp), parameter :: RESTITUTION = 0.0_dp
    !! Contact stiffness for positional correction, expressed as the fraction
    !! of overlap removed per step. Box2D's own bias factor is in this range.
    real(dp), parameter :: CONTACT_BIAS = 0.2_dp

    type :: fortbo_push_t
        real(dp) :: start_one(2) = [0.0_dp, 2.0_dp]
        real(dp) :: start_two(2) = [0.0_dp, -2.0_dp]
        real(dp) :: goal_one(2) = [4.0_dp, 3.5_dp]
        real(dp) :: goal_two(2) = [-4.0_dp, 3.5_dp]
    contains
        procedure, public :: dimension => push_dimension
        procedure, public :: bounds => push_bounds
        procedure, public :: value => push_value
        procedure, public :: gradient => push_gradient
        procedure, public :: final_positions => push_final_positions
        procedure, public :: final_state => push_final_state
        procedure, public :: best_possible => push_best_possible
    end type fortbo_push_t

    !! A planar rigid body.
    type :: body_t
        real(dp) :: position(2) = 0.0_dp
        real(dp) :: velocity(2) = 0.0_dp
        real(dp) :: angle = 0.0_dp
        real(dp) :: angular_velocity = 0.0_dp
        real(dp) :: mass = 1.0_dp
        real(dp) :: inertia = 1.0_dp
        !! Collision proxy radius; for a hand this is the capsule radius.
        real(dp) :: radius = 1.0_dp
        !! Half-length of the capsule segment. Zero for a disc.
        real(dp) :: half_length = 0.0_dp
        real(dp) :: max_friction_force = 0.0_dp
        real(dp) :: max_friction_torque = 0.0_dp
    end type body_t

contains

    pure integer function push_dimension(self) result(d)
        class(fortbo_push_t), intent(in) :: self

        d = FORTBO_PUSH_DIMENSION
    end function push_dimension

    !! The reference's box, entry for entry.
    subroutine push_bounds(self, lower, upper, status)
        class(fortbo_push_t), intent(in) :: self
        real(dp), intent(out) :: lower(:)
        real(dp), intent(out) :: upper(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), parameter :: two_pi = 6.283185307179586_dp

        lower = 0.0_dp
        upper = 0.0_dp
        if (size(lower) /= FORTBO_PUSH_DIMENSION .or. &
            size(upper) /= FORTBO_PUSH_DIMENSION) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo push: bounds must have fourteen entries")
            return
        end if

        !                 rx     ry    xvel    yvel   steps  angle
        lower(1:6) = [-5.0_dp, -5.0_dp, -10.0_dp, -10.0_dp, 2.0_dp, 0.0_dp]
        upper(1:6) = [5.0_dp, 5.0_dp, 10.0_dp, 10.0_dp, 30.0_dp, two_pi]
        lower(7:12) = lower(1:6)
        upper(7:12) = upper(1:6)
        ! The two torques.
        lower(13:14) = [-5.0_dp, -5.0_dp]
        upper(13:14) = [5.0_dp, 5.0_dp]
        call status_set(status, FORTNUM_OK, "")
    end subroutine push_bounds

    !! The largest reward attainable, which is the initial total distance:
    !! both objects landing exactly on their goals. Reported so a regret can be
    !! formed without pretending the optimum is known -- it is an upper bound,
    !! not a value anyone has reached.
    pure real(dp) function push_best_possible(self) result(value)
        class(fortbo_push_t), intent(in) :: self

        value = norm2(self%goal_one - self%start_one) &
            + norm2(self%goal_two - self%start_two)
    end function push_best_possible

    !! Mass and rotational inertia of a solid disc.
    pure subroutine disc_inertia(radius, density, mass, inertia)
        real(dp), intent(in) :: radius, density
        real(dp), intent(out) :: mass, inertia
        real(dp), parameter :: pi = 3.141592653589793_dp

        mass = density*pi*radius**2
        inertia = 0.5_dp*mass*radius**2
    end subroutine disc_inertia

    !! Mass and rotational inertia of a solid rectangle about its centre.
    pure subroutine box_inertia(half_x, half_y, density, mass, inertia)
        real(dp), intent(in) :: half_x, half_y, density
        real(dp), intent(out) :: mass, inertia

        mass = density*4.0_dp*half_x*half_y
        inertia = mass*(half_x**2 + half_y**2)/3.0_dp
    end subroutine box_inertia

    !! Closest point to `point` on the capsule's central segment.
    pure subroutine capsule_closest_point(body, point, closest)
        type(body_t), intent(in) :: body
        real(dp), intent(in) :: point(2)
        real(dp), intent(out) :: closest(2)
        real(dp) :: axis(2), offset(2), projection

        axis = [cos(body%angle), sin(body%angle)]
        offset = point - body%position
        projection = dot_product(offset, axis)
        projection = max(-body%half_length, min(body%half_length, projection))
        closest = body%position + projection*axis
    end subroutine capsule_closest_point

    !! Resolve one contact between a hand and an object.
    !!
    !! A sequential impulse, applied at the contact point so that an off-centre
    !! push spins the object. That coupling is the whole reason the torque
    !! parameters do anything, and a solver that applied impulses at the centre
    !! of mass would make dimensions 13 and 14 inert while still producing a
    !! fourteen-dimensional function.
    subroutine resolve_contact(hand, object)
        type(body_t), intent(inout) :: hand, object
        real(dp) :: closest(2), delta(2), distance, normal(2), overlap
        real(dp) :: contact(2), r_hand(2), r_object(2)
        real(dp) :: relative(2), separating, impulse_scale, impulse(2)
        real(dp) :: cross_hand, cross_object, denominator, correction

        call capsule_closest_point(hand, object%position, closest)
        delta = object%position - closest
        distance = norm2(delta)
        overlap = hand%radius + object%radius - distance
        if (overlap <= 0.0_dp) return

        if (distance > 1.0e-12_dp) then
            normal = delta/distance
        else
            ! Exactly coincident centres: pick a normal rather than divide by
            ! zero. Arbitrary, but this configuration is unreachable from a
            ! non-overlapping start at this timestep.
            normal = [1.0_dp, 0.0_dp]
        end if
        contact = closest + normal*hand%radius

        r_hand = contact - hand%position
        r_object = contact - object%position

        ! Relative velocity at the contact, including both spins.
        relative = (object%velocity &
            + object%angular_velocity*[-r_object(2), r_object(1)]) &
            - (hand%velocity + hand%angular_velocity*[-r_hand(2), r_hand(1)])
        separating = dot_product(relative, normal)
        if (separating > 0.0_dp) then
            call separate(hand, object, normal, overlap)
            return
        end if

        cross_hand = r_hand(1)*normal(2) - r_hand(2)*normal(1)
        cross_object = r_object(1)*normal(2) - r_object(2)*normal(1)
        denominator = 1.0_dp/hand%mass + 1.0_dp/object%mass &
            + cross_hand**2/hand%inertia + cross_object**2/object%inertia
        if (denominator <= 0.0_dp) return

        impulse_scale = -(1.0_dp + RESTITUTION)*separating/denominator
        impulse = impulse_scale*normal

        object%velocity = object%velocity + impulse/object%mass
        object%angular_velocity = object%angular_velocity &
            + cross_object*impulse_scale/object%inertia
        hand%velocity = hand%velocity - impulse/hand%mass
        hand%angular_velocity = hand%angular_velocity &
            - cross_hand*impulse_scale/hand%inertia

        call separate(hand, object, normal, overlap)
    end subroutine resolve_contact

    !! Push overlapping bodies apart, in proportion to their compliance.
    subroutine separate(hand, object, normal, overlap)
        type(body_t), intent(inout) :: hand, object
        real(dp), intent(in) :: normal(2), overlap
        real(dp) :: total, correction

        total = 1.0_dp/hand%mass + 1.0_dp/object%mass
        if (total <= 0.0_dp) return
        correction = CONTACT_BIAS*overlap/total
        object%position = object%position + normal*correction/object%mass
        hand%position = hand%position - normal*correction/hand%mass
    end subroutine separate

    !! Table friction, as the reference models it: a joint that opposes motion
    !! with a bounded force rather than a viscous drag.
    !!
    !! The bound is what makes the objects *stop* rather than drift forever
    !! after the hand lets go, and it is why the fixture has flat regions -- a
    !! push too weak to overcome it moves nothing at all.
    subroutine apply_table_friction(body)
        type(body_t), intent(inout) :: body
        real(dp) :: speed, impulse, direction(2), spin_impulse

        speed = norm2(body%velocity)
        if (speed > 1.0e-12_dp) then
            direction = body%velocity/speed
            impulse = body%max_friction_force*TIME_STEP/body%mass
            if (impulse >= speed) then
                body%velocity = 0.0_dp
            else
                body%velocity = body%velocity - direction*impulse
            end if
        end if

        if (abs(body%angular_velocity) > 1.0e-12_dp) then
            spin_impulse = body%max_friction_torque*TIME_STEP/body%inertia
            if (spin_impulse >= abs(body%angular_velocity)) then
                body%angular_velocity = 0.0_dp
            else
                body%angular_velocity = body%angular_velocity &
                    - sign(spin_impulse, body%angular_velocity)
            end if
        end if
    end subroutine apply_table_friction

    !! Where the objects end up. Exposed because a caller checking the fixture
    !! needs to see the simulation's outcome, not only its scalar reward.
    subroutine push_final_positions(self, x, first, second, status)
        class(fortbo_push_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: first(2), second(2)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: spin_one, spin_two

        call push_final_state(self, x, first, second, spin_one, spin_two, status)
    end subroutine push_final_positions

    !! Positions *and* accumulated rotations.
    !!
    !! A note on what the rotations do and do not show. Contact impulses here
    !! are applied at the contact point, which is the physically correct choice
    !! and changes both the impulse magnitude and the bodies' spin. It is
    !! tempting to test that by checking an off-centre push rotates the object
    !! -- but it does not, measurably, and the reason is the reference's own
    !! numbers: the objects' friction joint caps torque at 2 against a disc
    !! whose rotational inertia is about 0.08, so a step removes up to 0.25
    !! rad/s and any spin a push imparts is gone almost at once. Measured
    !! final rotation for a deliberately off-centre strike is 1.6e-15.
    !!
    !! So the application point is a modelling choice that is documented rather
    !! than pinned by a test, and this routine's rotations are exposed for
    !! inspection rather than as an oracle. Claiming a check for it would be
    !! worse than admitting there is none: an earlier version of the test suite
    !! asserted that the torque coordinates would go inert under a
    !! centre-of-mass solver, and mutation testing showed they do not -- they
    !! still steer the hand's own orientation, so the check passed against the
    !! very defect it named.
    subroutine push_final_state(self, x, first, second, spin_one, spin_two, status)
        class(fortbo_push_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: first(2), second(2)
        real(dp), intent(out) :: spin_one, spin_two
        type(fortnum_status_t), intent(out) :: status
        type(body_t) :: hand_one, hand_two, object_one, object_two
        integer :: steps_one, steps_two, total, t

        first = 0.0_dp
        second = 0.0_dp
        spin_one = 0.0_dp
        spin_two = 0.0_dp
        if (size(x) /= FORTBO_PUSH_DIMENSION) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo push: a configuration has fourteen entries")
            return
        end if

        ! Objects: a 1x1 square and a radius-1 disc, at the reference's starts.
        call box_inertia(SQUARE_HALF, SQUARE_HALF, OBJECT_DENSITY, &
            object_one%mass, object_one%inertia)
        object_one%position = self%start_one
        ! Inscribed radius: the square's corners are lost, which is stated in
        ! the module header rather than hidden here.
        object_one%radius = SQUARE_HALF
        object_one%max_friction_force = OBJECT_MAX_FRICTION_FORCE
        object_one%max_friction_torque = OBJECT_MAX_FRICTION_TORQUE

        call disc_inertia(DISC_RADIUS, OBJECT_DENSITY, object_two%mass, &
            object_two%inertia)
        object_two%position = self%start_two
        object_two%radius = DISC_RADIUS
        object_two%max_friction_force = OBJECT_MAX_FRICTION_FORCE
        object_two%max_friction_torque = OBJECT_MAX_FRICTION_TORQUE

        ! Hands: capsules standing in for the 2 x 0.6 rectangles.
        call box_inertia(HAND_HALF_LENGTH, HAND_HALF_WIDTH, HAND_DENSITY, &
            hand_one%mass, hand_one%inertia)
        hand_one%position = x(1:2)
        hand_one%angle = x(6)
        hand_one%radius = HAND_HALF_WIDTH
        hand_one%half_length = HAND_HALF_LENGTH - HAND_HALF_WIDTH
        hand_one%max_friction_force = HAND_MAX_FRICTION_FORCE
        hand_one%max_friction_torque = HAND_MAX_FRICTION_TORQUE

        hand_two = hand_one
        hand_two%position = x(7:8)
        hand_two%angle = x(12)
        hand_two%velocity = 0.0_dp
        hand_two%angular_velocity = 0.0_dp

        ! The reference takes `int(10 x)` steps, so the duration parameter is
        ! quantized to tenths of a second. That quantization is a real feature
        ! of the objective -- it is one of the sources of flat regions -- and
        ! smoothing it would make the fixture easier than the benchmark.
        steps_one = int(x(5)*10.0_dp)
        steps_two = int(x(11)*10.0_dp)
        total = max(steps_one, steps_two) + SETTLE_STEPS

        do t = 0, total - 1
            if (t < steps_one) call drive(hand_one, x(3:4), x(13))
            if (t < steps_two) call drive(hand_two, x(9:10), x(14))

            call resolve_contact(hand_one, object_one)
            call resolve_contact(hand_one, object_two)
            call resolve_contact(hand_two, object_one)
            call resolve_contact(hand_two, object_two)

            call apply_table_friction(object_one)
            call apply_table_friction(object_two)
            call apply_table_friction(hand_one)
            call apply_table_friction(hand_two)

            call integrate(hand_one)
            call integrate(hand_two)
            call integrate(object_one)
            call integrate(object_two)
        end do

        first = object_one%position
        second = object_two%position
        spin_one = object_one%angle
        spin_two = object_two%angle
        call status_set(status, FORTNUM_OK, "")
    end subroutine push_final_state

    !! The reference's proportional controller, not a prescribed trajectory.
    !!
    !! The hand is *pushed toward* the requested velocity rather than set to
    !! it, so contact with a heavy object genuinely slows the hand down. A
    !! kinematic hand that ignored reaction forces would make the problem far
    !! easier and would not be this benchmark.
    subroutine drive(hand, desired_velocity, desired_spin)
        type(body_t), intent(inout) :: hand
        real(dp), intent(in) :: desired_velocity(2), desired_spin
        real(dp) :: force(2), torque

        force = hand%mass*(desired_velocity - hand%velocity)*HAND_FORCE_UNIT
        torque = hand%mass*(desired_spin - hand%angular_velocity)*HAND_TORQUE_UNIT
        hand%velocity = hand%velocity + force*TIME_STEP/hand%mass
        hand%angular_velocity = hand%angular_velocity &
            + torque*TIME_STEP/hand%inertia
    end subroutine drive

    subroutine integrate(body)
        type(body_t), intent(inout) :: body

        body%position = body%position + body%velocity*TIME_STEP
        body%angle = body%angle + body%angular_velocity*TIME_STEP
    end subroutine integrate

    !! The objective, negated for minimization.
    !!
    !! The reference returns `initial_distance - final_distance`, a reward to
    !! be maximized; TuRBO flips the sign, and FortBO minimizes, so this
    !! returns the negated reward. Zero means the objects ended exactly as far
    !! from their goals as they began, which is what doing nothing achieves.
    subroutine push_value(self, x, value, status)
        class(fortbo_push_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: first(2), second(2), reward

        value = 0.0_dp
        call self%final_positions(x, first, second, status)
        if (status%code /= FORTNUM_OK) return

        reward = self%best_possible() &
            - norm2(self%goal_one - first) - norm2(self%goal_two - second)
        value = -reward
        call status_set(status, FORTNUM_OK, "")
    end subroutine push_value

    !! Refused, by construction.
    !!
    !! Sharper than the rover's refusal: contact makes this objective
    !! genuinely *discontinuous*, not merely non-differentiable. A hand that
    !! just misses and a hand that just touches produce different outcomes with
    !! no limit between them, and the step-quantized duration adds jumps of its
    !! own. A finite difference across either reports a slope describing
    !! nothing.
    subroutine push_gradient(self, x, gradient, status)
        class(fortbo_push_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "fortbo push: contact makes this objective discontinuous, so no "// &
            "gradient exists")
    end subroutine push_gradient

end module fortbo_push
