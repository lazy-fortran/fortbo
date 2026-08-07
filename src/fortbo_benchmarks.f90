module fortbo_benchmarks
    !! Synthetic objectives with known optima (ROADMAP BO6).
    !!
    !! These are the fixtures regret is measured against, so two things matter
    !! more than convenience:
    !!
    !!   * the optimum must be *known*, not discovered by the code under test.
    !!     Each function carries its literature optimizer and optimal value as
    !!     data, and the test checks that no point on a dense grid beats the
    !!     recorded value. A regret curve computed against an optimum the
    !!     optimizer found itself measures nothing;
    !!   * every function supplies an exact gradient. Without it the
    !!     derivative-observation path could only ever be exercised against
    !!     finite differences, and a benchmark that feeds approximate gradients
    !!     into a derivative-informed model measures the differencing error, not
    !!     the method.
    !!
    !! All functions are minimized and evaluated in their natural domain; the
    !! `bounds` accessor gives the box each is conventionally studied on, which
    !! `fortbo_space` maps to the unit cube.
    !!
    !! Ackley, Rosenbrock, Levy, and the sphere accept any dimension, which is
    !! what makes them usable for the high-dimensional trust-region evidence.
    !! Branin and the two Hartmann functions are fixed-dimension by definition.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: FORTBO_BENCH_BRANIN = 1
    integer, parameter, public :: FORTBO_BENCH_HARTMANN3 = 2
    integer, parameter, public :: FORTBO_BENCH_HARTMANN6 = 3
    integer, parameter, public :: FORTBO_BENCH_ACKLEY = 4
    integer, parameter, public :: FORTBO_BENCH_ROSENBROCK = 5
    integer, parameter, public :: FORTBO_BENCH_LEVY = 6
    integer, parameter, public :: FORTBO_BENCH_SPHERE = 7

    public :: fortbo_benchmark_t
    public :: fortbo_benchmark_name
    public :: fortbo_benchmark_fixed_dimension

    type :: fortbo_benchmark_t
        integer :: kind = FORTBO_BENCH_SPHERE
        integer :: dimension = 2
        !! Additive observation noise standard deviation. Zero is noiseless.
        real(dp) :: noise = 0.0_dp
    contains
        procedure, public :: value => benchmark_value
        procedure, public :: gradient => benchmark_gradient
        procedure, public :: bounds => benchmark_bounds
        procedure, public :: optimum => benchmark_optimum
        procedure, public :: optimal_value => benchmark_optimal_value
        procedure, public :: is_valid => benchmark_is_valid
    end type fortbo_benchmark_t

    ! Branin constants.
    real(dp), parameter :: BRANIN_A = 1.0_dp
    real(dp), parameter :: PI_VALUE = 3.14159265358979323846_dp
    real(dp), parameter :: BRANIN_B = 5.1_dp/(4.0_dp*PI_VALUE**2)
    real(dp), parameter :: BRANIN_C = 5.0_dp/PI_VALUE
    real(dp), parameter :: BRANIN_R = 6.0_dp
    real(dp), parameter :: BRANIN_S = 10.0_dp
    real(dp), parameter :: BRANIN_T = 1.0_dp/(8.0_dp*PI_VALUE)

    ! Ackley constants.
    real(dp), parameter :: ACKLEY_A = 20.0_dp
    real(dp), parameter :: ACKLEY_B = 0.2_dp
    real(dp), parameter :: ACKLEY_C = 2.0_dp*PI_VALUE

    ! Hartmann-3.
    real(dp), parameter :: H3_ALPHA(4) = [1.0_dp, 1.2_dp, 3.0_dp, 3.2_dp]
    real(dp), parameter :: H3_A(4, 3) = reshape([ &
        3.0_dp, 0.1_dp, 3.0_dp, 0.1_dp, &
        10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, &
        30.0_dp, 35.0_dp, 30.0_dp, 35.0_dp], [4, 3])
    real(dp), parameter :: H3_P(4, 3) = reshape([ &
        0.3689_dp, 0.4699_dp, 0.1091_dp, 0.03815_dp, &
        0.1170_dp, 0.4387_dp, 0.8732_dp, 0.5743_dp, &
        0.2673_dp, 0.7470_dp, 0.5547_dp, 0.8828_dp], [4, 3])

    ! Hartmann-6.
    real(dp), parameter :: H6_ALPHA(4) = [1.0_dp, 1.2_dp, 3.0_dp, 3.2_dp]
    real(dp), parameter :: H6_A(4, 6) = reshape([ &
        10.0_dp, 0.05_dp, 3.0_dp, 17.0_dp, &
        3.0_dp, 10.0_dp, 3.5_dp, 8.0_dp, &
        17.0_dp, 17.0_dp, 1.7_dp, 0.05_dp, &
        3.5_dp, 0.1_dp, 10.0_dp, 10.0_dp, &
        1.7_dp, 8.0_dp, 17.0_dp, 0.1_dp, &
        8.0_dp, 14.0_dp, 8.0_dp, 14.0_dp], [4, 6])
    real(dp), parameter :: H6_P(4, 6) = reshape([ &
        0.1312_dp, 0.2329_dp, 0.2348_dp, 0.4047_dp, &
        0.1696_dp, 0.4135_dp, 0.1451_dp, 0.8828_dp, &
        0.5569_dp, 0.8307_dp, 0.3522_dp, 0.8732_dp, &
        0.0124_dp, 0.3736_dp, 0.2883_dp, 0.5743_dp, &
        0.8283_dp, 0.1004_dp, 0.3047_dp, 0.1091_dp, &
        0.5886_dp, 0.9991_dp, 0.6650_dp, 0.0381_dp], [4, 6])

contains

    pure function fortbo_benchmark_name(kind) result(name)
        integer, intent(in) :: kind
        character(len=:), allocatable :: name

        select case (kind)
        case (FORTBO_BENCH_BRANIN)
            name = "branin"
        case (FORTBO_BENCH_HARTMANN3)
            name = "hartmann3"
        case (FORTBO_BENCH_HARTMANN6)
            name = "hartmann6"
        case (FORTBO_BENCH_ACKLEY)
            name = "ackley"
        case (FORTBO_BENCH_ROSENBROCK)
            name = "rosenbrock"
        case (FORTBO_BENCH_LEVY)
            name = "levy"
        case (FORTBO_BENCH_SPHERE)
            name = "sphere"
        case default
            name = "unknown"
        end select
    end function fortbo_benchmark_name

    !! Zero for the functions that accept any dimension.
    pure integer function fortbo_benchmark_fixed_dimension(kind) result(d)
        integer, intent(in) :: kind

        select case (kind)
        case (FORTBO_BENCH_BRANIN)
            d = 2
        case (FORTBO_BENCH_HARTMANN3)
            d = 3
        case (FORTBO_BENCH_HARTMANN6)
            d = 6
        case default
            d = 0
        end select
    end function fortbo_benchmark_fixed_dimension

    pure logical function benchmark_is_valid(self) result(ok)
        class(fortbo_benchmark_t), intent(in) :: self
        integer :: fixed

        ok = .false.
        if (self%dimension < 1) return
        if (self%noise < 0.0_dp) return
        fixed = fortbo_benchmark_fixed_dimension(self%kind)
        if (fixed > 0 .and. self%dimension /= fixed) return
        if (self%kind == FORTBO_BENCH_ROSENBROCK .and. self%dimension < 2) return
        ok = .true.
    end function benchmark_is_valid

    subroutine benchmark_value(self, x, value, status)
        class(fortbo_benchmark_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: sum_squares, sum_cos, inner, term
        real(dp) :: w(size(x)), w_last
        integer :: i, j

        value = 0.0_dp
        if (.not. self%is_valid() .or. size(x) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: invalid configuration or query width")
            return
        end if

        select case (self%kind)
        case (FORTBO_BENCH_BRANIN)
            value = BRANIN_A*(x(2) - BRANIN_B*x(1)**2 + BRANIN_C*x(1) - BRANIN_R)**2 &
                + BRANIN_S*(1.0_dp - BRANIN_T)*cos(x(1)) + BRANIN_S
        case (FORTBO_BENCH_HARTMANN3)
            value = 0.0_dp
            do i = 1, 4
                inner = 0.0_dp
                do j = 1, 3
                    inner = inner + H3_A(i, j)*(x(j) - H3_P(i, j))**2
                end do
                value = value - H3_ALPHA(i)*exp(-inner)
            end do
        case (FORTBO_BENCH_HARTMANN6)
            value = 0.0_dp
            do i = 1, 4
                inner = 0.0_dp
                do j = 1, 6
                    inner = inner + H6_A(i, j)*(x(j) - H6_P(i, j))**2
                end do
                value = value - H6_ALPHA(i)*exp(-inner)
            end do
        case (FORTBO_BENCH_ACKLEY)
            sum_squares = sum(x**2)/real(self%dimension, dp)
            sum_cos = sum(cos(ACKLEY_C*x))/real(self%dimension, dp)
            value = -ACKLEY_A*exp(-ACKLEY_B*sqrt(sum_squares)) - exp(sum_cos) &
                + ACKLEY_A + exp(1.0_dp)
        case (FORTBO_BENCH_ROSENBROCK)
            value = 0.0_dp
            do i = 1, self%dimension - 1
                value = value + 100.0_dp*(x(i + 1) - x(i)**2)**2 + (x(i) - 1.0_dp)**2
            end do
        case (FORTBO_BENCH_LEVY)
            w = 1.0_dp + (x - 1.0_dp)/4.0_dp
            w_last = w(self%dimension)
            value = sin(PI_VALUE*w(1))**2 &
                + (w_last - 1.0_dp)**2*(1.0_dp + sin(2.0_dp*PI_VALUE*w_last)**2)
            do i = 1, self%dimension - 1
                term = (w(i) - 1.0_dp)**2 &
                    *(1.0_dp + 10.0_dp*sin(PI_VALUE*w(i) + 1.0_dp)**2)
                value = value + term
            end do
        case (FORTBO_BENCH_SPHERE)
            value = sum(x**2)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: unknown function")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine benchmark_value

    !! Exact gradient. This is what feeds the derivative-observation path, so it
    !! is analytic throughout — never a difference of `value`.
    subroutine benchmark_gradient(self, x, gradient, status)
        class(fortbo_benchmark_t), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: inner, factor, u, root, mean_squares, mean_cos
        real(dp) :: w(size(x)), w_last, dw
        integer :: i, j, d

        d = self%dimension
        gradient = 0.0_dp
        if (.not. self%is_valid() .or. size(x) /= d .or. size(gradient) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: invalid configuration or query width")
            return
        end if

        select case (self%kind)
        case (FORTBO_BENCH_BRANIN)
            u = x(2) - BRANIN_B*x(1)**2 + BRANIN_C*x(1) - BRANIN_R
            gradient(1) = 2.0_dp*BRANIN_A*u*(-2.0_dp*BRANIN_B*x(1) + BRANIN_C) &
                - BRANIN_S*(1.0_dp - BRANIN_T)*sin(x(1))
            gradient(2) = 2.0_dp*BRANIN_A*u
        case (FORTBO_BENCH_HARTMANN3)
            do i = 1, 4
                inner = 0.0_dp
                do j = 1, 3
                    inner = inner + H3_A(i, j)*(x(j) - H3_P(i, j))**2
                end do
                factor = H3_ALPHA(i)*exp(-inner)
                do j = 1, 3
                    gradient(j) = gradient(j) &
                        + factor*2.0_dp*H3_A(i, j)*(x(j) - H3_P(i, j))
                end do
            end do
        case (FORTBO_BENCH_HARTMANN6)
            do i = 1, 4
                inner = 0.0_dp
                do j = 1, 6
                    inner = inner + H6_A(i, j)*(x(j) - H6_P(i, j))**2
                end do
                factor = H6_ALPHA(i)*exp(-inner)
                do j = 1, 6
                    gradient(j) = gradient(j) &
                        + factor*2.0_dp*H6_A(i, j)*(x(j) - H6_P(i, j))
                end do
            end do
        case (FORTBO_BENCH_ACKLEY)
            mean_squares = sum(x**2)/real(d, dp)
            mean_cos = sum(cos(ACKLEY_C*x))/real(d, dp)
            root = sqrt(mean_squares)
            ! At the origin the first term's gradient is the limit zero; the
            ! expression below would divide by zero there, and the Ackley
            ! optimum is exactly that point, so it must be handled and not
            ! stumbled into.
            if (root > 0.0_dp) then
                factor = ACKLEY_A*ACKLEY_B*exp(-ACKLEY_B*root)/(root*real(d, dp))
            else
                factor = 0.0_dp
            end if
            do j = 1, d
                gradient(j) = factor*x(j) &
                    + exp(mean_cos)*ACKLEY_C*sin(ACKLEY_C*x(j))/real(d, dp)
            end do
        case (FORTBO_BENCH_ROSENBROCK)
            do i = 1, d - 1
                gradient(i) = gradient(i) &
                    - 400.0_dp*x(i)*(x(i + 1) - x(i)**2) &
                    + 2.0_dp*(x(i) - 1.0_dp)
                gradient(i + 1) = gradient(i + 1) + 200.0_dp*(x(i + 1) - x(i)**2)
            end do
        case (FORTBO_BENCH_LEVY)
            w = 1.0_dp + (x - 1.0_dp)/4.0_dp
            dw = 0.25_dp
            w_last = w(d)
            gradient(1) = gradient(1) &
                + 2.0_dp*sin(PI_VALUE*w(1))*cos(PI_VALUE*w(1))*PI_VALUE*dw
            do i = 1, d - 1
                gradient(i) = gradient(i) + dw*( &
                    2.0_dp*(w(i) - 1.0_dp) &
                    *(1.0_dp + 10.0_dp*sin(PI_VALUE*w(i) + 1.0_dp)**2) &
                    + (w(i) - 1.0_dp)**2*20.0_dp &
                    *sin(PI_VALUE*w(i) + 1.0_dp) &
                    *cos(PI_VALUE*w(i) + 1.0_dp)*PI_VALUE)
            end do
            gradient(d) = gradient(d) + dw*( &
                2.0_dp*(w_last - 1.0_dp) &
                *(1.0_dp + sin(2.0_dp*PI_VALUE*w_last)**2) &
                + (w_last - 1.0_dp)**2*2.0_dp*sin(2.0_dp*PI_VALUE*w_last) &
                *cos(2.0_dp*PI_VALUE*w_last)*2.0_dp*PI_VALUE)
        case (FORTBO_BENCH_SPHERE)
            gradient = 2.0_dp*x
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: unknown function")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine benchmark_gradient

    subroutine benchmark_bounds(self, lower, upper, status)
        class(fortbo_benchmark_t), intent(in) :: self
        real(dp), intent(out) :: lower(:)
        real(dp), intent(out) :: upper(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_valid() .or. size(lower) /= self%dimension &
            .or. size(upper) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: bound width does not match")
            return
        end if
        select case (self%kind)
        case (FORTBO_BENCH_BRANIN)
            lower = [-5.0_dp, 0.0_dp]
            upper = [10.0_dp, 15.0_dp]
        case (FORTBO_BENCH_HARTMANN3, FORTBO_BENCH_HARTMANN6)
            lower = 0.0_dp
            upper = 1.0_dp
        case (FORTBO_BENCH_ACKLEY)
            lower = -5.0_dp
            upper = 10.0_dp
        case (FORTBO_BENCH_ROSENBROCK)
            lower = -5.0_dp
            upper = 10.0_dp
        case (FORTBO_BENCH_LEVY)
            lower = -10.0_dp
            upper = 10.0_dp
        case (FORTBO_BENCH_SPHERE)
            lower = -5.12_dp
            upper = 5.12_dp
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine benchmark_bounds

    !! A global minimizer. Several of these functions have more than one; the
    !! one returned is the conventionally quoted one, and the value is what
    !! matters for regret.
    subroutine benchmark_optimum(self, point, status)
        class(fortbo_benchmark_t), intent(in) :: self
        real(dp), intent(out) :: point(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_valid() .or. size(point) /= self%dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo benchmark: optimum width does not match")
            return
        end if
        select case (self%kind)
        case (FORTBO_BENCH_BRANIN)
            point = [PI_VALUE, 2.275_dp]
        case (FORTBO_BENCH_HARTMANN3)
            point = [0.114614_dp, 0.555649_dp, 0.852547_dp]
        case (FORTBO_BENCH_HARTMANN6)
            point = [0.20169_dp, 0.150011_dp, 0.476874_dp, 0.275332_dp, &
                0.311652_dp, 0.6573_dp]
        case (FORTBO_BENCH_ACKLEY, FORTBO_BENCH_SPHERE)
            point = 0.0_dp
        case (FORTBO_BENCH_ROSENBROCK, FORTBO_BENCH_LEVY)
            point = 1.0_dp
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine benchmark_optimum

    !! The published optimal value. Regret is measured against this, never
    !! against the best value a run happened to find.
    pure real(dp) function benchmark_optimal_value(self) result(value)
        class(fortbo_benchmark_t), intent(in) :: self

        select case (self%kind)
        case (FORTBO_BENCH_BRANIN)
            value = 0.397887357729738_dp
        case (FORTBO_BENCH_HARTMANN3)
            value = -3.86278214782076_dp
        case (FORTBO_BENCH_HARTMANN6)
            value = -3.32236801141551_dp
        case default
            value = 0.0_dp
        end select
    end function benchmark_optimal_value

end module fortbo_benchmarks
