module fortbo_normal
    !! The standard normal quantile function, shared by everything that turns a
    !! uniform draw into a Gaussian one.
    !!
    !! It lives in its own module because two independent policies need it —
    !! TuRBO's Thompson realizations and DTuRBO's truncated-normal `lambda` —
    !! and two copies of a rational approximation is two places for a
    !! transcription error to hide.

    use fortnum_kinds, only: dp
    implicit none
    private

    public :: fortbo_inverse_normal
    public :: fortbo_half_normal
    public :: fortbo_symmetric_truncated_normal

contains

    !! Inverse standard normal CDF by Acklam's rational approximation, accurate
    !! to about 1.15e-9 relative across the whole range — far beyond what any
    !! sampling use here can resolve.
    pure real(dp) function fortbo_inverse_normal(p) result(z)
        real(dp), intent(in) :: p
        real(dp), parameter :: a(6) = [-3.969683028665376e+01_dp, &
            2.209460984245205e+02_dp, -2.759285104469687e+02_dp, &
            1.383577518672690e+02_dp, -3.066479806614716e+01_dp, &
            2.506628277459239e+00_dp]
        real(dp), parameter :: b(5) = [-5.447609879822406e+01_dp, &
            1.615858368580409e+02_dp, -1.556989798598866e+02_dp, &
            6.680131188771972e+01_dp, -1.328068155288572e+01_dp]
        real(dp), parameter :: c(6) = [-7.784894002430293e-03_dp, &
            -3.223964580411365e-01_dp, -2.400758277161838e+00_dp, &
            -2.549732539343734e+00_dp, 4.374664141464968e+00_dp, &
            2.938163982698783e+00_dp]
        real(dp), parameter :: d(4) = [7.784695709041462e-03_dp, &
            3.224671290700398e-01_dp, 2.445134137142996e+00_dp, &
            3.754408661907416e+00_dp]
        real(dp), parameter :: split_low = 0.02425_dp
        real(dp) :: q, r, clamped

        clamped = min(max(p, 1.0e-300_dp), 1.0_dp - 1.0e-16_dp)
        if (clamped < split_low) then
            q = sqrt(-2.0_dp*log(clamped))
            z = (((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6))/ &
                ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0_dp)
        else if (clamped <= 1.0_dp - split_low) then
            q = clamped - 0.5_dp
            r = q*q
            z = (((((a(1)*r + a(2))*r + a(3))*r + a(4))*r + a(5))*r + a(6))*q/ &
                (((((b(1)*r + b(2))*r + b(3))*r + b(4))*r + b(5))*r + 1.0_dp)
        else
            q = sqrt(-2.0_dp*log(1.0_dp - clamped))
            z = -(((((c(1)*q + c(2))*q + c(3))*q + c(4))*q + c(5))*q + c(6))/ &
                ((((d(1)*q + d(2))*q + d(3))*q + d(4))*q + 1.0_dp)
        end if
    end function fortbo_inverse_normal

    !! A standard normal truncated to the non-negative half line, from a uniform
    !! draw. Mapping `u` to `Phi^-1((1 + u)/2)` is exact: the truncated
    !! variable's CDF is `2*Phi(x) - 1` on `x >= 0`, so inverting it gives
    !! precisely this. Taking `abs` of a normal draw would give the same
    !! distribution but would consume the draw differently and break replay
    !! against a recorded stream.
    pure real(dp) function fortbo_half_normal(uniform) result(z)
        real(dp), intent(in) :: uniform

        z = fortbo_inverse_normal(0.5_dp*(1.0_dp + min(max(uniform, 0.0_dp), 1.0_dp)))
    end function fortbo_half_normal

    !! A standard normal truncated to `(-bound, bound)`, from a uniform draw.
    !!
    !! Inverting the truncated CDF, `F(x) = (Phi(x) - Phi(-b)) / (Phi(b) -
    !! Phi(-b))`, is exact and consumes exactly one uniform. Rejection sampling
    !! would give the same distribution but consume an unpredictable number of
    !! draws, which breaks replay against a recorded stream.
    pure real(dp) function fortbo_symmetric_truncated_normal(uniform, bound) &
            result(z)
        real(dp), intent(in) :: uniform
        real(dp), intent(in) :: bound
        real(dp) :: lower_mass, upper_mass, clamped

        if (bound <= 0.0_dp) then
            z = 0.0_dp
            return
        end if
        lower_mass = 0.5_dp*(1.0_dp + erf(-bound/sqrt(2.0_dp)))
        upper_mass = 0.5_dp*(1.0_dp + erf(bound/sqrt(2.0_dp)))
        clamped = min(max(uniform, 0.0_dp), 1.0_dp)
        z = fortbo_inverse_normal(lower_mass + clamped*(upper_mass - lower_mass))
        ! Rounding in the quantile function can push the result a hair outside
        ! the interval, and a caller relying on the bound should get the bound.
        z = min(max(z, -bound), bound)
    end function fortbo_symmetric_truncated_normal

end module fortbo_normal
