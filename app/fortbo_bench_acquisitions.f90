program fortbo_bench_acquisitions
    !! Time every acquisition FortBO offers on a shared posterior and candidate
    !! set, and print its values so accuracy can be checked alongside speed.
    !!
    !! The TuRBO speed benchmark measures one policy. It says nothing about the
    !! acquisitions, which is where most of FortBO's surface area is and where
    !! a per-candidate cost is easiest to get wrong: an acquisition is called
    !! on thousands of points per step, so a constant factor here multiplies
    !! straight through every policy above it.
    !!
    !! One posterior, one candidate set, every acquisition. The model is pinned
    !! rather than fitted for the same reason the cross-framework benchmark
    !! pins it: fitting on each side would compare optimizers and report the
    !! difference as an acquisition cost.
    !!
    !! Values are printed beside the timings and not only summarized, because a
    !! fast acquisition returning the wrong number is not a fast acquisition.
    !! The Python counterpart in `fortbo-bench` checks them against BoTorch.
    !!
    !! Usage:
    !!
    !!     fo exec fortbo_bench_acquisitions N_CANDIDATES N_TRAIN N_SAMPLES

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortbo_acquisition, only: fortbo_acquisition_t, fortbo_ei_t, fortbo_log_ei_t, fortbo_pi_t, &
        fortbo_ucb_t
    use fortbo_monte_carlo, only: fortbo_mc_base_t, fortbo_mc_ei_t, &
        fortbo_mc_pi_t, fortbo_mc_noisy_ei_t
    use fortbo_entropy, only: fortbo_max_value_entropy_search
    use fortbo_fortml, only: fortbo_gp_posterior_t
    use fortbo_posterior, only: fortbo_posterior_t
    implicit none

    integer :: n_candidates, n_train, n_samples
    character(len=32) :: argument
    integer :: length, k, j
    real(dp), allocatable :: train_x(:, :), train_y(:, :), candidates(:, :)
    real(dp), allocatable :: values(:), mean(:), sd(:), variance(:)
    real(dp), allocatable :: optimum_samples(:)
    type(kernel_t) :: kernel
    type(fortbo_gp_posterior_t) :: posterior
    type(fortnum_status_t) :: status
    type(rng_t) :: generator
    type(fortbo_ei_t) :: ei
    type(fortbo_log_ei_t) :: log_ei
    type(fortbo_pi_t) :: pi
    type(fortbo_ucb_t) :: ucb
    type(fortbo_mc_ei_t) :: mc_ei
    type(fortbo_mc_pi_t) :: mc_pi
    type(fortbo_mc_noisy_ei_t) :: mc_noisy
    type(fortbo_mc_base_t) :: base
    real(dp) :: best, started, finished, draw
    integer, parameter :: dimension = 8

    n_candidates = 4000
    n_train = 40
    n_samples = 128
    if (command_argument_count() >= 1) then
        call get_command_argument(1, argument, length=length)
        read (argument(:length), *) n_candidates
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, argument, length=length)
        read (argument(:length), *) n_train
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, argument, length=length)
        read (argument(:length), *) n_samples
    end if

    ! A stated training set and candidate set, so the Python counterpart can
    ! reproduce the identical inputs without shipping arrays between them.
    allocate (train_x(n_train, dimension), train_y(n_train, 1))
    do k = 1, n_train
        do j = 1, dimension
            train_x(k, j) = sin(0.37_dp*real(k*j, dp))
        end do
        train_y(k, 1) = sum(sin(1.3_dp*train_x(k, :))) + 0.2_dp*real(k, dp)/ &
            real(n_train, dp)
    end do
    ! Candidates near the training region. An earlier version placed them on
    ! a wide cosine sweep, where the posterior mean sat far above the
    ! incumbent everywhere: expected improvement summed to 7e-5 and the Monte
    ! Carlo acquisitions returned exactly zero because no sample beat the
    ! incumbent. That times the code but measures nothing, since the branch
    ! where the work happens is never taken.
    allocate (candidates(n_candidates, dimension))
    do k = 1, n_candidates
        do j = 1, dimension
            candidates(k, j) = sin(0.37_dp*real(mod(k, n_train) + 1, dp)*real(j, dp)) &
                + 0.15_dp*cos(0.031_dp*real(k, dp) + 0.9_dp*real(j, dp))
        end do
    end do

    kernel = make_rbf_kernel(dimension, 1.3_dp, 0.7_dp, status)
    call posterior%model%fit(train_x, train_y, kernel, 0.05_dp, status, &
        jitter=0.0_dp)
    if (status%code /= FORTNUM_OK) then
        print *, "fit failed: ", trim(status%msg)
        error stop 1
    end if
    posterior%dimension = dimension
    posterior%fitted = .true.

    best = minval(train_y(:, 1))
    allocate (values(n_candidates), mean(n_candidates), sd(n_candidates))
    allocate (variance(n_candidates))

    print *, "CONFIG ", n_candidates, n_train, n_samples, dimension

    ! The posterior evaluation itself, timed separately. Every acquisition
    ! pays it, so reporting it apart is what stops a slow acquisition hiding
    ! behind a slow posterior or the reverse.
    call cpu_time(started)
    call posterior%moments(candidates, mean, variance, status)
    call cpu_time(finished)
    sd = sqrt(max(variance, 0.0_dp))
    print *, "TIME posterior_moments ", finished - started, sum(mean), sum(sd)

    ei%best = best
    ei%xi = 0.0_dp
    call time_acquisition("ei", ei)
    log_ei%best = best
    log_ei%xi = 0.0_dp
    call time_acquisition("log_ei", log_ei)
    pi%best = best
    pi%xi = 0.0_dp
    call time_acquisition("pi", pi)
    ucb%best = best
    ucb%beta = 2.0_dp
    call time_acquisition("ucb", ucb)

    call rng_seed(generator, 20260808_8, status)
    call base%generate(n_candidates, n_samples, generator, status)
    if (status%code /= FORTNUM_OK) then
        print *, "base draws failed: ", trim(status%msg)
        error stop 1
    end if
    mc_ei%base = base
    mc_ei%best = best
    call time_acquisition("mc_ei", mc_ei)
    mc_pi%base = base
    mc_pi%best = best
    call time_acquisition("mc_pi", mc_pi)
    mc_noisy%base = base
    mc_noisy%best = best
    call time_acquisition("mc_noisy_ei", mc_noisy)

    ! Max-value entropy search takes sampled optima rather than a posterior,
    ! so it is timed on the same moments the others used.
    allocate (optimum_samples(32))
    do k = 1, size(optimum_samples)
        call rng_uniform(generator, draw)
        optimum_samples(k) = best - 0.5_dp*draw
    end do
    call cpu_time(started)
    call fortbo_max_value_entropy_search(mean, sd, optimum_samples, values, &
        status)
    call cpu_time(finished)
    print *, "TIME mes ", finished - started, sum(values), &
        merge(0, 1, status%code == FORTNUM_OK)

contains

    subroutine time_acquisition(name, acquisition)
        character(len=*), intent(in) :: name
        class(fortbo_acquisition_t), intent(in) :: acquisition
        real(dp) :: begin, done

        call cpu_time(begin)
        call acquisition%value(posterior, candidates, values, status)
        call cpu_time(done)
        print *, "TIME "//name//" ", done - begin, sum(values), &
            merge(0, 1, status%code == FORTNUM_OK)
        if (status%code /= FORTNUM_OK) print *, "   refused: ", trim(status%msg)
    end subroutine time_acquisition

end program fortbo_bench_acquisitions
