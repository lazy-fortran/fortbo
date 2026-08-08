module fortbo_workers
    !! Asynchronous worker bookkeeping (ROADMAP BO3).
    !!
    !! An asynchronous run has evaluations in flight. That single fact changes
    !! what the acquisition must be told, and it is the reason this module
    !! exists rather than a comment saying "call `ask` again".
    !!
    !! **Pending points must be fantasized.** A point already dispatched will
    !! come back, and until it does the posterior still shows full uncertainty
    !! there. An acquisition that is not told about it will happily dispatch the
    !! same point to a second worker, and a batch of `q` asynchronous workers
    !! degenerates to `q` copies of one evaluation. The standard remedy is to
    !! condition on a *fantasized* value at each pending input — the posterior
    !! mean is the usual choice, and this module keeps the policy explicit
    !! because "constant liar" variants that fantasize the incumbent or the
    !! worst observed value behave very differently and a run should record
    !! which it used.
    !!
    !! **A failure is not a value.** An evaluation can fail, time out, or return
    !! something unusable. Recording a large number instead teaches the
    !! surrogate that the region is bad, which is a claim about the objective
    !! that nobody measured — the objective may be excellent there and the
    !! cluster merely unreliable. Failures are recorded as failures, retried up
    !! to a limit, and only then abandoned.
    !!
    !! **Cost is charged for failures too.** A timeout that burned an hour cost
    !! an hour. Charging only successes makes an unreliable configuration look
    !! cheap, which is exactly backwards for a cost-aware policy.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: fortbo_worker_pool_t
    public :: fortbo_task_t
    public :: fortbo_fantasy_name

    !! What to substitute for a pending evaluation.
    integer, parameter, public :: FORTBO_FANTASY_MEAN = 1
    integer, parameter, public :: FORTBO_FANTASY_INCUMBENT = 2
    integer, parameter, public :: FORTBO_FANTASY_WORST = 3

    !! Task states.
    integer, parameter, public :: FORTBO_TASK_IDLE = 0
    integer, parameter, public :: FORTBO_TASK_PENDING = 1
    integer, parameter, public :: FORTBO_TASK_DONE = 2
    integer, parameter, public :: FORTBO_TASK_FAILED = 3

    type :: fortbo_task_t
        integer :: state = FORTBO_TASK_IDLE
        real(dp), allocatable :: point(:)
        real(dp) :: value = 0.0_dp
        real(dp) :: cost = 0.0_dp
        integer :: attempts = 0
        !! Wall-clock budget for one attempt; zero means no timeout.
        real(dp) :: timeout_seconds = 0.0_dp
    end type fortbo_task_t

    type :: fortbo_worker_pool_t
        integer :: n_workers = 0
        integer :: n_inputs = 0
        integer :: max_attempts = 3
        integer :: fantasy_policy = FORTBO_FANTASY_MEAN
        type(fortbo_task_t), allocatable :: tasks(:)
        integer :: completed = 0
        integer :: abandoned = 0
        integer :: retries = 0
        !! Cost accumulated by *every* attempt, successful or not.
        real(dp) :: total_cost = 0.0_dp
    contains
        procedure, public :: initialize => pool_initialize
        procedure, public :: dispatch => pool_dispatch
        procedure, public :: complete => pool_complete
        procedure, public :: fail => pool_fail
        procedure, public :: pending_points => pool_pending_points
        procedure, public :: pending_count => pool_pending_count
        procedure, public :: idle_worker => pool_idle_worker
        procedure, public :: fantasize => pool_fantasize
    end type fortbo_worker_pool_t

contains

    pure function fortbo_fantasy_name(policy) result(name)
        integer, intent(in) :: policy
        character(len=:), allocatable :: name

        select case (policy)
        case (FORTBO_FANTASY_MEAN)
            name = "posterior_mean"
        case (FORTBO_FANTASY_INCUMBENT)
            name = "incumbent"
        case (FORTBO_FANTASY_WORST)
            name = "worst_observed"
        case default
            name = "unknown"
        end select
    end function fortbo_fantasy_name

    subroutine pool_initialize(self, n_workers, n_inputs, status, max_attempts, &
            fantasy_policy)
        class(fortbo_worker_pool_t), intent(out) :: self
        integer, intent(in) :: n_workers
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_attempts
        integer, intent(in), optional :: fantasy_policy
        integer :: k

        if (n_workers < 1 .or. n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: worker and input counts must be positive")
            return
        end if
        if (present(max_attempts)) then
            if (max_attempts < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo workers: at least one attempt must be allowed")
                return
            end if
            self%max_attempts = max_attempts
        end if
        if (present(fantasy_policy)) then
            if (fortbo_fantasy_name(fantasy_policy) == "unknown") then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo workers: unknown fantasy policy")
                return
            end if
            self%fantasy_policy = fantasy_policy
        end if

        self%n_workers = n_workers
        self%n_inputs = n_inputs
        allocate (self%tasks(n_workers))
        do k = 1, n_workers
            allocate (self%tasks(k)%point(n_inputs))
            self%tasks(k)%point = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_initialize

    pure integer function pool_idle_worker(self) result(worker)
        class(fortbo_worker_pool_t), intent(in) :: self
        integer :: k

        worker = 0
        if (.not. allocated(self%tasks)) return
        do k = 1, self%n_workers
            if (self%tasks(k)%state == FORTBO_TASK_PENDING) cycle
            worker = k
            return
        end do
    end function pool_idle_worker

    subroutine pool_dispatch(self, worker, point, status, timeout_seconds)
        class(fortbo_worker_pool_t), intent(inout) :: self
        integer, intent(in) :: worker
        real(dp), intent(in) :: point(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: timeout_seconds

        if (.not. allocated(self%tasks)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: pool is not initialized")
            return
        end if
        if (worker < 1 .or. worker > self%n_workers) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: worker index out of range")
            return
        end if
        if (size(point) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: point width does not match the pool")
            return
        end if
        if (self%tasks(worker)%state == FORTBO_TASK_PENDING) then
            ! Overwriting a pending task would lose the evaluation silently and
            ! leave the fantasy for a point nobody is working on.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: worker already has a pending evaluation")
            return
        end if

        self%tasks(worker)%point = point
        self%tasks(worker)%state = FORTBO_TASK_PENDING
        self%tasks(worker)%attempts = 1
        self%tasks(worker)%timeout_seconds = 0.0_dp
        if (present(timeout_seconds)) &
            self%tasks(worker)%timeout_seconds = timeout_seconds
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_dispatch

    subroutine pool_complete(self, worker, value, cost, status)
        class(fortbo_worker_pool_t), intent(inout) :: self
        integer, intent(in) :: worker
        real(dp), intent(in) :: value
        real(dp), intent(in) :: cost
        type(fortnum_status_t), intent(out) :: status

        call check_pending(self, worker, status)
        if (status%code /= FORTNUM_OK) return
        if (cost < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: cost must not be negative")
            return
        end if

        self%tasks(worker)%value = value
        self%tasks(worker)%cost = cost
        self%tasks(worker)%state = FORTBO_TASK_DONE
        self%completed = self%completed + 1
        self%total_cost = self%total_cost + cost
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_complete

    !! Record a failed attempt.
    !!
    !! `retrying` says whether the point goes back out. A failure is never
    !! turned into an objective value: the surrogate learns nothing about the
    !! objective from a crashed job, and teaching it that the region is bad is a
    !! claim nobody measured.
    subroutine pool_fail(self, worker, cost, retrying, status)
        class(fortbo_worker_pool_t), intent(inout) :: self
        integer, intent(in) :: worker
        real(dp), intent(in) :: cost
        logical, intent(out) :: retrying
        type(fortnum_status_t), intent(out) :: status

        retrying = .false.
        call check_pending(self, worker, status)
        if (status%code /= FORTNUM_OK) return
        if (cost < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: cost must not be negative")
            return
        end if

        ! Charged whether it worked or not: a timeout that burned an hour cost
        ! an hour, and charging only successes makes an unreliable
        ! configuration look cheap.
        self%total_cost = self%total_cost + cost
        if (self%tasks(worker)%attempts < self%max_attempts) then
            self%tasks(worker)%attempts = self%tasks(worker)%attempts + 1
            self%tasks(worker)%state = FORTBO_TASK_PENDING
            self%retries = self%retries + 1
            retrying = .true.
        else
            self%tasks(worker)%state = FORTBO_TASK_FAILED
            self%abandoned = self%abandoned + 1
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_fail

    pure integer function pool_pending_count(self) result(total)
        class(fortbo_worker_pool_t), intent(in) :: self
        integer :: k

        total = 0
        if (.not. allocated(self%tasks)) return
        do k = 1, self%n_workers
            if (self%tasks(k)%state == FORTBO_TASK_PENDING) total = total + 1
        end do
    end function pool_pending_count

    !! The inputs currently in flight, which the acquisition must condition on.
    subroutine pool_pending_points(self, points, n_found, status)
        class(fortbo_worker_pool_t), intent(in) :: self
        real(dp), intent(out) :: points(:, :)
        integer, intent(out) :: n_found
        type(fortnum_status_t), intent(out) :: status
        integer :: k

        n_found = 0
        points = 0.0_dp
        if (.not. allocated(self%tasks)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: pool is not initialized")
            return
        end if
        if (size(points, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: point width does not match the pool")
            return
        end if
        do k = 1, self%n_workers
            if (self%tasks(k)%state /= FORTBO_TASK_PENDING) cycle
            if (n_found >= size(points, 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo workers: pending buffer is too small")
                return
            end if
            n_found = n_found + 1
            points(n_found, :) = self%tasks(k)%point
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_pending_points

    !! Fantasized values for the pending points under the recorded policy.
    !!
    !! `posterior_mean` is supplied by the caller because this module has no
    !! surrogate; the other two policies ignore it and use the observed history
    !! instead. Which policy was used belongs in the run record: the "constant
    !! liar" variants deliberately bias the surrogate to push later points away
    !! from pending ones, and a run that used one is not comparable with a run
    !! that used another.
    subroutine pool_fantasize(self, posterior_mean, incumbent, worst_observed, &
            fantasy, n_found, status)
        class(fortbo_worker_pool_t), intent(in) :: self
        real(dp), intent(in) :: posterior_mean(:)
        real(dp), intent(in) :: incumbent
        real(dp), intent(in) :: worst_observed
        real(dp), intent(out) :: fantasy(:)
        integer, intent(out) :: n_found
        type(fortnum_status_t), intent(out) :: status
        integer :: k

        n_found = 0
        fantasy = 0.0_dp
        if (.not. allocated(self%tasks)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: pool is not initialized")
            return
        end if
        n_found = self%pending_count()
        if (size(posterior_mean) < n_found .or. size(fantasy) < n_found) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: fantasy buffers are too small")
            return
        end if

        do k = 1, n_found
            select case (self%fantasy_policy)
            case (FORTBO_FANTASY_MEAN)
                fantasy(k) = posterior_mean(k)
            case (FORTBO_FANTASY_INCUMBENT)
                fantasy(k) = incumbent
            case (FORTBO_FANTASY_WORST)
                fantasy(k) = worst_observed
            end select
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pool_fantasize

    pure subroutine check_pending(self, worker, status)
        class(fortbo_worker_pool_t), intent(in) :: self
        integer, intent(in) :: worker
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%tasks)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: pool is not initialized")
            return
        end if
        if (worker < 1 .or. worker > self%n_workers) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: worker index out of range")
            return
        end if
        if (self%tasks(worker)%state /= FORTBO_TASK_PENDING) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo workers: worker has no pending evaluation")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_pending

end module fortbo_workers
