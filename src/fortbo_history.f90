module fortbo_history
    !! Observation history for sequential design (ROADMAP BO0).
    !!
    !! The history is the only durable state of a Bayesian-optimization run. It
    !! records what was queried, what came back, what it cost, and what did not
    !! come back at all. Three things it deliberately does not do:
    !!
    !!   * it does not invent values for failed evaluations. A missing objective
    !!     stays missing, and the configured policy decides whether the caller
    !!     is refused, the row is skipped, or the row is imputed with an
    !!     explicit worst-case value that is recorded as imputed;
    !!   * it does not silently swallow repeated queries. Duplicates are
    !!     detected in the infinity norm and handled by a declared policy, so a
    !!     surrogate is never handed a numerically singular design matrix by
    !!     accident;
    !!   * it does not reorder rows. Insertion order is the replay order, which
    !!     is what makes a checkpoint a faithful description of the run.
    !!
    !! Constraints follow the standard sign convention: `constraint <= 0` is
    !! feasible.
    !!
    !! Derivative observations are first-class and orthogonal to everything
    !! else. A row may carry a gradient, part of a gradient, or none, and the
    !! per-coordinate presence flags say which. Nothing in this module treats a
    !! gradient as belonging to a particular optimization policy: a run that
    !! collects adjoints feeds them to whichever surrogate is configured, and
    !! every acquisition and policy downstream sees only the resulting
    !! posterior. That is what makes derivative information usable with *any*
    !! method rather than only with the derivative-aware ones.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    !! Checkpoint format version, independent of the posterior contract version.
    integer, parameter, public :: FORTBO_HISTORY_FORMAT_VERSION = 1

    ! Evaluation outcome for one row.
    integer, parameter, public :: FORTBO_OUTCOME_PENDING = 0
    integer, parameter, public :: FORTBO_OUTCOME_OK = 1
    integer, parameter, public :: FORTBO_OUTCOME_FAILED = 2

    ! Missing-observation policy.
    integer, parameter, public :: FORTBO_MISSING_REFUSE = 0
    integer, parameter, public :: FORTBO_MISSING_SKIP = 1
    integer, parameter, public :: FORTBO_MISSING_IMPUTE_WORST = 2

    ! Duplicate-query policy.
    integer, parameter, public :: FORTBO_DUPLICATE_ALLOW = 0
    integer, parameter, public :: FORTBO_DUPLICATE_REJECT = 1
    integer, parameter, public :: FORTBO_DUPLICATE_REPLACE = 2

    integer, parameter :: INITIAL_CAPACITY = 16

    public :: fortbo_history_t

    type :: fortbo_history_t
        integer :: n_inputs = 0
        integer :: n_constraints = 0
        integer :: count = 0
        integer :: missing_policy = FORTBO_MISSING_REFUSE
        integer :: duplicate_policy = FORTBO_DUPLICATE_ALLOW
        real(dp) :: duplicate_tolerance = 0.0_dp
        real(dp), allocatable :: inputs(:, :)
        real(dp), allocatable :: objectives(:)
        logical, allocatable :: objective_present(:)
        logical, allocatable :: objective_imputed(:)
        real(dp), allocatable :: costs(:)
        real(dp), allocatable :: constraints(:, :)
        logical, allocatable :: constraint_present(:, :)
        real(dp), allocatable :: gradients(:, :)
        logical, allocatable :: gradient_present(:, :)
        integer, allocatable :: outcome(:)
        integer, allocatable :: batch(:)
    contains
        procedure, public :: initialize => history_initialize
        procedure, public :: add => history_add
        procedure, public :: add_pending => history_add_pending
        procedure, public :: complete => history_complete
        procedure, public :: usable_count => history_usable_count
        procedure, public :: pending_count => history_pending_count
        procedure, public :: is_feasible => history_is_feasible
        procedure, public :: is_usable => history_is_usable
        procedure, public :: has_gradient => history_has_gradient
        procedure, public :: gradient_count => history_gradient_count
        procedure, public :: gradient_data => history_gradient_data
        procedure, public :: best_index => history_best_index
        procedure, public :: incumbent => history_incumbent
        procedure, public :: total_cost => history_total_cost
        procedure, public :: find_duplicate => history_find_duplicate
        procedure, public :: training_data => history_training_data
        procedure, public :: checkpoint_write => history_checkpoint_write
        procedure, public :: checkpoint_read => history_checkpoint_read
    end type fortbo_history_t

contains

    subroutine history_initialize(self, n_inputs, n_constraints, status)
        class(fortbo_history_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        integer, intent(in) :: n_constraints
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: n_inputs must be positive")
            return
        end if
        if (n_constraints < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: n_constraints must not be negative")
            return
        end if
        self%n_inputs = n_inputs
        self%n_constraints = n_constraints
        self%count = 0
        allocate (self%inputs(INITIAL_CAPACITY, n_inputs))
        allocate (self%objectives(INITIAL_CAPACITY))
        allocate (self%objective_present(INITIAL_CAPACITY))
        allocate (self%objective_imputed(INITIAL_CAPACITY))
        allocate (self%costs(INITIAL_CAPACITY))
        allocate (self%constraints(INITIAL_CAPACITY, max(n_constraints, 1)))
        allocate (self%constraint_present(INITIAL_CAPACITY, max(n_constraints, 1)))
        allocate (self%gradients(INITIAL_CAPACITY, n_inputs))
        allocate (self%gradient_present(INITIAL_CAPACITY, n_inputs))
        allocate (self%outcome(INITIAL_CAPACITY))
        allocate (self%batch(INITIAL_CAPACITY))
        self%inputs = 0.0_dp
        self%objectives = 0.0_dp
        self%objective_present = .false.
        self%objective_imputed = .false.
        self%costs = 0.0_dp
        self%constraints = 0.0_dp
        self%constraint_present = .false.
        self%gradients = 0.0_dp
        self%gradient_present = .false.
        self%outcome = FORTBO_OUTCOME_PENDING
        self%batch = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_initialize

    !! Append one evaluated row. `objective` absent means the evaluation
    !! produced no usable value; the missing policy then decides what happens.
    subroutine history_add(self, input, status, objective, cost, constraints, &
            batch, outcome, gradient, gradient_mask)
        class(fortbo_history_t), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: objective
        real(dp), intent(in), optional :: cost
        real(dp), intent(in), optional :: constraints(:)
        integer, intent(in), optional :: batch
        integer, intent(in), optional :: outcome
        !! Objective gradient at `input`, and an optional per-coordinate mask
        !! selecting which components were actually measured. Absent mask means
        !! every component of `gradient` is present.
        real(dp), intent(in), optional :: gradient(:)
        logical, intent(in), optional :: gradient_mask(:)
        integer :: row, duplicate
        logical :: have_objective

        if (self%n_inputs == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: not initialized")
            return
        end if
        if (size(input) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: input width does not match n_inputs")
            return
        end if
        if (present(constraints)) then
            if (size(constraints) /= self%n_constraints) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: constraint count does not match")
                return
            end if
        end if
        call check_gradient_shapes(self%n_inputs, gradient, gradient_mask, status)
        if (status%code /= FORTNUM_OK) return

        have_objective = present(objective)
        if (.not. have_objective) then
            select case (self%missing_policy)
            case (FORTBO_MISSING_REFUSE)
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "fortbo history: missing objective refused by policy")
                return
            case (FORTBO_MISSING_SKIP)
                call status_set(status, FORTNUM_OK, "")
                return
            case (FORTBO_MISSING_IMPUTE_WORST)
                continue
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: unknown missing-observation policy")
                return
            end select
        end if

        duplicate = self%find_duplicate(input)
        if (duplicate > 0) then
            select case (self%duplicate_policy)
            case (FORTBO_DUPLICATE_REJECT)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: duplicate query rejected by policy")
                return
            case (FORTBO_DUPLICATE_REPLACE)
                row = duplicate
            case (FORTBO_DUPLICATE_ALLOW)
                row = 0
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: unknown duplicate policy")
                return
            end select
        else
            row = 0
        end if

        if (row == 0) then
            call grow_if_needed(self)
            self%count = self%count + 1
            row = self%count
        end if

        self%inputs(row, :) = input
        self%objective_present(row) = have_objective
        self%objective_imputed(row) = .not. have_objective
        if (have_objective) then
            self%objectives(row) = objective
        else
            self%objectives(row) = huge(1.0_dp)
        end if
        self%costs(row) = 0.0_dp
        if (present(cost)) self%costs(row) = cost
        self%constraints(row, :) = 0.0_dp
        self%constraint_present(row, :) = .false.
        if (present(constraints) .and. self%n_constraints > 0) then
            self%constraints(row, 1:self%n_constraints) = constraints
            self%constraint_present(row, 1:self%n_constraints) = .true.
        end if
        call store_gradient(self, row, gradient, gradient_mask)
        self%batch(row) = 0
        if (present(batch)) self%batch(row) = batch
        if (present(outcome)) then
            self%outcome(row) = outcome
        else if (have_objective) then
            self%outcome(row) = FORTBO_OUTCOME_OK
        else
            self%outcome(row) = FORTBO_OUTCOME_FAILED
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_add

    !! Register a query that has been dispatched but not yet answered. Pending
    !! rows are what asynchronous batch policies fantasize over; they never
    !! reach a surrogate as training data.
    subroutine history_add_pending(self, input, row, status, batch)
        class(fortbo_history_t), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        integer, intent(out) :: row
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: batch

        row = 0
        if (size(input) /= self%n_inputs .or. self%n_inputs == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: input width does not match n_inputs")
            return
        end if
        call grow_if_needed(self)
        self%count = self%count + 1
        row = self%count
        self%inputs(row, :) = input
        self%objectives(row) = 0.0_dp
        self%objective_present(row) = .false.
        self%objective_imputed(row) = .false.
        self%costs(row) = 0.0_dp
        self%constraints(row, :) = 0.0_dp
        self%constraint_present(row, :) = .false.
        self%gradients(row, :) = 0.0_dp
        self%gradient_present(row, :) = .false.
        self%outcome(row) = FORTBO_OUTCOME_PENDING
        self%batch(row) = 0
        if (present(batch)) self%batch(row) = batch
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_add_pending

    !! Fill in a pending row. Completing a row twice is an error, because a
    !! second answer to the same dispatch means the bookkeeping has lost track
    !! of which worker produced which value.
    subroutine history_complete(self, row, status, objective, cost, constraints, &
            outcome, gradient, gradient_mask)
        class(fortbo_history_t), intent(inout) :: self
        integer, intent(in) :: row
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: objective
        real(dp), intent(in), optional :: cost
        real(dp), intent(in), optional :: constraints(:)
        integer, intent(in), optional :: outcome
        real(dp), intent(in), optional :: gradient(:)
        logical, intent(in), optional :: gradient_mask(:)

        if (row < 1 .or. row > self%count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: completion row out of range")
            return
        end if
        if (self%outcome(row) /= FORTBO_OUTCOME_PENDING) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: row is not pending")
            return
        end if
        call check_gradient_shapes(self%n_inputs, gradient, gradient_mask, status)
        if (status%code /= FORTNUM_OK) return
        call store_gradient(self, row, gradient, gradient_mask)
        if (present(objective)) then
            self%objectives(row) = objective
            self%objective_present(row) = .true.
            self%outcome(row) = FORTBO_OUTCOME_OK
        else
            self%outcome(row) = FORTBO_OUTCOME_FAILED
            if (self%missing_policy == FORTBO_MISSING_IMPUTE_WORST) then
                self%objectives(row) = huge(1.0_dp)
                self%objective_imputed(row) = .true.
            end if
        end if
        if (present(cost)) self%costs(row) = cost
        if (present(constraints) .and. self%n_constraints > 0) then
            if (size(constraints) /= self%n_constraints) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: constraint count does not match")
                return
            end if
            self%constraints(row, 1:self%n_constraints) = constraints
            self%constraint_present(row, 1:self%n_constraints) = .true.
        end if
        if (present(outcome)) self%outcome(row) = outcome
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_complete

    !! A row is usable when it carries a real, non-imputed objective.
    pure logical function history_is_usable(self, row) result(usable)
        class(fortbo_history_t), intent(in) :: self
        integer, intent(in) :: row

        usable = .false.
        if (row < 1 .or. row > self%count) return
        usable = self%objective_present(row) .and. .not. self%objective_imputed(row)
    end function history_is_usable

    !! True when every coordinate of the gradient at `row` was measured. A
    !! partially measured gradient is deliberately not "has a gradient": a
    !! consumer that wants the partial information reads `gradient_present`
    !! directly and decides what to do with the holes.
    pure logical function history_has_gradient(self, row) result(has)
        class(fortbo_history_t), intent(in) :: self
        integer, intent(in) :: row

        has = .false.
        if (row < 1 .or. row > self%count) return
        has = all(self%gradient_present(row, 1:self%n_inputs))
    end function history_has_gradient

    pure integer function history_gradient_count(self) result(n)
        class(fortbo_history_t), intent(in) :: self
        integer :: row

        n = 0
        do row = 1, self%count
            if (self%is_usable(row) .and. self%has_gradient(row)) n = n + 1
        end do
    end function history_gradient_count

    !! Pack the rows that carry both a usable objective and a complete gradient
    !! into the dense arrays a derivative-observation surrogate expects. Any
    !! surrogate may consume these; nothing here is specific to one policy.
    subroutine history_gradient_data(self, inputs, objectives, gradients, status)
        class(fortbo_history_t), intent(in) :: self
        real(dp), intent(out), allocatable :: inputs(:, :)
        real(dp), intent(out), allocatable :: objectives(:)
        real(dp), intent(out), allocatable :: gradients(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: row, n, k

        n = self%gradient_count()
        allocate (inputs(n, self%n_inputs))
        allocate (objectives(n))
        allocate (gradients(n, self%n_inputs))
        k = 0
        do row = 1, self%count
            if (.not. self%is_usable(row)) cycle
            if (.not. self%has_gradient(row)) cycle
            k = k + 1
            inputs(k, :) = self%inputs(row, :)
            objectives(k) = self%objectives(row)
            gradients(k, :) = self%gradients(row, 1:self%n_inputs)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_gradient_data

    pure subroutine check_gradient_shapes(n_inputs, gradient, gradient_mask, status)
        integer, intent(in) :: n_inputs
        real(dp), intent(in), optional :: gradient(:)
        logical, intent(in), optional :: gradient_mask(:)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_OK, "")
        if (present(gradient)) then
            if (size(gradient) /= n_inputs) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: gradient width does not match n_inputs")
                return
            end if
        end if
        if (present(gradient_mask)) then
            if (.not. present(gradient)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: gradient mask without a gradient")
                return
            end if
            if (size(gradient_mask) /= n_inputs) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "fortbo history: gradient mask width does not match")
                return
            end if
        end if
    end subroutine check_gradient_shapes

    !! Store a gradient row, honoring a partial mask. Unmeasured coordinates
    !! keep a zero value that is flagged absent, so a consumer that ignores the
    !! flags gets a visibly wrong answer rather than a plausible one.
    subroutine store_gradient(self, row, gradient, gradient_mask)
        type(fortbo_history_t), intent(inout) :: self
        integer, intent(in) :: row
        real(dp), intent(in), optional :: gradient(:)
        logical, intent(in), optional :: gradient_mask(:)
        integer :: j

        if (.not. present(gradient)) return
        do j = 1, self%n_inputs
            if (present(gradient_mask)) then
                if (.not. gradient_mask(j)) then
                    self%gradients(row, j) = 0.0_dp
                    self%gradient_present(row, j) = .false.
                    cycle
                end if
            end if
            self%gradients(row, j) = gradient(j)
            self%gradient_present(row, j) = .true.
        end do
    end subroutine store_gradient

    pure integer function history_usable_count(self) result(n)
        class(fortbo_history_t), intent(in) :: self
        integer :: row

        n = 0
        do row = 1, self%count
            if (self%is_usable(row)) n = n + 1
        end do
    end function history_usable_count

    pure integer function history_pending_count(self) result(n)
        class(fortbo_history_t), intent(in) :: self

        n = count(self%outcome(1:self%count) == FORTBO_OUTCOME_PENDING)
    end function history_pending_count

    !! Feasible means every declared constraint is present and non-positive. A
    !! row with an unmeasured constraint is not feasible; it is unknown, and
    !! unknown is not a licence to treat it as satisfied.
    pure logical function history_is_feasible(self, row) result(feasible)
        class(fortbo_history_t), intent(in) :: self
        integer, intent(in) :: row
        integer :: j

        feasible = .false.
        if (row < 1 .or. row > self%count) return
        do j = 1, self%n_constraints
            if (.not. self%constraint_present(row, j)) return
            if (self%constraints(row, j) > 0.0_dp) return
        end do
        feasible = .true.
    end function history_is_feasible

    !! Row index of the best feasible usable observation, or zero when none
    !! exists. Ties go to the earliest row so replay is deterministic.
    pure integer function history_best_index(self) result(best)
        class(fortbo_history_t), intent(in) :: self
        integer :: row
        real(dp) :: best_value

        best = 0
        best_value = huge(1.0_dp)
        do row = 1, self%count
            if (.not. self%is_usable(row)) cycle
            if (.not. self%is_feasible(row)) cycle
            if (self%objectives(row) < best_value) then
                best_value = self%objectives(row)
                best = row
            end if
        end do
    end function history_best_index

    subroutine history_incumbent(self, input, objective, status)
        class(fortbo_history_t), intent(in) :: self
        real(dp), intent(out) :: input(:)
        real(dp), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status
        integer :: best

        input = 0.0_dp
        objective = 0.0_dp
        best = self%best_index()
        if (best == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: no feasible observation yet")
            return
        end if
        if (size(input) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: incumbent output width does not match")
            return
        end if
        input = self%inputs(best, :)
        objective = self%objectives(best)
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_incumbent

    pure real(dp) function history_total_cost(self) result(total)
        class(fortbo_history_t), intent(in) :: self

        total = sum(self%costs(1:self%count))
    end function history_total_cost

    !! Index of the first stored row within `duplicate_tolerance` of `input` in
    !! the infinity norm, or zero. A zero tolerance still catches exact repeats.
    pure integer function history_find_duplicate(self, input) result(row)
        class(fortbo_history_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        integer :: i

        row = 0
        if (size(input) /= self%n_inputs) return
        do i = 1, self%count
            if (maxval(abs(self%inputs(i, :) - input)) <= self%duplicate_tolerance) then
                row = i
                return
            end if
        end do
    end function history_find_duplicate

    !! Pack the usable rows into the dense arrays a surrogate expects. Pending,
    !! failed, and imputed rows are excluded, so a model never trains on a value
    !! that was never measured.
    subroutine history_training_data(self, inputs, objectives, status)
        class(fortbo_history_t), intent(in) :: self
        real(dp), intent(out), allocatable :: inputs(:, :)
        real(dp), intent(out), allocatable :: objectives(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: row, n, k

        n = self%usable_count()
        allocate (inputs(n, self%n_inputs))
        allocate (objectives(n))
        k = 0
        do row = 1, self%count
            if (.not. self%is_usable(row)) cycle
            k = k + 1
            inputs(k, :) = self%inputs(row, :)
            objectives(k) = self%objectives(row)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_training_data

    !! Write a complete, human-readable checkpoint. Everything needed to resume
    !! is written, including the policies, because a run resumed under different
    !! policies is a different run.
    subroutine history_checkpoint_write(self, path, status)
        class(fortbo_history_t), intent(in) :: self
        character(len=*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, io_status, row, j

        open (newunit=unit, file=path, status="replace", action="write", &
            form="formatted", iostat=io_status)
        if (io_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: cannot open checkpoint for writing")
            return
        end if
        write (unit, *) FORTBO_HISTORY_FORMAT_VERSION
        write (unit, *) self%n_inputs, self%n_constraints, self%count
        write (unit, *) self%missing_policy, self%duplicate_policy
        write (unit, *) self%duplicate_tolerance
        do row = 1, self%count
            write (unit, *) (self%inputs(row, j), j=1, self%n_inputs)
            write (unit, *) self%objectives(row), self%objective_present(row), &
                self%objective_imputed(row), self%costs(row)
            write (unit, *) self%outcome(row), self%batch(row)
            write (unit, *) (self%gradients(row, j), j=1, self%n_inputs)
            write (unit, *) (self%gradient_present(row, j), j=1, self%n_inputs)
            if (self%n_constraints > 0) then
                write (unit, *) (self%constraints(row, j), j=1, self%n_constraints)
                write (unit, *) (self%constraint_present(row, j), &
                    j=1, self%n_constraints)
            end if
        end do
        close (unit)
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_checkpoint_write

    !! Read a checkpoint back. A format version this build does not know is
    !! refused rather than parsed optimistically.
    subroutine history_checkpoint_read(self, path, status)
        class(fortbo_history_t), intent(out) :: self
        character(len=*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, io_status, row, j, version, n_inputs, n_constraints, stored

        open (newunit=unit, file=path, status="old", action="read", &
            form="formatted", iostat=io_status)
        if (io_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: cannot open checkpoint for reading")
            return
        end if
        read (unit, *, iostat=io_status) version
        if (io_status /= 0 .or. version /= FORTBO_HISTORY_FORMAT_VERSION) then
            close (unit)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "fortbo history: unsupported checkpoint format version")
            return
        end if
        read (unit, *, iostat=io_status) n_inputs, n_constraints, stored
        if (io_status /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo history: malformed checkpoint header")
            return
        end if
        call history_initialize(self, n_inputs, n_constraints, status)
        if (status%code /= FORTNUM_OK) then
            close (unit)
            return
        end if
        read (unit, *) self%missing_policy, self%duplicate_policy
        read (unit, *) self%duplicate_tolerance
        do row = 1, stored
            call grow_if_needed(self)
            self%count = self%count + 1
            read (unit, *) (self%inputs(row, j), j=1, n_inputs)
            read (unit, *) self%objectives(row), self%objective_present(row), &
                self%objective_imputed(row), self%costs(row)
            read (unit, *) self%outcome(row), self%batch(row)
            read (unit, *) (self%gradients(row, j), j=1, n_inputs)
            read (unit, *) (self%gradient_present(row, j), j=1, n_inputs)
            if (n_constraints > 0) then
                read (unit, *) (self%constraints(row, j), j=1, n_constraints)
                read (unit, *) (self%constraint_present(row, j), j=1, n_constraints)
            end if
        end do
        close (unit)
        call status_set(status, FORTNUM_OK, "")
    end subroutine history_checkpoint_read

    !! Double the storage when the next append would not fit. Growth preserves
    !! row order, which the replay contract depends on.
    subroutine grow_if_needed(self)
        type(fortbo_history_t), intent(inout) :: self
        real(dp), allocatable :: inputs(:, :), objectives(:), costs(:)
        real(dp), allocatable :: constraints(:, :), gradients(:, :)
        logical, allocatable :: present_flags(:), imputed(:), constraint_flags(:, :)
        logical, allocatable :: gradient_flags(:, :)
        integer, allocatable :: outcome(:), batch(:)
        integer :: capacity, new_capacity, n

        capacity = size(self%objectives)
        if (self%count < capacity) return
        new_capacity = max(2*capacity, INITIAL_CAPACITY)
        n = self%count

        allocate (inputs(new_capacity, self%n_inputs))
        inputs = 0.0_dp
        inputs(1:n, :) = self%inputs(1:n, :)
        call move_alloc(inputs, self%inputs)

        allocate (objectives(new_capacity))
        objectives = 0.0_dp
        objectives(1:n) = self%objectives(1:n)
        call move_alloc(objectives, self%objectives)

        allocate (present_flags(new_capacity))
        present_flags = .false.
        present_flags(1:n) = self%objective_present(1:n)
        call move_alloc(present_flags, self%objective_present)

        allocate (imputed(new_capacity))
        imputed = .false.
        imputed(1:n) = self%objective_imputed(1:n)
        call move_alloc(imputed, self%objective_imputed)

        allocate (costs(new_capacity))
        costs = 0.0_dp
        costs(1:n) = self%costs(1:n)
        call move_alloc(costs, self%costs)

        allocate (constraints(new_capacity, max(self%n_constraints, 1)))
        constraints = 0.0_dp
        constraints(1:n, :) = self%constraints(1:n, :)
        call move_alloc(constraints, self%constraints)

        allocate (constraint_flags(new_capacity, max(self%n_constraints, 1)))
        constraint_flags = .false.
        constraint_flags(1:n, :) = self%constraint_present(1:n, :)
        call move_alloc(constraint_flags, self%constraint_present)

        allocate (gradients(new_capacity, self%n_inputs))
        gradients = 0.0_dp
        gradients(1:n, :) = self%gradients(1:n, :)
        call move_alloc(gradients, self%gradients)

        allocate (gradient_flags(new_capacity, self%n_inputs))
        gradient_flags = .false.
        gradient_flags(1:n, :) = self%gradient_present(1:n, :)
        call move_alloc(gradient_flags, self%gradient_present)

        allocate (outcome(new_capacity))
        outcome = FORTBO_OUTCOME_PENDING
        outcome(1:n) = self%outcome(1:n)
        call move_alloc(outcome, self%outcome)

        allocate (batch(new_capacity))
        batch = 0
        batch(1:n) = self%batch(1:n)
        call move_alloc(batch, self%batch)
    end subroutine grow_if_needed

end module fortbo_history
