module fortbo_space
    !! Normalized search spaces (ROADMAP BO0).
    !!
    !! Every surrogate, acquisition, and candidate optimizer in FortBO works in
    !! the unit hypercube. This module owns the only translation between that
    !! cube and the user's variables, so no other component has to know that a
    !! coordinate is really an integer, a log-scaled rate, or one slot of a
    !! one-hot block.
    !!
    !! Representations. A point has two forms:
    !!   * the *user* form, one real per declared variable: a continuous value,
    !!     an integer value stored as a real, or a one-based category index
    !!     stored as a real;
    !!   * the *unit* form, one real per normalized coordinate, every coordinate
    !!     in [0, 1]. Continuous and integer variables occupy one coordinate;
    !!     a categorical variable with `k` categories occupies `k` coordinates
    !!     that are decoded by arg-max, which is what keeps the search region a
    !!     plain box that a bound-constrained optimizer can handle.
    !!
    !! Differentiability is a property of the normalized coordinate, not of the
    !! space as a whole. `differentiable_mask` marks the continuous coordinates;
    !! integer and categorical coordinates are marked non-differentiable and any
    !! derivative product against them must be refused by name rather than
    !! approximated. That refusal is the whole point: a relaxed integer
    !! coordinate has a derivative that means nothing at the rounded point.
    !!
    !! Conditional variables are declared with a parent and the parent category
    !! that activates them. An inactive coordinate is pinned to its default and
    !! is excluded from `active_mask`, so an optimizer cannot spend budget
    !! exploring a coordinate the objective ignores.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: FORTBO_VAR_CONTINUOUS = 1
    integer, parameter, public :: FORTBO_VAR_INTEGER = 2
    integer, parameter, public :: FORTBO_VAR_CATEGORICAL = 3

    integer, parameter :: MAX_NAME = 48
    integer, parameter :: INITIAL_CAPACITY = 8

    public :: fortbo_variable_t
    public :: fortbo_space_t

    type :: fortbo_variable_t
        integer :: kind = FORTBO_VAR_CONTINUOUS
        character(len=MAX_NAME) :: name = ""
        real(dp) :: lower = 0.0_dp
        real(dp) :: upper = 1.0_dp
        logical :: log_scale = .false.
        integer :: n_categories = 0
        !! First normalized coordinate owned by this variable, and how many.
        integer :: offset = 0
        integer :: width = 1
        !! Conditional activation: this variable is active only when variable
        !! `parent` takes category `parent_category`. Zero parent means always.
        integer :: parent = 0
        integer :: parent_category = 0
        real(dp) :: default_value = 0.0_dp
    end type fortbo_variable_t

    type :: fortbo_space_t
        integer :: count = 0
        integer :: n_normalized = 0
        logical :: finalized = .false.
        type(fortbo_variable_t), allocatable :: variables(:)
    contains
        procedure, public :: add_continuous => space_add_continuous
        procedure, public :: add_integer => space_add_integer
        procedure, public :: add_categorical => space_add_categorical
        procedure, public :: set_condition => space_set_condition
        procedure, public :: finalize => space_finalize
        procedure, public :: n_variables => space_n_variables
        procedure, public :: n_coordinates => space_n_coordinates
        procedure, public :: to_unit => space_to_unit
        procedure, public :: from_unit => space_from_unit
        procedure, public :: clamp => space_clamp
        procedure, public :: contains_point => space_contains_point
        procedure, public :: differentiable_mask => space_differentiable_mask
        procedure, public :: active_mask => space_active_mask
        procedure, public :: apply_defaults => space_apply_defaults
        procedure, public :: variable_index => space_variable_index
    end type fortbo_space_t

contains

    subroutine space_add_continuous(self, name, lower, upper, status, log_scale)
        class(fortbo_space_t), intent(inout) :: self
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: lower
        real(dp), intent(in) :: upper
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: log_scale
        type(fortbo_variable_t) :: variable

        if (upper <= lower) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: continuous bounds must be increasing")
            return
        end if
        variable%kind = FORTBO_VAR_CONTINUOUS
        variable%name = name
        variable%lower = lower
        variable%upper = upper
        variable%width = 1
        variable%default_value = lower
        if (present(log_scale)) variable%log_scale = log_scale
        if (variable%log_scale .and. lower <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: log-scaled bounds must be positive")
            return
        end if
        call append(self, variable, status)
    end subroutine space_add_continuous

    subroutine space_add_integer(self, name, lower, upper, status)
        class(fortbo_space_t), intent(inout) :: self
        character(len=*), intent(in) :: name
        integer, intent(in) :: lower
        integer, intent(in) :: upper
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_variable_t) :: variable

        if (upper <= lower) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: integer bounds must be increasing")
            return
        end if
        variable%kind = FORTBO_VAR_INTEGER
        variable%name = name
        variable%lower = real(lower, dp)
        variable%upper = real(upper, dp)
        variable%width = 1
        variable%default_value = real(lower, dp)
        call append(self, variable, status)
    end subroutine space_add_integer

    subroutine space_add_categorical(self, name, n_categories, status)
        class(fortbo_space_t), intent(inout) :: self
        character(len=*), intent(in) :: name
        integer, intent(in) :: n_categories
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_variable_t) :: variable

        if (n_categories < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: a categorical needs at least two categories")
            return
        end if
        variable%name = name
        variable%kind = FORTBO_VAR_CATEGORICAL
        variable%lower = 1.0_dp
        variable%upper = real(n_categories, dp)
        variable%n_categories = n_categories
        variable%width = n_categories
        variable%default_value = 1.0_dp
        call append(self, variable, status)
    end subroutine space_add_categorical

    !! Make `child` active only when `parent` takes `parent_category`. The
    !! parent must be categorical, because a condition on a continuous value has
    !! no well-defined activation boundary for a candidate optimizer.
    subroutine space_set_condition(self, child, parent, parent_category, status, &
            default_value)
        class(fortbo_space_t), intent(inout) :: self
        integer, intent(in) :: child
        integer, intent(in) :: parent
        integer, intent(in) :: parent_category
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: default_value

        if (child < 1 .or. child > self%count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: conditional child index out of range")
            return
        end if
        if (parent < 1 .or. parent > self%count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: conditional parent index out of range")
            return
        end if
        if (parent == child) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: a variable cannot be its own parent")
            return
        end if
        if (parent > child) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: a parent must be declared before its child")
            return
        end if
        if (self%variables(parent)%kind /= FORTBO_VAR_CATEGORICAL) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: a conditional parent must be categorical")
            return
        end if
        if (parent_category < 1 .or. &
            parent_category > self%variables(parent)%n_categories) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: conditional parent category out of range")
            return
        end if
        self%variables(child)%parent = parent
        self%variables(child)%parent_category = parent_category
        if (present(default_value)) self%variables(child)%default_value = default_value
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_set_condition

    !! Assign normalized coordinate offsets. Nothing may be added afterwards,
    !! because every stored point would then refer to a different layout.
    subroutine space_finalize(self, status)
        class(fortbo_space_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset

        if (self%count < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: cannot finalize an empty space")
            return
        end if
        offset = 0
        do i = 1, self%count
            self%variables(i)%offset = offset
            offset = offset + self%variables(i)%width
        end do
        self%n_normalized = offset
        self%finalized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_finalize

    pure integer function space_n_variables(self) result(n)
        class(fortbo_space_t), intent(in) :: self

        n = self%count
    end function space_n_variables

    pure integer function space_n_coordinates(self) result(n)
        class(fortbo_space_t), intent(in) :: self

        n = self%n_normalized
    end function space_n_coordinates

    pure integer function space_variable_index(self, name) result(index)
        class(fortbo_space_t), intent(in) :: self
        character(len=*), intent(in) :: name
        integer :: i

        index = 0
        do i = 1, self%count
            if (trim(self%variables(i)%name) == trim(name)) then
                index = i
                return
            end if
        end do
    end function space_variable_index

    !! User form to unit form.
    subroutine space_to_unit(self, values, unit, status)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(in) :: values(:)
        real(dp), intent(out) :: unit(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, category
        real(dp) :: lower, upper, value

        call check_shapes(self, size(values), size(unit), status)
        if (status%code /= FORTNUM_OK) return
        unit = 0.0_dp
        do i = 1, self%count
            associate (variable => self%variables(i))
                value = values(i)
                select case (variable%kind)
                case (FORTBO_VAR_CONTINUOUS)
                    if (value < variable%lower .or. value > variable%upper) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo space: continuous value out of bounds")
                        return
                    end if
                    if (variable%log_scale) then
                        lower = log(variable%lower)
                        upper = log(variable%upper)
                        unit(variable%offset + 1) = (log(value) - lower)/(upper - lower)
                    else
                        unit(variable%offset + 1) = (value - variable%lower) &
                            /(variable%upper - variable%lower)
                    end if
                case (FORTBO_VAR_INTEGER)
                    if (value < variable%lower .or. value > variable%upper) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo space: integer value out of bounds")
                        return
                    end if
                    if (abs(value - nint(value)) > 1.0e-9_dp) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo space: integer value is not integral")
                        return
                    end if
                    unit(variable%offset + 1) = (value - variable%lower) &
                        /(variable%upper - variable%lower)
                case (FORTBO_VAR_CATEGORICAL)
                    category = nint(value)
                    if (category < 1 .or. category > variable%n_categories) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "fortbo space: category index out of range")
                        return
                    end if
                    unit(variable%offset + category) = 1.0_dp
                end select
            end associate
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_to_unit

    !! Unit form to user form. Integers round to the nearest representable
    !! value and categoricals decode by arg-max with the lowest index winning a
    !! tie, so decoding is deterministic and replayable.
    subroutine space_from_unit(self, unit, values, status)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(in) :: unit(:)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, best
        real(dp) :: lower, upper, coordinate, best_value

        call check_shapes(self, size(values), size(unit), status)
        if (status%code /= FORTNUM_OK) return
        values = 0.0_dp
        do i = 1, self%count
            associate (variable => self%variables(i))
                select case (variable%kind)
                case (FORTBO_VAR_CONTINUOUS)
                    coordinate = min(max(unit(variable%offset + 1), 0.0_dp), 1.0_dp)
                    if (variable%log_scale) then
                        lower = log(variable%lower)
                        upper = log(variable%upper)
                        values(i) = exp(lower + coordinate*(upper - lower))
                    else
                        values(i) = variable%lower &
                            + coordinate*(variable%upper - variable%lower)
                    end if
                case (FORTBO_VAR_INTEGER)
                    coordinate = min(max(unit(variable%offset + 1), 0.0_dp), 1.0_dp)
                    values(i) = real(nint(variable%lower + coordinate &
                        *(variable%upper - variable%lower)), dp)
                case (FORTBO_VAR_CATEGORICAL)
                    best = 1
                    best_value = unit(variable%offset + 1)
                    do j = 2, variable%n_categories
                        if (unit(variable%offset + j) > best_value) then
                            best_value = unit(variable%offset + j)
                            best = j
                        end if
                    end do
                    values(i) = real(best, dp)
                end select
            end associate
        end do
        call self%apply_defaults(values, status)
    end subroutine space_from_unit

    !! Overwrite inactive conditional variables with their declared defaults.
    !! Two points that differ only in an inactive coordinate must map to the
    !! same user point, or the surrogate sees phantom variation.
    subroutine space_apply_defaults(self, values, status)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(inout) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        logical, allocatable :: active(:)
        integer :: i

        allocate (active(self%count))
        call space_active_mask(self, values, active, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%count
            if (.not. active(i)) values(i) = self%variables(i)%default_value
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_apply_defaults

    !! A variable is active when it has no parent, or when its parent is itself
    !! active and takes the activating category. Parents precede children, so a
    !! single forward pass resolves chains of conditions.
    subroutine space_active_mask(self, values, mask, status)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(in) :: values(:)
        logical, intent(out) :: mask(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, parent

        if (size(values) /= self%count .or. size(mask) /= self%count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: active mask width does not match")
            return
        end if
        do i = 1, self%count
            parent = self%variables(i)%parent
            if (parent == 0) then
                mask(i) = .true.
            else
                mask(i) = mask(parent) .and. &
                    nint(values(parent)) == self%variables(i)%parent_category
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_active_mask

    !! Mark the normalized coordinates on which a derivative is meaningful.
    !! Integer and categorical coordinates are false: their relaxation is a
    !! search device, not a differentiable reparameterization of the objective.
    subroutine space_differentiable_mask(self, mask, status)
        class(fortbo_space_t), intent(in) :: self
        logical, intent(out) :: mask(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (size(mask) /= self%n_normalized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: differentiable mask width does not match")
            return
        end if
        mask = .false.
        do i = 1, self%count
            associate (variable => self%variables(i))
                do j = 1, variable%width
                    mask(variable%offset + j) = variable%kind == FORTBO_VAR_CONTINUOUS
                end do
            end associate
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine space_differentiable_mask

    pure subroutine space_clamp(self, unit)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(inout) :: unit(:)
        integer :: i

        do i = 1, min(size(unit), self%n_normalized)
            unit(i) = min(max(unit(i), 0.0_dp), 1.0_dp)
        end do
    end subroutine space_clamp

    pure logical function space_contains_point(self, unit) result(inside)
        class(fortbo_space_t), intent(in) :: self
        real(dp), intent(in) :: unit(:)

        inside = .false.
        if (size(unit) /= self%n_normalized) return
        inside = all(unit >= 0.0_dp) .and. all(unit <= 1.0_dp)
    end function space_contains_point

    pure subroutine check_shapes(self, n_values, n_unit, status)
        class(fortbo_space_t), intent(in) :: self
        integer, intent(in) :: n_values
        integer, intent(in) :: n_unit
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%finalized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: space is not finalized")
            return
        end if
        if (n_values /= self%count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: user vector width does not match")
            return
        end if
        if (n_unit /= self%n_normalized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: unit vector width does not match")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_shapes

    subroutine append(self, variable, status)
        class(fortbo_space_t), intent(inout) :: self
        type(fortbo_variable_t), intent(in) :: variable
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_variable_t), allocatable :: grown(:)
        integer :: capacity

        if (self%finalized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo space: cannot add to a finalized space")
            return
        end if
        if (.not. allocated(self%variables)) then
            allocate (self%variables(INITIAL_CAPACITY))
        end if
        capacity = size(self%variables)
        if (self%count >= capacity) then
            allocate (grown(2*capacity))
            grown(1:self%count) = self%variables(1:self%count)
            call move_alloc(grown, self%variables)
        end if
        self%count = self%count + 1
        self%variables(self%count) = variable
        call status_set(status, FORTNUM_OK, "")
    end subroutine append

end module fortbo_space
