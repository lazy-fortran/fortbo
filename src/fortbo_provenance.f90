module fortbo_provenance
    !! Benchmark rows with their execution lane and provenance (ROADMAP BO6).
    !!
    !! A benchmark number without provenance is not evidence, it is a rumour.
    !! Which commit, which compiler, which device, which precision — and, above
    !! all, **which lane** — decide whether two numbers can be compared at all.
    !!
    !! Four lanes, kept separate because mixing them makes a comparison
    !! meaningless in a way no amount of averaging repairs:
    !!
    !!   * **CPU** — host only, the correctness reference;
    !!   * **transfer-inclusive GPU** — device compute plus every host round
    !!     trip it needed. This is what a user actually experiences, and it is
    !!     the number most often quietly omitted;
    !!   * **resident GPU** — device compute with the data already there. A
    !!     legitimate number that answers a *different* question, and reporting
    !!     it as though it were the previous one is the single most common way a
    !!     GPU claim misleads;
    !!   * **refusal** — the operation was declined, by name. A refusal is a
    !!     result and belongs in the table. Dropping refused rows is what turns
    !!     "this works on four of nine configurations" into "this works".
    !!
    !! The lane is recorded, never inferred. Guessing it from a timing would be
    !! backwards: the timing is what the lane explains.
    !!
    !! A refusal row carries no timing at all rather than a zero. Zero is a
    !! number, and a mean taken over a table containing it is wrong in a way
    !! nobody can see afterwards.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: FORTBO_LANE_CPU = 1
    integer, parameter, public :: FORTBO_LANE_GPU_TRANSFER = 2
    integer, parameter, public :: FORTBO_LANE_GPU_RESIDENT = 3
    integer, parameter, public :: FORTBO_LANE_REFUSED = 4

    integer, parameter :: TEXT_LEN = 96
    integer, parameter :: INITIAL_CAPACITY = 32

    public :: fortbo_lane_name
    public :: fortbo_provenance_row_t
    public :: fortbo_provenance_table_t

    !! One measurement, or one refusal.
    type :: fortbo_provenance_row_t
        integer :: lane = FORTBO_LANE_CPU
        character(len=TEXT_LEN) :: case_name = ""
        !! Source revision, toolchain, and device. All three, because a number
        !! is only reproducible if all three are known.
        character(len=TEXT_LEN) :: source_revision = ""
        character(len=TEXT_LEN) :: toolchain = ""
        character(len=TEXT_LEN) :: device = ""
        character(len=TEXT_LEN) :: precision = "float64"
        !! Present only for measured lanes.
        real(dp) :: wall_seconds = 0.0_dp
        logical :: has_timing = .false.
        !! Present only for a refusal, and required there.
        character(len=TEXT_LEN) :: refusal_reason = ""
        integer :: seed = 0
    end type fortbo_provenance_row_t

    type :: fortbo_provenance_table_t
        integer :: count = 0
        type(fortbo_provenance_row_t), allocatable :: rows(:)
    contains
        procedure, public :: initialize => table_initialize
        procedure, public :: record => table_record
        procedure, public :: refuse => table_refuse
        procedure, public :: lane_count => table_lane_count
        procedure, public :: lane_mean_seconds => table_lane_mean_seconds
        procedure, public :: comparable => table_comparable
    end type fortbo_provenance_table_t

contains

    pure function fortbo_lane_name(lane) result(name)
        integer, intent(in) :: lane
        character(len=:), allocatable :: name

        select case (lane)
        case (FORTBO_LANE_CPU)
            name = "cpu"
        case (FORTBO_LANE_GPU_TRANSFER)
            name = "gpu_transfer_inclusive"
        case (FORTBO_LANE_GPU_RESIDENT)
            name = "gpu_resident"
        case (FORTBO_LANE_REFUSED)
            name = "refused"
        case default
            name = "unknown"
        end select
    end function fortbo_lane_name

    subroutine table_initialize(self, status)
        class(fortbo_provenance_table_t), intent(out) :: self
        type(fortnum_status_t), intent(out) :: status

        allocate (self%rows(INITIAL_CAPACITY))
        self%count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine table_initialize

    !! Record a measured row. Provenance is mandatory: a timing whose source
    !! revision, toolchain, or device is unknown cannot be reproduced, and a
    !! number that cannot be reproduced should not be in a table that will be
    !! read as evidence.
    subroutine table_record(self, lane, case_name, source_revision, toolchain, &
            device, wall_seconds, status, seed, precision)
        class(fortbo_provenance_table_t), intent(inout) :: self
        integer, intent(in) :: lane
        character(len=*), intent(in) :: case_name
        character(len=*), intent(in) :: source_revision
        character(len=*), intent(in) :: toolchain
        character(len=*), intent(in) :: device
        real(dp), intent(in) :: wall_seconds
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: seed
        character(len=*), intent(in), optional :: precision
        type(fortbo_provenance_row_t) :: row

        if (.not. allocated(self%rows)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: table is not initialized")
            return
        end if
        if (lane == FORTBO_LANE_REFUSED) then
            ! A refusal has no timing, so it must go through `refuse`, which
            ! requires the reason a measured row has no field for.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: a refusal is recorded through refuse")
            return
        end if
        if (fortbo_lane_name(lane) == "unknown") then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: unknown lane")
            return
        end if
        if (len_trim(case_name) == 0 .or. len_trim(source_revision) == 0 .or. &
            len_trim(toolchain) == 0 .or. len_trim(device) == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: case, revision, toolchain, and device are required")
            return
        end if
        if (wall_seconds < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: wall time must not be negative")
            return
        end if

        row%lane = lane
        row%case_name = case_name
        row%source_revision = source_revision
        row%toolchain = toolchain
        row%device = device
        row%wall_seconds = wall_seconds
        row%has_timing = .true.
        if (present(seed)) row%seed = seed
        if (present(precision)) row%precision = precision
        call append(self, row)
        call status_set(status, FORTNUM_OK, "")
    end subroutine table_record

    !! Record a refusal, which is a result rather than an absence.
    !!
    !! Carries no timing at all — not a zero. A zero would be averaged with the
    !! measured rows and would drag every summary toward it, invisibly.
    subroutine table_refuse(self, case_name, source_revision, toolchain, device, &
            reason, status)
        class(fortbo_provenance_table_t), intent(inout) :: self
        character(len=*), intent(in) :: case_name
        character(len=*), intent(in) :: source_revision
        character(len=*), intent(in) :: toolchain
        character(len=*), intent(in) :: device
        character(len=*), intent(in) :: reason
        type(fortnum_status_t), intent(out) :: status
        type(fortbo_provenance_row_t) :: row

        if (.not. allocated(self%rows)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: table is not initialized")
            return
        end if
        if (len_trim(reason) == 0) then
            ! A refusal without a reason is indistinguishable from a row nobody
            ! got round to filling in.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: a refusal must say why")
            return
        end if
        if (len_trim(case_name) == 0 .or. len_trim(source_revision) == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "fortbo provenance: a refusal still needs its case and revision")
            return
        end if

        row%lane = FORTBO_LANE_REFUSED
        row%case_name = case_name
        row%source_revision = source_revision
        row%toolchain = toolchain
        row%device = device
        row%refusal_reason = reason
        row%has_timing = .false.
        call append(self, row)
        call status_set(status, FORTNUM_OK, "")
    end subroutine table_refuse

    subroutine append(self, row)
        class(fortbo_provenance_table_t), intent(inout) :: self
        type(fortbo_provenance_row_t), intent(in) :: row
        type(fortbo_provenance_row_t), allocatable :: grown(:)

        if (self%count == size(self%rows)) then
            allocate (grown(2*size(self%rows)))
            grown(:self%count) = self%rows(:self%count)
            call move_alloc(grown, self%rows)
        end if
        self%count = self%count + 1
        self%rows(self%count) = row
    end subroutine append

    pure integer function table_lane_count(self, lane) result(total)
        class(fortbo_provenance_table_t), intent(in) :: self
        integer, intent(in) :: lane
        integer :: i

        total = 0
        if (.not. allocated(self%rows)) return
        do i = 1, self%count
            if (self%rows(i)%lane == lane) total = total + 1
        end do
    end function table_lane_count

    !! Mean wall time within one lane, over rows that actually have a timing.
    !!
    !! `available` is false when the lane has no measured rows, rather than
    !! returning zero — the distinction between "fast" and "never ran" is the
    !! entire point of keeping refusals in the table.
    subroutine table_lane_mean_seconds(self, lane, mean, available)
        class(fortbo_provenance_table_t), intent(in) :: self
        integer, intent(in) :: lane
        real(dp), intent(out) :: mean
        logical, intent(out) :: available
        real(dp) :: total
        integer :: i, counted

        mean = 0.0_dp
        available = .false.
        if (.not. allocated(self%rows)) return
        total = 0.0_dp
        counted = 0
        do i = 1, self%count
            if (self%rows(i)%lane /= lane) cycle
            if (.not. self%rows(i)%has_timing) cycle
            total = total + self%rows(i)%wall_seconds
            counted = counted + 1
        end do
        if (counted == 0) return
        mean = total/real(counted, dp)
        available = .true.
    end subroutine table_lane_mean_seconds

    !! Whether two rows may be compared at all.
    !!
    !! Same case, same precision, same source revision. Different lanes *are*
    !! comparable — comparing lanes is the point — but a CPU row from one
    !! revision against a GPU row from another measures the revisions, not the
    !! lanes, and is the most common way a speedup is manufactured by accident.
    pure logical function table_comparable(self, first, second) result(allowed)
        class(fortbo_provenance_table_t), intent(in) :: self
        integer, intent(in) :: first, second

        allowed = .false.
        if (.not. allocated(self%rows)) return
        if (first < 1 .or. first > self%count) return
        if (second < 1 .or. second > self%count) return
        ! A refusal has nothing to compare.
        if (self%rows(first)%lane == FORTBO_LANE_REFUSED) return
        if (self%rows(second)%lane == FORTBO_LANE_REFUSED) return
        allowed = self%rows(first)%case_name == self%rows(second)%case_name &
            .and. self%rows(first)%precision == self%rows(second)%precision &
            .and. self%rows(first)%source_revision &
            == self%rows(second)%source_revision
    end function table_comparable

end module fortbo_provenance
