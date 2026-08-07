program test_history
    !! BO0: observation history, policies, and checkpoint/resume.
    !!
    !! The oracles here are behavioral and independent of the storage layout:
    !!   * the incumbent is checked against a brute-force scan written in the
    !!     test over the values the test itself inserted;
    !!   * duplicate and missing-observation policies are checked by their
    !!     observable consequences (row counts, refusals, replaced values), not
    !!     by inspecting the flags the implementation set;
    !!   * checkpoint/resume is checked by round-tripping and then re-deriving
    !!     the incumbent, feasibility, cost, and gradient set from the restored
    !!     history, so a checkpoint that silently drops a field is caught.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortbo_history, only: fortbo_history_t, FORTBO_MISSING_REFUSE, &
        FORTBO_MISSING_SKIP, FORTBO_MISSING_IMPUTE_WORST, FORTBO_DUPLICATE_ALLOW, &
        FORTBO_DUPLICATE_REJECT, FORTBO_DUPLICATE_REPLACE, FORTBO_OUTCOME_OK, &
        FORTBO_OUTCOME_PENDING, FORTBO_OUTCOME_FAILED
    implicit none

    integer :: failures

    failures = 0
    call check_basic_bookkeeping(failures)
    call check_missing_policies(failures)
    call check_duplicate_policies(failures)
    call check_constraints_and_incumbent(failures)
    call check_pending_and_completion(failures)
    call check_gradient_observations(failures)
    call check_growth_preserves_order(failures)
    call check_checkpoint_round_trip(failures)

    if (failures == 0) then
        print *, "test_history: PASS"
    else
        print *, "test_history: FAIL", failures
        error stop 1
    end if

contains

    subroutine check_basic_bookkeeping(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status

        call history%initialize(2, 0, status)
        call expect(status%code == FORTNUM_OK, "initialize succeeds", failures)
        call history%add([0.0_dp, 1.0_dp], status, objective=3.0_dp, cost=2.0_dp)
        call history%add([1.0_dp, 1.0_dp], status, objective=1.0_dp, cost=5.0_dp)
        call expect(history%count == 2, "two rows are stored", failures)
        call expect(history%usable_count() == 2, "both rows are usable", failures)
        call expect(history%pending_count() == 0, "no rows are pending", failures)
        call expect_close(history%total_cost(), 7.0_dp, 1.0e-14_dp, &
                          "costs accumulate", failures)
        call expect(history%best_index() == 2, "the incumbent is the smaller value", &
                    failures)

        call history%initialize(0, 0, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "a zero-width space is refused", failures)
    end subroutine check_basic_bookkeeping

    !! A failed evaluation must not become a number nobody measured.
    subroutine check_missing_policies(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status

        call history%initialize(1, 0, status)
        history%missing_policy = FORTBO_MISSING_REFUSE
        call history%add([0.5_dp], status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
                    "the refuse policy refuses a missing objective", failures)
        call expect(history%count == 0, "a refused row is not stored", failures)

        history%missing_policy = FORTBO_MISSING_SKIP
        call history%add([0.5_dp], status)
        call expect(status%code == FORTNUM_OK, "the skip policy succeeds", failures)
        call expect(history%count == 0, "a skipped row is not stored", failures)

        history%missing_policy = FORTBO_MISSING_IMPUTE_WORST
        call history%add([0.5_dp], status)
        call expect(status%code == FORTNUM_OK, "the impute policy succeeds", failures)
        call expect(history%count == 1, "an imputed row is stored", failures)
        call expect(history%usable_count() == 0, &
                    "an imputed row is never usable training data", failures)
        call expect(history%best_index() == 0, &
                    "an imputed row cannot become the incumbent", failures)
        call expect(history%outcome(1) == FORTBO_OUTCOME_FAILED, &
                    "an imputed row records the failure", failures)
    end subroutine check_missing_policies

    subroutine check_duplicate_policies(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status

        call history%initialize(1, 0, status)
        history%duplicate_tolerance = 1.0e-9_dp

        history%duplicate_policy = FORTBO_DUPLICATE_ALLOW
        call history%add([0.25_dp], status, objective=1.0_dp)
        call history%add([0.25_dp], status, objective=2.0_dp)
        call expect(history%count == 2, "the allow policy keeps both rows", failures)

        history%duplicate_policy = FORTBO_DUPLICATE_REJECT
        call history%add([0.25_dp], status, objective=3.0_dp)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "the reject policy refuses a repeat", failures)
        call expect(history%count == 2, "a rejected repeat is not stored", failures)

        history%duplicate_policy = FORTBO_DUPLICATE_REPLACE
        call history%add([0.25_dp], status, objective=-4.0_dp)
        call expect(history%count == 2, "the replace policy adds no row", failures)
        call expect_close(history%objectives(1), -4.0_dp, 1.0e-14_dp, &
                          "the replace policy overwrites the first match", failures)

        call history%add([0.25_dp + 1.0e-12_dp], status, objective=-5.0_dp)
        call expect(history%count == 2, &
                    "a point inside the tolerance counts as a duplicate", failures)
        call history%add([0.25_dp + 1.0e-3_dp], status, objective=-6.0_dp)
        call expect(history%count == 3, &
                    "a point outside the tolerance is a new row", failures)
    end subroutine check_duplicate_policies

    !! Brute-force incumbent oracle: the test scans its own inserted values
    !! under the feasibility rule and compares against the history's answer.
    subroutine check_constraints_and_incumbent(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status
        real(dp) :: inputs(4), objectives(4), constraint(4)
        real(dp) :: incumbent(1), incumbent_value, expected_value
        integer :: i, expected_index

        inputs = [0.1_dp, 0.4_dp, 0.7_dp, 0.9_dp]
        objectives = [5.0_dp, -3.0_dp, 1.0_dp, -8.0_dp]
        constraint = [-1.0_dp, 2.0_dp, -0.5_dp, 0.25_dp]

        call history%initialize(1, 1, status)
        do i = 1, 4
            call history%add([inputs(i)], status, objective=objectives(i), &
                             constraints=[constraint(i)])
        end do

        expected_index = 0
        expected_value = huge(1.0_dp)
        do i = 1, 4
            if (constraint(i) > 0.0_dp) cycle
            if (objectives(i) < expected_value) then
                expected_value = objectives(i)
                expected_index = i
            end if
        end do

        call expect(history%best_index() == expected_index, &
                    "the incumbent matches a brute-force feasible scan", failures)
        call expect(history%is_feasible(2) .eqv. .false., &
                    "a violated constraint is infeasible", failures)
        call expect(history%is_feasible(3), "a satisfied constraint is feasible", &
                    failures)
        call history%incumbent(incumbent, incumbent_value, status)
        call expect(status%code == FORTNUM_OK, "the incumbent is reported", failures)
        call expect_close(incumbent_value, expected_value, 1.0e-14_dp, &
                          "the incumbent value is the best feasible value", failures)
        call expect_close(incumbent(1), inputs(expected_index), 1.0e-14_dp, &
                          "the incumbent input is the matching row", failures)
    end subroutine check_constraints_and_incumbent

    !! An unmeasured constraint must not be read as a satisfied one, and a
    !! pending row must never reach a surrogate.
    subroutine check_pending_and_completion(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status
        real(dp), allocatable :: train_inputs(:, :), train_objectives(:)
        integer :: row

        call history%initialize(1, 1, status)
        call history%add_pending([0.3_dp], row, status)
        call expect(status%code == FORTNUM_OK, "a pending row is registered", failures)
        call expect(row == 1, "the pending row index is returned", failures)
        call expect(history%pending_count() == 1, "the pending row is counted", &
                    failures)
        call expect(history%usable_count() == 0, "a pending row is not usable", &
                    failures)
        call expect(.not. history%is_feasible(1), &
                    "an unmeasured constraint is not feasible", failures)

        call history%training_data(train_inputs, train_objectives, status)
        call expect(size(train_objectives) == 0, &
                    "a pending row is excluded from training data", failures)

        call history%complete(row, status, objective=2.5_dp, constraints=[-1.0_dp])
        call expect(status%code == FORTNUM_OK, "completion succeeds", failures)
        call expect(history%usable_count() == 1, "a completed row is usable", failures)
        call expect(history%is_feasible(1), "the measured constraint is feasible", &
                    failures)
        call expect(history%outcome(1) == FORTBO_OUTCOME_OK, &
                    "a completed row records success", failures)

        call history%complete(row, status, objective=9.0_dp)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "completing a row twice is refused", failures)
        call expect_close(history%objectives(1), 2.5_dp, 1.0e-14_dp, &
                          "the first answer is not overwritten", failures)

        call history%complete(99, status, objective=1.0_dp)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "an out-of-range completion is refused", failures)
    end subroutine check_pending_and_completion

    !! Derivative observations are ordinary history rows: any surrogate may
    !! consume them, and a partially measured gradient does not masquerade as a
    !! complete one.
    subroutine check_gradient_observations(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status
        real(dp), allocatable :: inputs(:, :), objectives(:), gradients(:, :)
        integer :: row

        call history%initialize(2, 0, status)
        call history%add([0.0_dp, 0.0_dp], status, objective=1.0_dp)
        call history%add([1.0_dp, 0.0_dp], status, objective=2.0_dp, &
                         gradient=[3.0_dp, -4.0_dp])
        call history%add([0.0_dp, 1.0_dp], status, objective=3.0_dp, &
                         gradient=[7.0_dp, 8.0_dp], &
                         gradient_mask=[.true., .false.])
        call expect(status%code == FORTNUM_OK, "a masked gradient is accepted", &
                    failures)

        call expect(.not. history%has_gradient(1), "a row without a gradient", failures)
        call expect(history%has_gradient(2), "a fully measured gradient", failures)
        call expect(.not. history%has_gradient(3), &
                    "a partially measured gradient is not a complete gradient", failures)
        call expect(history%gradient_present(3, 1), "the measured component is kept", &
                    failures)
        call expect(.not. history%gradient_present(3, 2), &
                    "the unmeasured component is flagged absent", failures)
        call expect_close(history%gradients(3, 2), 0.0_dp, 1.0e-14_dp, &
                          "an unmeasured component is not silently stored", failures)

        call expect(history%gradient_count() == 1, "one complete gradient row", &
                    failures)
        call history%gradient_data(inputs, objectives, gradients, status)
        call expect(size(objectives) == 1, "gradient data packs only complete rows", &
                    failures)
        call expect_close(gradients(1, 1), 3.0_dp, 1.0e-14_dp, &
                          "the packed gradient is the measured one", failures)
        call expect_close(gradients(1, 2), -4.0_dp, 1.0e-14_dp, &
                          "the packed gradient keeps its second component", failures)
        call expect_close(objectives(1), 2.0_dp, 1.0e-14_dp, &
                          "the packed objective matches its row", failures)

        call history%add([2.0_dp, 2.0_dp], status, objective=4.0_dp, &
                         gradient=[1.0_dp])
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "a mis-shaped gradient is refused", failures)

        call history%add_pending([3.0_dp, 3.0_dp], row, status)
        call history%complete(row, status, objective=5.0_dp, &
                              gradient=[-1.0_dp, -2.0_dp])
        call expect(history%has_gradient(row), &
                    "a gradient may arrive with the completion", failures)
    end subroutine check_gradient_observations

    !! Reallocation must not reorder or lose rows; insertion order is the
    !! replay order.
    subroutine check_growth_preserves_order(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: history
        type(fortnum_status_t) :: status
        integer, parameter :: n = 200
        integer :: i
        logical :: ordered

        call history%initialize(1, 1, status)
        do i = 1, n
            call history%add([real(i, dp)], status, objective=real(n - i, dp), &
                             cost=1.0_dp, constraints=[-1.0_dp], &
                             gradient=[real(2*i, dp)])
        end do
        call expect(history%count == n, "every row survives growth", failures)

        ordered = .true.
        do i = 1, n
            if (abs(history%inputs(i, 1) - real(i, dp)) > 0.0_dp) ordered = .false.
            if (abs(history%objectives(i) - real(n - i, dp)) > 0.0_dp) ordered = .false.
            if (abs(history%gradients(i, 1) - real(2*i, dp)) > 0.0_dp) ordered = .false.
            if (.not. history%constraint_present(i, 1)) ordered = .false.
        end do
        call expect(ordered, "growth preserves row order and every field", failures)
        call expect(history%best_index() == n, "the incumbent survives growth", &
                    failures)
        call expect(history%gradient_count() == n, "gradients survive growth", failures)
    end subroutine check_growth_preserves_order

    !! A checkpoint is only useful if the restored run behaves like the original.
    subroutine check_checkpoint_round_trip(failures)
        integer, intent(inout) :: failures
        type(fortbo_history_t) :: original, restored
        type(fortnum_status_t) :: status
        character(len=*), parameter :: path = "fortbo_history_checkpoint.txt"
        character(len=*), parameter :: bad_path = "fortbo_history_bad.txt"
        real(dp) :: incumbent_a(2), incumbent_b(2), value_a, value_b
        integer :: unit, i
        logical :: identical

        call original%initialize(2, 1, status)
        original%missing_policy = FORTBO_MISSING_IMPUTE_WORST
        original%duplicate_policy = FORTBO_DUPLICATE_REJECT
        original%duplicate_tolerance = 1.0e-7_dp
        call original%add([0.1_dp, 0.2_dp], status, objective=4.0_dp, cost=1.5_dp, &
                          constraints=[-1.0_dp], gradient=[0.5_dp, -0.5_dp])
        call original%add([0.3_dp, 0.4_dp], status, objective=-2.0_dp, cost=2.5_dp, &
                          constraints=[0.5_dp])
        call original%add([0.5_dp, 0.6_dp], status, objective=-1.0_dp, cost=0.5_dp, &
                          constraints=[-0.25_dp], gradient=[1.0_dp, 2.0_dp], &
                          gradient_mask=[.true., .false.])
        call original%checkpoint_write(path, status)
        call expect(status%code == FORTNUM_OK, "checkpoint writing succeeds", failures)

        call restored%checkpoint_read(path, status)
        call expect(status%code == FORTNUM_OK, "checkpoint reading succeeds", failures)
        call expect(restored%count == original%count, "the row count survives", &
                    failures)
        call expect(restored%missing_policy == original%missing_policy, &
                    "the missing policy survives", failures)
        call expect(restored%duplicate_policy == original%duplicate_policy, &
                    "the duplicate policy survives", failures)
        call expect_close(restored%duplicate_tolerance, original%duplicate_tolerance, &
                          1.0e-14_dp, "the duplicate tolerance survives", failures)

        identical = .true.
        do i = 1, original%count
            if (maxval(abs(restored%inputs(i, :) - original%inputs(i, :))) > 1.0e-12_dp) &
                identical = .false.
            if (abs(restored%objectives(i) - original%objectives(i)) > 1.0e-12_dp) &
                identical = .false.
            if (abs(restored%costs(i) - original%costs(i)) > 1.0e-12_dp) &
                identical = .false.
            if (restored%outcome(i) /= original%outcome(i)) identical = .false.
            if (restored%has_gradient(i) .neqv. original%has_gradient(i)) &
                identical = .false.
            if (restored%is_feasible(i) .neqv. original%is_feasible(i)) &
                identical = .false.
        end do
        call expect(identical, "every restored row matches the original", failures)

        call original%incumbent(incumbent_a, value_a, status)
        call restored%incumbent(incumbent_b, value_b, status)
        call expect_close(value_b, value_a, 1.0e-12_dp, &
                          "the restored incumbent value matches", failures)
        call expect(maxval(abs(incumbent_b - incumbent_a)) < 1.0e-12_dp, &
                    "the restored incumbent input matches", failures)
        call expect_close(restored%total_cost(), original%total_cost(), 1.0e-12_dp, &
                          "the restored total cost matches", failures)
        call expect(restored%gradient_count() == original%gradient_count(), &
                    "the restored gradient set matches", failures)

        open (newunit=unit, file=bad_path, status="replace", action="write")
        write (unit, *) FORTBO_HISTORY_FORMAT_VERSION_PLUS_ONE()
        write (unit, *) 1, 0, 0
        close (unit)
        call restored%checkpoint_read(bad_path, status)
        call expect(status%code == FORTNUM_NOT_IMPLEMENTED, &
                    "an unknown checkpoint version is refused", failures)

        call restored%checkpoint_read("fortbo_no_such_checkpoint.txt", status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
                    "a missing checkpoint is refused", failures)

        open (newunit=unit, file=path, status="old")
        close (unit, status="delete")
        open (newunit=unit, file=bad_path, status="old")
        close (unit, status="delete")
    end subroutine check_checkpoint_round_trip

    integer function FORTBO_HISTORY_FORMAT_VERSION_PLUS_ONE() result(version)
        use fortbo_history, only: FORTBO_HISTORY_FORMAT_VERSION

        version = FORTBO_HISTORY_FORMAT_VERSION + 1
    end function FORTBO_HISTORY_FORMAT_VERSION_PLUS_ONE

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

    subroutine expect_close(actual, expected, tolerance, description, failures)
        real(dp), intent(in) :: actual
        real(dp), intent(in) :: expected
        real(dp), intent(in) :: tolerance
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (abs(actual - expected) > tolerance) then
            failures = failures + 1
            print *, "  FAIL: ", description, actual, expected
        end if
    end subroutine expect_close

end program test_history
