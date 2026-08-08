module fortbo_generated
    !! FortSym leaves exposed as module procedures.
    !!
    !! The emitted files remain in `src/generated` so they can be regenerated
    !! independently. Including them in this module gives the Fortran module
    !! dependency graph an explicit edge from each consumer to its generated
    !! implementation. Without that edge a static archive can omit the leaf
    !! from a target that reaches it only through an external procedure
    !! interface, which makes otherwise valid executables fail at link time.

    implicit none
    private

    public :: fortbo_generated_acquisition_leaf
    public :: fortbo_generated_preference_leaf
    public :: fortbo_generated_trust_region_leaf
    public :: fortbo_generated_posterior_moment_leaf

contains

    include "generated/fortbo_generated_acquisition_leaf.f90"
    include "generated/fortbo_generated_preference_leaf.f90"
    include "generated/fortbo_generated_trust_region_leaf.f90"
    include "generated/fortbo_generated_posterior_moment_leaf.f90"

end module fortbo_generated
