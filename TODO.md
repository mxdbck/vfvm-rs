# TODO

- [x] Quickly plot somethings with `plotly`
- [x] Look over functional.rs, semiconductor.rs, physics/mod.rs.
- [x] Check solver code
- [x] Implement auto-differentiation
- [x] Implement boundary conditions
- [x] Actually build and solve the system.

- [x] Cleanup code.
- [x] Refactor calculate_residual into flux and reaction contributions.
- [x] Include error.rs with thiserror.rs
- [x] Make the code general for 3d again.
- [x] Implement a sparse jacobian and system solver.
- [x] Add an actually half decent boundary condition implementation
- [x] Add optional logging to solver and configuration functions for cleaner benchmarking.
- [x] Add benchmarking for dense and sparse jacobian assembly.
- [x] Remove some cloning for dual number operations.
- [x] Go through code and simplify where possible.
- [x] Implement less smooth-brained newton-raphson or alternative.
- [x] Make a simple electrostatics example.
- [x] Implement transient simulations.

- [x] Improve Logging
- [x] Benchmark
- [x] Parallize the trivially parallelizable
- [x] Implement all boundary condition types
- [ ] Bump dependencies
- [ ] Fix preconditionning
- [ ] Compatbility with ParaView
- [ ] consider abstracting the linear solver step to optionally use direct sparse solvers (UMFPACK, MUMPS ...) for highly nonlinear, coupled, or ill-conditioned systems.

- [ ] Make documents that explains the code and more complex bits. 
- [ ] Look into Schaffeter-Gummel from WIAS paper
- [ ] Look into adaptive time stepping.
- [ ] Consider adding a method to validate that all boundary faces have associated BCs.

- [ ] Implement more complex domain cell spacing.

- [ ] Remove the need for #[allow(unused)]

### Examples to Implement
- [ ] Heat conduction (Idea : copper on silicon with bottom heating and convective cooling on top)
- [ ] Transient Diffusion (Idea : anisotropic diffusion)
- [ ] Convection-Diffusion (Idea : rotating plume)
- [x] Reaction-Diffusion (Free quantum particle)
