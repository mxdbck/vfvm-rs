use vfvm_rs::models::pn::pn::{pn_problem_def, PnJunctionModel, create_pn_initial_condition, create_pn_bc_registry};
use vfvm_rs::system::{System, InitialCondition, Geometry, SolverConfig};
use vfvm_rs::numerics::timing::{reset_timing, finalize_and_print};
use std::time::Instant;

fn main() {
    env_logger::init();

    let (mesh, params) = pn_problem_def(1.0, 2000); // Large mesh to stress the solver
    let model = PnJunctionModel::new(params.clone(), 0.0).with_mesh(&mesh);
    let bc_registry = create_pn_bc_registry(&params, 0.0);
    let initial_guess = create_pn_initial_condition(&mesh, &params, 0.0);

    let mut system = System::new(
        model.functional,
        Geometry { mesh },
        SolverConfig {
            max_iterations: 10, // Force a specific number of iterations
            tolerance: 1e-12,   // Make it strict so it does the work
            ..SolverConfig::default()
        },
        InitialCondition::Vector(initial_guess),
        bc_registry,
    );

    system.output_config = None;

    println!("Running benchmark...");
    reset_timing();
    let start = Instant::now();

    let _ = system.solve();

    let total_time = start.elapsed();

    finalize_and_print(total_time);
}
