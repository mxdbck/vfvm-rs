mod discretization;
mod models;
mod numerics;
mod physics;
mod processing;
mod system;

use crate::models::pn::pn::{
    PnJunctionModel, create_pn_bc_registry, create_pn_initial_condition, tag_pn_boundary_faces,
};
use crate::processing::summary::SimulationSummary;
use crate::system::{Geometry, InitialCondition, OutputConfig, SolverConfig, System};
use log::{error, info};
use std::path::PathBuf;

fn main() {
    env_logger::builder().format_timestamp(None).init();
    let (mesh, params) = models::pn::pn::pn_problem_def(1.0, 1000);

    let v_applied = 1e-4;
    let mut model = PnJunctionModel::new(params.clone(), v_applied).with_mesh(&mesh);

    // Manual setup of BCs and ICs
    let bc_registry = create_pn_bc_registry(&params, v_applied);
    tag_pn_boundary_faces(&mesh, &mut model.functional.face_tags);

    let initial_guess = create_pn_initial_condition(&mesh, &params, v_applied);

    let mut solver_config = SolverConfig::default();
    solver_config.max_step = Some(2.0);

    // Build the system
    let mut system = System::new(
        model.functional,
        Geometry { mesh },
        solver_config,
        InitialCondition::Vector(initial_guess),
        bc_registry,
    );

    system.length_scale = params.l_scale;
    system.state_scales = vec![params.v_scale; 3];

    system.output_config = Some(OutputConfig {
        dir: PathBuf::from("output/main"),
        file_pattern: "solution.csv".to_string(),
        save_initial: true,
        save_final: true,
        save_transient_interval: None,
    });

    system.summary_handler = Some(Box::new(|system, result| {
        info!("Sparse solver finished successfully.");
        let mut summary = SimulationSummary::new(
            &system.geometry.mesh,
            &system.physics.data,
            &system.physics.tolerances,
        );

        summary.add_sparse_solver_info(result.iterations, result.final_residual);

        summary
            .write_to_file("output/main/simulation_summary.txt")
            .expect("Failed to write summary");
        summary.print_to_console();

        info!("Summary saved to output/main/simulation_summary.txt");
    }));

    // Solve using System abstraction
    if let Err(e) = system.solve() {
        error!("Sparse solver failed: {}", e);
    }
}
