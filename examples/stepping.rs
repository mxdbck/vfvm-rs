use log::{error, info};
use std::path::PathBuf;
use vfvm_rs::models::pn::pn::{PnJunctionModel, create_pn_bc_registry, create_pn_initial_condition, tag_pn_boundary_faces};
use vfvm_rs::processing::csv_writer;
use vfvm_rs::system::{System, InitialCondition, Geometry, SolverConfig, OutputConfig};

fn main() {
    env_logger::init();
    let mesh_size = 1.0;
    let num_points = 500;

    let v_start: f64 = 0.0;
    let v_end: f64 = -1.0;
    let v_step: f64 = -0.05;

    info!("Voltage Stepping Simulation");
    info!("============================");
    info!("Start voltage: {:.3} V", v_start);
    info!("End voltage: {:.3} V", v_end);
    info!("Step size: {:.3} V", v_step);

    let (mesh, params) = vfvm_rs::models::pn::pn::pn_problem_def(mesh_size, num_points);

    let mut solver_config = SolverConfig::default();
    solver_config.max_step = Some(2.0);

    // Start with initial guess at equilibrium
    let mut current_solution = create_pn_initial_condition(&mesh, &params, 0.0);

    // Voltage stepping loop
    let num_steps = ((v_end - v_start) / v_step).round() as usize;
    let mut voltage_log: Vec<(f64, usize, f64)> = vec![];

    for step in 0..=num_steps {
        let voltage = v_start + step as f64 * v_step;

        info!(
            "Step {}/{}: Solving at V = {:.3} V",
            step,
            num_steps,
            voltage
        );

        let mut model = PnJunctionModel::new(params.clone(), voltage).with_mesh(&mesh);

        // Update voltage and reconfigure boundary conditions
        let bc_registry = create_pn_bc_registry(&params, voltage);
        tag_pn_boundary_faces(&mesh, &mut model.functional.face_tags);

        // Recreate config because it's consumed
        let step_config = SolverConfig {
            tolerance: solver_config.tolerance,
            max_iterations: solver_config.max_iterations,
            preconditioner: solver_config.preconditioner,
            history_handler: None,
            min_step_size: solver_config.min_step_size,
            armijo_param: solver_config.armijo_param,
            max_step: solver_config.max_step,
            forcing_term: solver_config.forcing_term,
        };

        let mut system = System::new(
            model.functional,
            Geometry { mesh: mesh.clone() },
            step_config,
            InitialCondition::Vector(current_solution.clone()),
            bc_registry,
        );

        system.output_config = Some(OutputConfig {
            dir: PathBuf::from("output/stepping"),
            file_pattern: "ignored.csv".to_string(),
            save_initial: false,
            save_final: false,
            save_transient_interval: None,
        });

        match system.solve() {
            Ok(result) => {
                info!(
                    "  Converged in {} iterations, residual: {:.3e}",
                    result.iterations,
                    result.final_residual
                );

                current_solution = result.solution.clone();
                voltage_log.push((
                    voltage,
                    result.iterations.try_into().unwrap(),
                    result.final_residual,
                ));

                // Save
                if let Some(cfg) = &mut system.output_config {
                     cfg.file_pattern = format!("step_{:03}_V_{:.3}.csv", step, voltage);
                }
                system.save_snapshot(step, voltage, &result.solution).expect("Failed to save snapshot");
            }
            Err(e) => {
                error!("  Failed to converge: {}", e);
                error!("  Stopping voltage stepping.");
                break;
            }
        }
    }

    save_convergence_history(&voltage_log);

    info!("Voltage stepping completed!");
    info!("Results saved to output/stepping/");
}

fn save_convergence_history(voltage_log: &[(f64, usize, f64)]) {
    let voltages: Vec<f64> = voltage_log.iter().map(|(v, _, _)| *v).collect();
    let iterations: Vec<f64> = voltage_log.iter().map(|(_, i, _)| *i as f64).collect();
    let residuals: Vec<f64> = voltage_log.iter().map(|(_, _, r)| *r).collect();

    csv_writer::write_csv(
        "output/stepping/convergence_history.csv",
        &["voltage", "iterations", "final_residual"],
        &[voltages, iterations, residuals],
    )
    .expect("Failed to write convergence history");

    info!("Convergence history saved to output/stepping/convergence_history.csv");
}
