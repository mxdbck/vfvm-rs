use nalgebra::DVector;
use std::fs;
use vfvm_rs::models::pn::pn::PnJunctionModel;
use vfvm_rs::numerics::sparse::SparseNewtonSolver;
use vfvm_rs::physics::PhysicsModel;
use vfvm_rs::processing::csv_writer;

fn main() {
    fs::create_dir_all("output/stepping").expect("Failed to create stepping output directory");

    let mesh_size = 1.0;
    let num_points = 500;
    let logging = true;

    let v_start = 0.0;
    let v_end = 1.0;
    let v_step = 0.1;

    println!("Voltage Stepping Simulation");
    println!("============================");
    println!("Start voltage: {:.3} V", v_start);
    println!("End voltage: {:.3} V", v_end);
    println!("Step size: {:.3} V", v_step);
    println!();

    let (mesh, params) = vfvm_rs::models::pn::pn::pn_problem_def(mesh_size, num_points, logging);
    let (v_scale, ni_norm, n_scale) = (params.v_scale, params.ni_norm, params.n_scale);

    let mut model = PnJunctionModel::new(params.clone(), 0.0, true).with_mesh(&mesh);

    let solver = SparseNewtonSolver {
        tolerance: 1e-8,
        max_iterations: 10000,
    };

    // Start with initial guess at equilibrium
    let mut current_solution = model.initial_condition(&mesh);

    save_step_solution(
        &mesh,
        &current_solution,
        0,
        v_start,
        v_scale,
        ni_norm,
        n_scale,
    );

    // Voltage stepping loop
    let num_steps = ((v_end - v_start) / v_step).round() as usize;
    let mut voltage_log: Vec<(f64, usize, f64)> = vec![(v_start, 0, 0.0)];

    for step in 0..=num_steps {
        let voltage = v_start + step as f64 * v_step;

        println!(
            "Step {}/{}: Solving at V = {:.3} V",
            step, num_steps, voltage
        );

        // model.v_applied = voltage / v_scale;
        model = PnJunctionModel::new(params.clone(), voltage, true).with_mesh(&mesh);

        // Update voltage and reconfigure boundary conditions
        model.apply_boundary_conditions(&mesh, &mut vec![]);

        // Solve using previous solution as initial guess
        match solver.solve(&model.functional, &mesh, current_solution.clone(), true) {
            Ok(result) => {
                println!(
                    "  Converged in {} iterations, residual: {:.3e}",
                    result.iterations, result.final_residual
                );

                current_solution = result.solution;
                voltage_log.push((
                    voltage,
                    result.iterations.try_into().unwrap(),
                    result.final_residual,
                ));

                // Save solution for this step
                // if step == num_steps {
                if true {
                    save_step_solution(
                        &mesh,
                        &current_solution,
                        step,
                        voltage,
                        v_scale,
                        ni_norm,
                        n_scale,
                    );
                }
            }
            Err(e) => {
                eprintln!("  Failed to converge: {}", e);
                eprintln!("  Stopping voltage stepping.");
                break;
            }
        }
    }

    save_convergence_history(&voltage_log);

    println!();
    println!("Voltage stepping completed!");
    println!("Results saved to output/stepping/");
}

fn save_step_solution(
    mesh: &vfvm_rs::discretization::mesh::Mesh,
    solution: &DVector<f64>,
    step: usize,
    voltage: f64,
    v_scale: f64,
    ni_norm: f64,
    n_scale: f64,
) {
    let x_positions: Vec<f64> = mesh.nodes.iter().map(|n| n.position[0]).collect();

    let mut psi: Vec<f64> = solution.iter().step_by(3).cloned().collect();
    let phi_n: Vec<f64> = solution.iter().skip(1).step_by(3).cloned().collect();
    let phi_p: Vec<f64> = solution.iter().skip(2).step_by(3).cloned().collect();

    let n_e: Vec<f64> = psi
        .iter()
        .zip(&phi_n)
        .map(|(psi, phi_n)| ni_norm * (psi - phi_n).exp() * n_scale)
        .collect();

    let p_h: Vec<f64> = psi
        .iter()
        .zip(&phi_p)
        .map(|(psi, phi_p)| ni_norm * (phi_p - psi).exp() * n_scale)
        .collect();

    // Scale and shift potential
    let shift = psi[0];
    psi.iter_mut().for_each(|p| {
        *p = (*p - shift) * v_scale;
    });

    let filename = format!("output/stepping/step_{:03}_V_{:.3}.csv", step, voltage);
    csv_writer::write_csv(
        &filename,
        &["x", "psi", "phi_n", "phi_p", "n_e", "p_h"],
        &[x_positions, psi, phi_n, phi_p, n_e, p_h],
    )
    .expect(&format!("Failed to write solution for step {}", step));
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

    println!("Convergence history saved to output/stepping/convergence_history.csv");
}
