mod discretization;
mod models;
mod numerics;
mod physics;
mod processing;

use crate::models::pn::pn::PnJunctionModel;
use crate::numerics::solver::NewtonSolver;
use crate::numerics::solver::SolverResult;
use crate::numerics::sparse_aramijo::NewtonArmijoSolver;
use crate::physics::PhysicsModel;
use crate::processing::csv_writer;
use crate::processing::summary::SimulationSummary;
use nalgebra::DVector;
use std::fs;

fn main() {
    fs::create_dir_all("output/main").expect("Failed to create output directory");

    let (mesh, params) = models::pn::pn::pn_problem_def(1.0, 300, true);
    let (v_scale, ni_norm, n_scale) = (params.v_scale, params.ni_norm, params.n_scale);

    let mut model = PnJunctionModel::new(params.clone(), 1e-4, true).with_mesh(&mesh);
    model.apply_boundary_conditions(&mesh, &mut vec![]);

    let mut summary = SimulationSummary::from_problem(&mesh, &params, &model);

    let initial_guess = model.initial_condition(&mesh);
    save_initial_guess(&mesh, &initial_guess);

    let dense_result = solve_dense(&model, &mesh, initial_guess.clone());
    let sparse_result = solve_sparse(&model, &mesh, initial_guess);

    if let Some(ref result) = dense_result {
        summary.add_dense_solver_info(result.iterations, result.final_residual);
        save_solution(&mesh, &result.solution, v_scale, ni_norm, n_scale);
    }

    if let Some(ref result) = sparse_result {
        summary.add_sparse_solver_info(result.iterations, result.final_residual);
    }

    if let (Some(dense), Some(sparse)) = (dense_result.as_ref(), sparse_result.as_ref()) {
        summary.add_comparison(&dense.solution, &sparse.solution);
        compare_solutions(&dense.solution, &sparse.solution);
    }

    summary
        .write_to_file("output/main/simulation_summary.txt")
        .expect("Failed to write summary");
    summary.print_to_console();

    println!("Summary saved to output/main/simulation_summary.txt");
}

fn save_initial_guess(mesh: &discretization::mesh::Mesh, guess: &DVector<f64>) {
    let x_positions: Vec<f64> = mesh.nodes.iter().map(|n| n.position[0]).collect();
    let psi_guess: Vec<f64> = guess.iter().cloned().step_by(3).collect();

    csv_writer::write_xy(
        "output/main/initial_guess.csv",
        "x",
        "psi_initial",
        &x_positions,
        &psi_guess,
    )
    .expect("Failed to write initial guess");
    println!("Initial guess saved to output/main/initial_guess.csv");
    println!()
}

fn solve_dense(
    model: &PnJunctionModel,
    mesh: &discretization::mesh::Mesh,
    guess: DVector<f64>,
) -> Option<SolverResult> {
    let solver = NewtonSolver {
        tolerance: 1e-8,
        max_iterations: 10000,
    };

    println!("Running Newton solver...");
    match solver.solve(model, mesh, guess, true) {
        Ok(result) => {
            println!("Solver finished successfully.\n");
            println!();
            Some(result)
        }
        Err(e) => {
            eprintln!("Solver failed: {}", e);
            None
        }
    }
}

fn solve_sparse(
    model: &PnJunctionModel,
    mesh: &discretization::mesh::Mesh,
    guess: DVector<f64>,
) -> Option<SolverResult> {
    // let solver = SparseNewtonSolver {
    //     tolerance: 1e-8,
    //     max_iterations: 10000,
    // };
    let mut solver = NewtonArmijoSolver::default();
    solver.max_step = Some(10.0);

    println!("Running Sparse Newton solver...");
    match solver.solve(&model.functional, mesh, guess, true) {
        Ok(result) => {
            println!("Sparse solver finished successfully.\n");
            Some(result)
        }
        Err(e) => {
            eprintln!("Sparse solver failed: {}", e);
            None
        }
    }
}

fn save_solution(
    mesh: &discretization::mesh::Mesh,
    solution: &DVector<f64>,
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

    let shift = psi[0];
    psi.iter_mut().for_each(|p| {
        *p = (*p - shift) * v_scale;
    });

    csv_writer::write_csv(
        "output/main/solution.csv",
        &["x", "psi", "phi_n", "phi_p", "n_e", "p_h"],
        &[x_positions, psi, phi_n, phi_p, n_e, p_h],
    )
    .expect("Failed to write solution");

    println!("Solution saved to output/main/solution.csv");
}

fn compare_solutions(dense: &DVector<f64>, sparse: &DVector<f64>) {
    let n = dense.len();
    if sparse.len() != n {
        eprintln!(
            "Cannot compare: different lengths (dense {}, sparse {})",
            n,
            sparse.len()
        );
        return;
    }

    let (l2, max_abs, max_idx, mae) = compute_error_metrics(dense, sparse);
    let rel_l2 = l2 / dense.norm();
    let cos_sim = sparse.dot(dense) / (sparse.norm() * dense.norm());
    let per_var_rel_l2 = compute_per_variable_error(dense, sparse, 3);

    println!("Solution comparison (sparse vs dense):");
    println!("  L2 diff: {:.3e}", l2);
    println!("  Relative L2 diff: {:.3e}", rel_l2);
    println!("  Mean abs error: {:.3e}", mae);
    println!(
        "  Max abs diff: {:.3e} at index {} (var {}, cell {})",
        max_abs,
        max_idx,
        max_idx % 3,
        max_idx / 3
    );
    println!("  Cosine similarity: {:.6}", cos_sim);
    println!(
        "  Per-variable rel L2 [psi, phi_n, phi_p]: [{:.3e}, {:.3e}, {:.3e}]",
        per_var_rel_l2[0], per_var_rel_l2[1], per_var_rel_l2[2]
    );

    // Save comparison
    let diff: Vec<f64> = (0..n).map(|i| (sparse[i] - dense[i]).abs()).collect();
    csv_writer::write_single_column("output/main/solver_comparison.csv", "abs_diff", &diff)
        .expect("Failed to write comparison");
    println!("Solver comparison saved to output/main/solver_comparison.csv");
}

fn compute_error_metrics(dense: &DVector<f64>, sparse: &DVector<f64>) -> (f64, f64, usize, f64) {
    let mut l2_sq = 0.0;
    let mut max_abs = 0.0;
    let mut max_idx = 0;
    let mut mae_sum = 0.0;

    for (i, (d, s)) in dense.iter().zip(sparse.iter()).enumerate() {
        let diff = (s - d).abs();
        l2_sq += diff * diff;
        mae_sum += diff;
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
    }

    (l2_sq.sqrt(), max_abs, max_idx, mae_sum / dense.len() as f64)
}

fn compute_per_variable_error(
    dense: &DVector<f64>,
    sparse: &DVector<f64>,
    num_vars: usize,
) -> Vec<f64> {
    (0..num_vars)
        .map(|var| {
            let (mut num, mut den) = (0.0, 0.0);
            for i in (var..dense.len()).step_by(num_vars) {
                let diff = sparse[i] - dense[i];
                num += diff * diff;
                den += dense[i] * dense[i];
            }
            if den > 0.0 {
                num.sqrt() / den.sqrt()
            } else {
                0.0
            }
        })
        .collect()
}
