use crate::discretization::mesh::Mesh;
#[allow(unused)]
use crate::numerics::timing::{
    finalize_and_print, record_jacobian, record_linear_solve, reset_timing,
};
use crate::physics::PhysicsModel;
use nalgebra::{DMatrix, DVector};
use num_dual::{DualDVec64, jacobian};
use std::fs::File;
use std::io::{self, Write};
use thiserror::Error;

#[cfg(feature = "timing")]
use std::time::Instant;

pub struct NewtonSolver {
    pub tolerance: f64,
    pub max_iterations: u32,
}

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("linear solve failed")]
    LinearSolveFailed,
    #[error("Newton's method failed to converge")]
    NonConvergence,
}

pub struct SolverResult {
    pub solution: DVector<f64>,
    pub iterations: u32,
    pub final_residual: f64,
}

impl NewtonSolver {
    pub fn solve<M>(
        &self,
        model: &M,
        mesh: &Mesh,
        initial_guess: DVector<f64>,
        logging: bool,
    ) -> Result<SolverResult, SolverError>
    where
        M: PhysicsModel<DualDVec64>,
    {
        #[cfg(feature = "timing")]
        {
            reset_timing();
        }

        #[cfg(feature = "timing")]
        let solve_start = Instant::now();

        let mut u = initial_guess;
        let mut history: Vec<(u32, f64, f64, f64)> = Vec::new();
        let mut initial_residual = None;
        let mut previous_residual = None;

        if logging {
            println!("{} unknowns \n", u.len());
            println!("    Iter   | Residual |  Fraction |  Step % |  Initial");
        }

        for i in 0..self.max_iterations {
            let (residual, jacobian) =
                record_jacobian(|| self.compute_residual_and_jacobian(model, mesh, &u));

            let res_norm = residual.norm();
            let init = *initial_residual.get_or_insert(res_norm);
            let fraction = res_norm / init;
            let step_percent =
                previous_residual.map_or(0.0, |prev| (prev - res_norm) / prev * 100.0);
            previous_residual = Some(res_norm);

            log_iteration(
                i,
                self.max_iterations,
                res_norm,
                fraction,
                step_percent,
                init,
                logging,
            );
            history.push((i, res_norm, fraction, step_percent));

            if res_norm < self.tolerance {
                #[cfg(feature = "timing")]
                finalize_and_print(solve_start.elapsed());

                write_hist_to_file(history, initial_residual);
                return Ok(SolverResult {
                    solution: u,
                    iterations: i,
                    final_residual: res_norm,
                });
            }

            let delta_u = record_linear_solve(|| {
                jacobian
                    .lu()
                    .solve(&-residual)
                    .ok_or(SolverError::LinearSolveFailed)
            })?;

            let damping_factor = (i as f64 / 200.0).min(1.0);
            u += delta_u * damping_factor;
        }

        #[cfg(feature = "timing")]
        finalize_and_print(solve_start.elapsed());

        write_hist_to_file(history, initial_residual);
        Err(SolverError::NonConvergence)
    }

    // A helper that wraps the call to the AD library.
    pub fn compute_residual_and_jacobian<M: PhysicsModel<DualDVec64>>(
        &self,
        model: &M,
        mesh: &Mesh,
        u: &DVector<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let (residual, jac) = jacobian(
            |arg: DVector<DualDVec64>| model.calculate_residual(mesh, arg),
            u.clone(),
        );
        (residual, jac)
    }
}

pub(crate) fn log_iteration(
    i: u32,
    max_iter: u32,
    res_norm: f64,
    fraction: f64,
    step_percent: f64,
    init: f64,
    logging: bool,
) {
    if !logging {
        return;
    }
    if i == 0 {
        println!(
            "{i:>4} | {res_norm:>8.3e} | {fraction:>8.3e} | {step_percent:>6.2}% | {init:>8.3e}"
        );
    } else {
        print!("\x1B[1F\x1B[2K");
        println!(
            "{i:>4}/{max_iter} | {res_norm:>8.3e} | {fraction:>9.3e} | {step_percent:>6.2}% | {init:>8.3e}"
        );
    }
    io::stdout().flush().ok();
}

pub fn write_hist_to_file(history: Vec<(u32, f64, f64, f64)>, initial_residual: Option<f64>) {
    let init = initial_residual.unwrap_or(0.0);
    if let Ok(mut file) = File::create("output/main/solver_history.csv") {
        let _ = writeln!(file, "iter,residual,fraction,step_percent,initial");
        for (i, res, frac, step) in history {
            let _ = writeln!(file, "{i},{res},{frac},{step},{init}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::pn::pn::{PnJunctionModel, pn_problem_def};
    use crate::numerics::sparse::SparseNewtonSolver;

    #[test]
    fn sparse_and_dense_jacobians_are_close() {
        let (mesh, params) = pn_problem_def(1.0, 1000, true);
        let mut model = PnJunctionModel::new(params, 0.0, true).with_mesh(&mesh);
        model.apply_boundary_conditions(&mesh, &mut vec![]);
        let init = model.initial_condition(&mesh);

        let dense_solver = NewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };
        let sparse_solver = SparseNewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };

        let (_, j_dense) = dense_solver.compute_residual_and_jacobian(&model, &mesh, &init);
        let (_, j_sparse) =
            sparse_solver.compute_residual_and_jacobian(&model.functional, &mesh, &init);

        let n = j_dense.nrows();
        let j_sp_dense = j_sparse.to_dense();

        // Convert faer::Mat to nalgebra::DMatrix
        let j_sp_dense_nalgebra = DMatrix::from_fn(n, n, |i, j| j_sp_dense[(i, j)]);

        {
            let frob_dense = j_dense.norm();
            let frob_sparse_dense = j_sp_dense_nalgebra.norm();
            let diff_mat = &j_sp_dense_nalgebra - j_dense.clone();
            let frob_diff = diff_mat.norm();
            let rel_frob = frob_diff / frob_dense;

            let mut max_abs = 0.0;
            let mut max_abs_idx = (0usize, 0usize);
            for i in 0..n {
                for j in 0..n {
                    let d = (j_sp_dense_nalgebra[(i, j)] - j_dense[(i, j)]).abs();
                    if d > max_abs {
                        max_abs = d;
                        max_abs_idx = (i, j);
                    }
                }
            }

            let eps = 1e-12;
            let mut sum_rel2 = 0.0;
            let mut max_rel = 0.0;
            let mut max_rel_idx = (0usize, 0usize);
            for i in 0..n {
                for j in 0..n {
                    let denom = j_dense[(i, j)].abs().max(eps);
                    let rel = (j_sp_dense_nalgebra[(i, j)] - j_dense[(i, j)]).abs() / denom;
                    sum_rel2 += rel * rel;
                    if rel > max_rel {
                        max_rel = rel;
                        max_rel_idx = (i, j);
                    }
                }
            }
            let rms_rel = (sum_rel2 / ((n * n) as f64)).sqrt();

            let ab = j_sp_dense_nalgebra
                .iter()
                .zip(j_dense.iter())
                .fold(0.0, |acc, (a, b)| acc + (*a) * (*b));
            let cos = ab / (frob_dense * frob_sparse_dense);

            println!("Jacobian similarity metrics:");
            println!("  size: {}x{}", n, n);
            println!("  sparse nnz: {}", j_sparse.nnz());
            println!(
                "  Frobenius norms  | dense: {:.3e}  sparse->dense: {:.3e}",
                frob_dense, frob_sparse_dense
            );
            println!(
                "  Frobenius diff   | abs: {:.3e}  rel: {:.3e}",
                frob_diff, rel_frob
            );
            println!(
                "  Max abs diff     | {:.3e} at ({},{})",
                max_abs, max_abs_idx.0, max_abs_idx.1
            );
            println!(
                "  Max rel diff     | {:.3e} at ({},{})",
                max_rel, max_rel_idx.0, max_rel_idx.1
            );
            println!("  RMS rel diff     | {:.3e}", rms_rel);
            println!("  Cosine similarity| {:.6}", cos);
        }

        let diff = (&j_sp_dense_nalgebra - j_dense.clone()).norm();
        let norm = j_dense.norm();
        assert!(
            diff / norm < 1e-8,
            "relative difference too large: {diff} / {norm}"
        );
    }
}
