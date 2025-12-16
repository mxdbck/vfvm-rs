use kryst::solver::LinearSolver;
use kryst::{
    parallel::{NoComm, UniverseComm},
    preconditioner::PcSide,
};
use nalgebra::DVector;
use num_dual::DualDVec64;

#[allow(unused)]
use crate::numerics::timing::{finalize_and_print, reset_timing};
use crate::numerics::timing::{record_jacobian, record_linear_solve};
use crate::{
    discretization::mesh::Mesh,
    numerics::solver::{SolverError, SolverResult, write_hist_to_file},
    physics::functional::FunctionalPhysics,
};

use std::sync::Arc;
use std::time::Instant;

pub struct NewtonArmijoSolver {
    pub tolerance: f64,
    pub max_iterations: u32,
    /// Minimum step size before we give up (prevents infinite loops)
    pub min_step_size: f64,
    /// Parameter for sufficient decrease (usually 1e-4)
    pub armijo_param: f64,
    /// Maximum step size to prevent large jumps (None = no limit)
    pub max_step: Option<f64>,
    /// Forcing term for inexact Newton
    pub forcing_term: f64,
}

impl Default for NewtonArmijoSolver {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 50,
            min_step_size: 1e-3,
            armijo_param: 1e-4,
            max_step: None,
            forcing_term: 0.01,
        }
    }
}

impl NewtonArmijoSolver {
    pub fn solve<D: 'static>(
        &self,
        model: &FunctionalPhysics<DualDVec64, D>,
        mesh: &Mesh,
        initial_guess: DVector<f64>,
        logging: bool,
    ) -> Result<SolverResult, SolverError> {
        reset_timing();
        let solve_start = Instant::now();

        let mut u = initial_guess;
        let mut history: Vec<(u32, f64, f64, f64)> = Vec::new();

        let initial_res_vec = self.compute_residual_only(model, mesh, &u);
        let mut current_res_norm = initial_res_vec.norm();
        let initial_residual_norm = current_res_norm;

        if logging {
            println!("Newton-Armijo Solver started. {} unknowns.", u.len());
            println!("Initial Residual: {:.4e}", initial_residual_norm);
            println!("  Iter |  Residual  |   Step   | Alpha |  Lin. It |");
            println!("-------|------------|----------|-------|----------|");
        }

        for i in 0..self.max_iterations {
            // Compute Jacobian and Residual at current point
            // Note: We re-compute residual here to get the Jacobian.
            let (residual, mut jacobian) =
                record_jacobian(|| self.compute_residual_and_jacobian(model, mesh, &u));

            // Sanity check
            if !residual.iter().all(|x| x.is_finite()) {
                eprintln!("Error: Residual contains NaN or Inf at iteration {}", i);
                return Err(SolverError::LinearSolveFailed);
            }

            current_res_norm = residual.norm();

            // Check convergence
            if current_res_norm < self.tolerance {
                return self.success(
                    u,
                    i,
                    current_res_norm,
                    history,
                    initial_residual_norm,
                    solve_start,
                );
            }

            let n = residual.len();

            // We calculate row scaling factors D
            let d_inv: Vec<f64> = (0..n)
                .map(|row_idx| {
                    let row_start = jacobian.row_ptr()[row_idx];
                    let row_end = jacobian.row_ptr()[row_idx + 1];
                    let diag = (row_start..row_end)
                        .find(|&idx| jacobian.col_idx()[idx] == row_idx)
                        .map(|idx| jacobian.values()[idx])
                        .unwrap_or(1.0);
                    if diag.abs() < 1e-12 { 1.0 } else { 1.0 / diag }
                })
                .collect();

            // Scale Jacobian rows: A' = D^{-1} * A
            for row_idx in 0..n {
                let scale = d_inv[row_idx];
                let row_vals = jacobian.row_values_mut(row_idx);
                for val in row_vals.iter_mut() {
                    *val *= scale;
                }
            }

            // Scale RHS: b' = D^{-1} * (-r)
            let b: DVector<f64> =
                DVector::from_iterator(n, (0..n).map(|idx| -residual[idx] * d_inv[idx]));

            // Linear Solve (BiCGStab)
            let op = kryst::matrix::op::CsrOp::new(Arc::new(jacobian));
            let linear_tol = (current_res_norm * self.forcing_term)
                .max(self.tolerance)
                .min(1e-2);

            let mut bicgstab_solver =
                kryst::solver::bicgstab::BiCgStabSolver::new(linear_tol, 2000);
            let mut workspace = kryst::context::ksp_context::Workspace::new(n);
            bicgstab_solver.setup_workspace(&mut workspace);

            let mut delta_u = DVector::from_element(n, 0.0);

            let linear_stats = record_linear_solve(|| {
                bicgstab_solver.solve(
                    &op,
                    None,
                    b.as_slice(),
                    delta_u.as_mut_slice(),
                    PcSide::Left,
                    &UniverseComm::NoComm(NoComm {}),
                    None,
                    Some(&mut workspace),
                )
            });

            if let Err(e) = linear_stats {
                eprintln!("Linear solve failed at iter {}: {:?}", i, e);
                return Err(SolverError::LinearSolveFailed);
            }

            // Backtracking Line Search
            let mut alpha = 1.0;
            let mut accepted = false;
            let mut next_u = u.clone();
            let mut next_res_norm = current_res_norm;

            // Optionally limit the max norm of delta_u to prevent massive potential jumps
            if let Some(max_step) = self.max_step {
                let max_update = delta_u.amax();
                if max_update > max_step {
                    let scaling_factor = max_step / max_update;
                    delta_u *= scaling_factor;
                }
            }

            while alpha > self.min_step_size {
                // Candidate solution
                next_u = &u + &delta_u * alpha;

                // Compute ONLY the residual vector for the candidate (cheaper than Jacobian)
                let next_res = self.compute_residual_only(model, mesh, &next_u);
                next_res_norm = next_res.norm();

                // The Condition: ||F_new|| <= (1 - alpha * t) * ||F_old||
                // If armijo_param is 0.0, this simplifies to next < current
                let target_norm = (1.0 - alpha * self.armijo_param) * current_res_norm;

                if next_res_norm < target_norm {
                    accepted = true;
                    u = next_u;
                    current_res_norm = next_res_norm;
                    break;
                }

                // Backtrack
                alpha *= 0.5;
            }

            if !accepted {
                if logging {
                    println!("  Line search failed to find sufficient decrease.");
                }
                return Err(SolverError::NonConvergence);
            }

            let step_percent = (current_res_norm - next_res_norm) / current_res_norm * 100.0;
            history.push((
                i,
                next_res_norm,
                next_res_norm / initial_residual_norm,
                step_percent,
            ));

            if logging {
                let lin_iters = linear_stats.ok().map(|s| s.iterations).unwrap_or(0);
                println!(
                    "  {:4} | {:.4e} | {:.4e} | {:.3} | {:8} |",
                    i,
                    next_res_norm,
                    delta_u.norm(),
                    alpha,
                    lin_iters
                );
            }
        }
        Err(SolverError::NonConvergence)
    }

    fn success(
        &self,
        u: DVector<f64>,
        iter: u32,
        final_res: f64,
        history: Vec<(u32, f64, f64, f64)>,
        initial_res: f64,
        start_time: Instant,
    ) -> Result<SolverResult, SolverError> {
        finalize_and_print(start_time.elapsed());

        write_hist_to_file(history, Some(initial_res));

        Ok(SolverResult {
            solution: u,
            iterations: iter,
            final_residual: final_res,
        })
    }

    /// Helper to compute only the residual vector.
    /// This avoids calculating the Jacobian matrix elements during the line search loop.
    pub fn compute_residual_only<D: 'static>(
        &self,
        model: &FunctionalPhysics<DualDVec64, D>,
        mesh: &Mesh,
        u: &DVector<f64>,
    ) -> DVector<f64> {
        let n = u.len();
        let mut residual = DVector::<f64>::zeros(n);

        let u_dual: Vec<DualDVec64> = u.iter().map(|&x| DualDVec64::from_re(x)).collect();

        for r in 0..n {
            // Only compute the real part (value)
            residual[r] = model.residual_component_row(mesh, &u_dual, r).re;
        }
        residual
    }

    pub fn compute_residual_and_jacobian<D: 'static>(
        &self,
        model: &FunctionalPhysics<DualDVec64, D>,
        mesh: &Mesh,
        u: &DVector<f64>,
    ) -> (DVector<f64>, kryst::matrix::sparse::CsrMatrix<f64>) {
        let n = u.len();
        let mut residual = DVector::<f64>::zeros(n);
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);

        let m = model.num_vars_per_cell;
        let mut cols_reuse: Vec<usize> = Vec::with_capacity(8 * m);
        let mut vals_reuse: Vec<f64> = Vec::with_capacity(8 * m);
        let mut diag_reuse: Vec<f64> = Vec::with_capacity(m);

        let u_dual: Vec<DualDVec64> = u.iter().map(|&x| DualDVec64::from_re(x)).collect();
        let u_slice = u.as_slice();

        for r in 0..n {
            residual[r] = model.residual_component_row(mesh, &u_dual, r).re;

            cols_reuse.clear();
            vals_reuse.clear();
            diag_reuse.clear();

            diag_reuse.resize(m, 0.0);
            diag_reuse.fill(0.0);

            model.jacobian_row_locals(
                mesh,
                u_slice,
                r,
                &mut cols_reuse,
                &mut vals_reuse,
                &mut diag_reuse,
            );


            // let (cols, vals) = model.jacobian_row_locals(mesh, u_slice, r);
            // indices.extend(cols);
            // data.extend(vals);
            // indptr.push(indices.len());
            indices.extend_from_slice(&cols_reuse);
            data.extend_from_slice(&vals_reuse);
            indptr.push(indices.len());

            // let (cols, vals) = model.jacobian_row_locals(mesh, u_slice, r);
            // indices.extend(cols);
            // data.extend(vals);
            // indptr.push(indices.len());
        }

        let jacobian = kryst::matrix::sparse::CsrMatrix::from_csr(n, n, indptr, indices, data);
        (residual, jacobian)
    }
}
