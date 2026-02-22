use kryst::solver::LinearSolver;
use kryst::{
    parallel::{NoComm, UniverseComm},
    preconditioner::PcSide,
};
use nalgebra::DVector;
use std::sync::Arc;
use std::time::Instant;

use crate::numerics::timing::{finalize_and_print, record_jacobian, record_linear_solve, reset_timing};
use crate::numerics::{Convergence, ConvergenceCriteria, ConvergenceMetric, Tolerance};
use crate::physics::DiscreteModel;
use crate::system::SolverConfig;
use log::{error, info, warn};

#[derive(Debug, thiserror::Error)]
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
    pub solve_time: std::time::Duration,
    pub step_count: usize,
}

pub struct NonlinearSolver {
    pub config: SolverConfig,
}

impl NonlinearSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    pub fn solve(
        &mut self,
        model: &impl DiscreteModel,
        initial_guess: DVector<f64>,
    ) -> Result<SolverResult, SolverError> {
        reset_timing();
        let solve_start = Instant::now();

        let mut u = initial_guess;

        let (initial_res, _) = model.compute_jacobian_and_residual(&u);
        let initial_residual_norm = initial_res.norm();
        let mut current_res_norm;
        let mut initial_update_norm: Option<f64> = None;

        // Default logger if no handler
        if self.config.history_handler.is_none() {
            info!("Nonlinear Solver started. {} unknowns.", u.len());
            info!("Initial Residual: {:.4e}", initial_residual_norm);
            info!("  Iter |  Res Norm  |   Step   | Alpha |  Lin. It |");
            info!("-------|------------|----------|-------|----------|");
        }

        let convergence = Convergence {
            criteria: ConvergenceCriteria::Residual,
            tolerance: Tolerance::Relative(self.config.tolerance),
            metric: ConvergenceMetric::MaxNorm,
        };

        // Prevent panics if the initial guess is already converged
        if convergence.check_convergence(&initial_res, &DVector::zeros(u.len()), initial_residual_norm, 1.0) {
            return Ok(SolverResult {
                solution: u,
                iterations: 0,
                final_residual: initial_residual_norm,
                solve_time: solve_start.elapsed(),
                step_count: 0,
            });
        }

        for i in 0..self.config.max_iterations {
            let (residual, mut jacobian) =
                record_jacobian(|| model.compute_jacobian_and_residual(&u));

            if !residual.iter().all(|x| x.is_finite()) {
                error!("Error: Residual contains NaN or Inf at iteration {}", i);
                return Err(SolverError::LinearSolveFailed);
            }

            current_res_norm = residual.norm();

            let n = residual.len();

            // Jacobi scaling
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

            for row_idx in 0..n {
                let scale = d_inv[row_idx];
                let row_vals = jacobian.row_values_mut(row_idx);
                for val in row_vals.iter_mut() {
                    *val *= scale;
                }
            }

            let b: DVector<f64> =
                DVector::from_iterator(n, (0..n).map(|idx| -residual[idx] * d_inv[idx]));

            let op = kryst::matrix::op::CsrOp::new(Arc::new(jacobian));
            // Adaptive tolerance for inexact Newton
            let linear_tol = (b.norm() * self.config.forcing_term)
                .max(1e-18)
                .min(1e-2);

            let mut bicgstab_solver =
                kryst::solver::bicgstab::BiCgStabSolver::new(linear_tol, 2000);
            bicgstab_solver.atol = 1e-32; // Set absolute tolerance to a very small value to rely mostly on relative tolerance

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
                error!("Linear solve failed at iter {}: {:?}", i, e);
                return Err(SolverError::LinearSolveFailed);
            }

            // Backtracking Line Search
            let mut alpha = 1.0;
            let mut accepted = false;
            let mut next_u;
            let mut next_res_norm;

            let mut new_residual = residual.clone();

            if let Some(max_step) = self.config.max_step {
                let max_update = delta_u.amax();
                if max_update > max_step {
                    let scaling_factor = max_step / max_update;
                    delta_u *= scaling_factor;
                }
            }

            while alpha > self.config.min_step_size {
                next_u = &u + &delta_u * alpha;

                let next_u_dual = next_u.map(|x| num_dual::DualDVec64::from_re(x));
                let residual_dual = model.calculate_residual(next_u_dual);
                let residual_next: DVector<f64> = residual_dual.map(|x| x.re);

                next_res_norm = residual_next.norm();

                let target_norm = (1.0 - alpha * self.config.armijo_param) * current_res_norm;

                if next_res_norm < target_norm {
                    if initial_update_norm.is_none() {
                        initial_update_norm = Some(delta_u.norm());
                    }
                    accepted = true;
                    u = next_u;
                    current_res_norm = next_res_norm;
                    new_residual.copy_from(&residual_next);
                    break;
                }

                alpha *= 0.5;
            }

            if !accepted {
                if self.config.history_handler.is_none() {
                    warn!("  Line search failed to find sufficient decrease.");
                }
                return Err(SolverError::NonConvergence);
            }

            // Logging
            if let Some(handler) = &mut self.config.history_handler {
                if let Err(e) = handler(i, &u, current_res_norm) {
                    error!("History handler error: {}", e);
                }
            } else {
                let lin_iters = linear_stats.ok().map(|s| s.iterations).unwrap_or(0);
                info!(
                    "  {:4} | {:.4e} | {:.4e} | {:.3} | {:8} |",
                    i,
                    current_res_norm,
                    delta_u.norm(),
                    alpha,
                    lin_iters
                );
            }

            if convergence.check_convergence(&new_residual, &delta_u, initial_residual_norm, initial_update_norm.unwrap_or(1.0)) {
                 finalize_and_print(solve_start.elapsed());
                 return Ok(SolverResult {
                    solution: u,
                    iterations: i + 1,
                    final_residual: current_res_norm,
                    solve_time: solve_start.elapsed(),
                    step_count: 1,
                });
            }
        }

        Err(SolverError::NonConvergence)
    }
}
