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

pub struct SparseNewtonSolver {
    pub tolerance: f64,
    pub max_iterations: u32,
}

impl SparseNewtonSolver {
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
        let mut initial_residual = None;
        let mut previous_residual = None;
        // let mut inner_tol: f64 = 1e-2;

        if logging {
            println!("{} unknowns \n", u.len());
            println!("    Iter   | Residual |  Fraction |  Step % |  Initial");
        }

        for i in 0..self.max_iterations {
            let (residual, mut jacobian) =
                record_jacobian(|| self.compute_residual_and_jacobian(model, mesh, &u));

            // Check for NaN/Inf in residual
            if !residual.iter().all(|x| x.is_finite()) {
                eprintln!("Error: Residual contains NaN or Inf at iteration {}", i);
                return Err(SolverError::LinearSolveFailed);
            }

            let n = residual.len();
            let res_norm = residual.norm();
            let init = *initial_residual.get_or_insert(res_norm);
            let fraction = res_norm / init;
            let step_percent =
                previous_residual.map_or(0.0, |prev| (prev - res_norm) / prev * 100.0);
            previous_residual = Some(res_norm);

            super::solver::log_iteration(
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
                finalize_and_print(solve_start.elapsed());

                write_hist_to_file(history, initial_residual);
                return Ok(SolverResult {
                    solution: u,
                    iterations: i,
                    final_residual: res_norm,
                });
            }

            // Apply Jacobi row scaling to improve conditioning
            let d: Vec<f64> = (0..n)
                .map(|row_idx| {
                    // Get row range from indptr
                    let row_start = jacobian.row_ptr()[row_idx];
                    let row_end = jacobian.row_ptr()[row_idx + 1];

                    // Find diagonal element in this row
                    let diag = (row_start..row_end)
                        .find(|&idx| jacobian.col_idx()[idx] == row_idx)
                        .map(|idx| jacobian.values()[idx])
                        .unwrap_or(1.0);

                    if diag.abs() < 1e-12 { 1.0 } else { diag }
                })
                .collect();

            // Debug: print some diagonal statistics at first iteration
            if i == 0 && logging {
                let d_min = d.iter().fold(f64::INFINITY, |a, &b| a.min(b.abs()));
                let d_max = d.iter().fold(0.0, |a: f64, &b| a.max(b.abs()));
                let d_mean = d.iter().sum::<f64>() / n as f64;
                println!(
                    "Diagonal stats: min={:.3e}, max={:.3e}, mean={:.3e}",
                    d_min, d_max, d_mean
                );
            }

            // Scale Jacobian rows
            for row_idx in 0..n {
                let scale = 1.0 / d[row_idx];
                let row_vals = jacobian.row_values_mut(row_idx);
                for val in row_vals.iter_mut() {
                    *val *= scale;
                }
            }

            let op = kryst::matrix::op::CsrOp::new(Arc::new(jacobian));

            // Use more relaxed tolerance for linear solver, relative to Newton residual
            let linear_tol = (res_norm * 0.1).max(self.tolerance).min(1e-2);
            // let linear_tol = 1e-12;
            let mut bicgstab_solver =
                kryst::solver::bicgstab::BiCgStabSolver::new(linear_tol, 2000);
            let mut workspace = kryst::context::ksp_context::Workspace::new(n);
            bicgstab_solver.setup_workspace(&mut workspace);
            // let mut pc = kryst::preconditioner::ilu_csr::IluCsr::new_with_config(Default::default());

            // let pc_result = pc.setup(&op);
            // if i == 0 && logging {
            //     println!("Preconditioner setup result: {:?}", pc_result);
            // }

            let mut x = DVector::from_element(n, 0.0);
            // Scale RHS by same factors
            let b: DVector<f64> =
                DVector::from_iterator(n, (0..n).map(|idx| -residual[idx] / d[idx]));

            // Check for NaN/Inf in RHS
            if !b.iter().all(|x| x.is_finite()) {
                eprintln!("Error: RHS contains NaN or Inf at iteration {}", i);
                return Err(SolverError::LinearSolveFailed);
            }

            // let result = bicgstab_solver.solve(&op, Some(&mut pc), b.as_slice(), x.as_mut_slice(), PcSide::Left, &UniverseComm::NoComm(NoComm {}), None, Some(&mut workspace));

            let b_norm = b.norm();
            if i == 0 && logging {
                println!("Scaled RHS norm: {:.3e}", b_norm);
            }

            let result = record_linear_solve(|| {
                bicgstab_solver.solve(
                    &op,
                    None,
                    b.as_slice(),
                    x.as_mut_slice(),
                    PcSide::Left,
                    &UniverseComm::NoComm(NoComm {}),
                    None,
                    Some(&mut workspace),
                )
            });

            match result {
                Ok(stats) => {
                    if i == 0 && logging {
                        println!("Linear solve result: {:?}", stats);
                    }
                    // Check for NaN in solution
                    if !x.iter().all(|val| val.is_finite()) {
                        eprintln!("Error: Linear solver produced NaN/Inf");
                        return Err(SolverError::LinearSolveFailed);
                    }
                }
                Err(e) => {
                    eprintln!("Linear solve failed: {:?}", e);
                    return Err(SolverError::LinearSolveFailed);
                }
            }

            let delta_u = x;
            let damping_factor = ((i + 19) as f64 / 200.0).min(1.0);
            u += delta_u * damping_factor;
        }

        #[cfg(feature = "timing")]
        finalize_and_print(solve_start.elapsed());

        write_hist_to_file(history, initial_residual);

        if let Some(r) = previous_residual {
            if r < self.tolerance {
                Ok(SolverResult {
                    solution: u,
                    iterations: self.max_iterations,
                    final_residual: r,
                })
            } else {
                Err(SolverError::NonConvergence)
            }
        } else {
            Err(SolverError::NonConvergence)
        }
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

        // dual representation for residual evaluation
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
                &mut diag_reuse
            );


            // let (cols, vals) = model.jacobian_row_locals(mesh, u_slice, r);
            // indices.extend(cols);
            // data.extend(vals);
            // indptr.push(indices.len());
            indices.extend_from_slice(&cols_reuse);
            data.extend_from_slice(&vals_reuse);
            indptr.push(indices.len());
        }

        // let jacobian = CsMatI::<f64, usize>::new((n, n), indptr, indices, data);
        let jacobian = kryst::matrix::sparse::CsrMatrix::from_csr(n, n, indptr, indices, data);
        (residual, jacobian)
    }
}
