use crate::discretization::fvm::FvmDiscretizer;
use crate::numerics::nonlinear::{NonlinearSolver, SolverResult};
use crate::physics::functional::{FluxFn, ReactionFn, StorageFn};
use crate::system::{Preconditioner, SolverConfig, SystemError};
use log::{error, info};
use nalgebra::DVector;

pub struct TransientSolver {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
    pub tolerance: f64,
    pub theta: f64,
}

impl Default for TransientSolver {
    fn default() -> Self {
        Self {
            t_start: 0.0,
            t_end: 1.0,
            dt: 1e-4,
            tolerance: 1e-5,
            theta: 1.0,
        }
    }
}

impl TransientSolver {
    pub fn solve<D, F, R, S>(
        &self,
        discretizer: &mut FvmDiscretizer<'_, D, F, R, S>,
        initial_condition: DVector<f64>,
        mut callback: impl FnMut(usize, f64, &DVector<f64>),
    ) -> Result<SolverResult, SystemError>
    where
        D: 'static + Sync,
        F: FluxFn<D>,
        R: ReactionFn<D>,
        S: StorageFn<D>,
    {
        discretizer.theta = self.theta;

        let mut u = initial_condition;
        let mut t = self.t_start;
        let mut dt = self.dt;

        let config = SolverConfig {
            tolerance: self.tolerance,
            max_iterations: 50,
            preconditioner: Preconditioner::None,
            history_handler: None,
            min_step_size: 1e-6,
            armijo_param: 1e-4,
            max_step: None,
            forcing_term: 0.1,
        };

        let mut solver = NonlinearSolver::new(config);

        // Initialize history in discretizer
        discretizer.prepare_time_step(u.clone(), dt);

        info!(
            "Starting Transient Simulation: T={:.2} -> {:.2}",
            self.t_start,
            self.t_end
        );

        let mut step = 0;
        let mut last_result: Option<SolverResult> = None;
        let start_time = std::time::Instant::now();

        while t < self.t_end {
            step += 1;

            discretizer.prepare_time_step(u.clone(), dt);
            discretizer.model.current_time = Some(t + dt);

            match solver.solve(discretizer, u.clone()) {
                Ok(result) => {
                    t += dt;
                    u = result.solution.clone();
                    info!(
                        "Step {:>4} | t = {:.4e} | dt = {:.3e} | iters = {}",
                        step,
                        t,
                        dt,
                        result.iterations
                    );
                    callback(step, t, &u);
                    last_result = Some(result);
                }
                Err(e) => {
                    error!(
                        "Step {:>4} | t = {:.4e} | dt = {:.3e} | FAILED: {}",
                        step,
                        t,
                        dt,
                        e
                    );
                    dt *= 0.5;
                    // If dt becomes too small, return error
                    if dt < 1e-15 {
                         return Err(SystemError::SimulationFailed(format!("Step failed: {}", e)));
                    }
                }
            }
        }

        last_result
            .map(|mut res| {
                res.solve_time = start_time.elapsed();
                res.step_count = step;
                res
            })
            .ok_or_else(|| SystemError::SimulationFailed("No steps completed".to_string()))
    }
}
