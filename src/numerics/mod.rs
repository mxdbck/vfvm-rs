pub mod nonlinear;
pub mod timing;
pub mod transient;

use nalgebra::DVector;

#[derive(Debug, Clone, Copy)]
pub enum Tolerance {
    Absolute(f64),
    Relative(f64),
    Combined(f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub enum ConvergenceMetric {
    MaxNorm,
    L2Norm,
}

#[derive(Debug, Clone, Copy)]
pub enum ConvergenceCriteria {
    Residual,
    Update,
    Both,
}

#[derive(Debug, Clone, Copy)]
pub struct Convergence {
    pub criteria: ConvergenceCriteria,
    pub tolerance: Tolerance,
    pub metric: ConvergenceMetric,
}

impl Convergence {
    pub fn norm(&self, v: &DVector<f64>) -> f64 {
        match self.metric {
            ConvergenceMetric::MaxNorm => v.amax(),
            ConvergenceMetric::L2Norm => v.norm(),
        }
    }

    pub fn check_convergence(
        &self,
        residual: &DVector<f64>,
        update: &DVector<f64>,
        initial_residual: f64,
        initial_update: f64,
    ) -> bool {
        let res_norm = self.norm(residual);
        let upd_norm = self.norm(update);

        let res_converged = match self.tolerance {
            Tolerance::Absolute(tol) => res_norm < tol,
            Tolerance::Relative(tol) => res_norm < tol * initial_residual,
            Tolerance::Combined(atol, rtol) => res_norm < atol + rtol * initial_residual,
        };

        let upd_converged = match self.tolerance {
            Tolerance::Absolute(tol) => upd_norm < tol,
            Tolerance::Relative(tol) => upd_norm < tol * initial_update,
            Tolerance::Combined(atol, rtol) => upd_norm < atol + rtol * initial_update,
        };

        match self.criteria {
            ConvergenceCriteria::Residual => res_converged,
            ConvergenceCriteria::Update => upd_converged,
            ConvergenceCriteria::Both => res_converged && upd_converged,
        }
    }
}
