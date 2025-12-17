pub mod solver;
pub mod sparse;
pub mod sparse_aramijo;
pub mod timing;
pub mod transient;

pub enum Tolerance{
    Absolute(f64),
    Relative(f64),
    Combined(f64, f64),
}

pub enum ConvergenceMetric {
    L2Norm,
    MaxNorm,
}

/// Convergence criteria for iterative solvers
pub enum ConvergenceCriteria {
    Residual,
    Update,
    Both,
}

pub struct Convergence {
    pub criteria: ConvergenceCriteria,
    pub tolerance: Tolerance,
    pub metric: ConvergenceMetric,
}

impl Convergence {
    pub fn norm(&self, vector: &nalgebra::DVector<f64>) -> f64 {
        match self.metric {
            ConvergenceMetric::L2Norm => vector.norm(),
            ConvergenceMetric::MaxNorm => vector.amax(),
        }
    }

    pub fn check_tolerance(&self, norm: f64, initial_norm: f64) -> bool {
        match self.tolerance {
            Tolerance::Absolute(tol) => norm < tol,
            Tolerance::Relative(tol) => norm / initial_norm < tol,
            Tolerance::Combined(abs_tol, rel_tol) => {
                norm < abs_tol || (norm / initial_norm) < rel_tol
            }
        }
    }

    pub fn check_convergence(&self, residual: &nalgebra::DVector<f64>, update: &nalgebra::DVector<f64>, initial_residual_norm: f64, initial_update_norm: f64) -> bool {
        match self.criteria {
            ConvergenceCriteria::Residual => self.check_tolerance(self.norm(residual), initial_residual_norm),
            ConvergenceCriteria::Update => self.check_tolerance(self.norm(update), initial_update_norm),
            ConvergenceCriteria::Both => {
                self.check_tolerance(self.norm(residual), initial_residual_norm) &&
                self.check_tolerance(self.norm(update), initial_update_norm)
            }
        }
    }
}
