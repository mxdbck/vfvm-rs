pub mod bc;
pub mod functional;

use kryst::matrix::sparse::CsrMatrix;
use nalgebra::DVector;
use num_dual::DualDVec64;

/// Defines the contract for a discretized model that can be solved.
/// This replaces the previous `PhysicsModel` which was mixing physics and discretization.
pub trait DiscreteModel {
    /// Returns the number of unknown variables per mesh cell.
    fn num_variables(&self) -> usize;

    /// Calculates the residual vector `R(u)` for the system of equations.
    /// This is the function that will be automatically differentiated.
    fn calculate_residual(&self, u: DVector<DualDVec64>) -> DVector<DualDVec64>;

    /// Computes both the residual vector and the Jacobian matrix at state `u`.
    fn compute_jacobian_and_residual(&self, u: &DVector<f64>) -> (DVector<f64>, CsrMatrix<f64>);
}
