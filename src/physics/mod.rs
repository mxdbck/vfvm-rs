pub mod bc;
pub mod functional;
pub mod sparse;

use crate::discretization::mesh::Mesh;
use nalgebra::DVector;

/// Defines the contract for any physical model to be solved.
pub trait PhysicsModel<T: nalgebra::Scalar> {
    /// Returns the number of unknown variables per mesh cell.
    /// For semiconductors, this would be 3 (potential, electron concentration, hole concentration).
    fn num_variables(&self) -> usize;

    /// Calculates the residual vector `R(u)` for the system of equations.
    /// For a transient problem `M * du/dt = R(u)`, this function defines `R(u)`.
    /// This is the function that will be automatically differentiated.
    fn calculate_residual(&self, mesh: &Mesh, u: DVector<T>) -> DVector<T>;

    /// Compute a physics-aware initial condition.
    /// Default: zeros (but models should override with something smarter)
    fn initial_condition(&self, mesh: &Mesh) -> DVector<f64> {
        DVector::zeros(mesh.cells.len() * self.num_variables())
    }

    /// Apply boundary conditions based on input vector `u`.
    fn apply_boundary_conditions(&mut self, mesh: &Mesh, u: &mut Vec<f64>);
}
