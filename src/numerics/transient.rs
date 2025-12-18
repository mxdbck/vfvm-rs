use crate::discretization::mesh::Mesh;
use crate::numerics::sparse_aramijo::NewtonArmijoSolver;
use crate::numerics::Tolerance;
use crate::physics::functional::FunctionalPhysics;
use nalgebra::DVector;
use num_dual::DualDVec64;

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
    pub fn solve<F>(
        &self,
        model: &mut FunctionalPhysics<DualDVec64, F>,
        mesh: &Mesh,
        initial_condition: DVector<f64>,
        mut callback: impl FnMut(f64, &DVector<f64>),
    ) where
        F: 'static + Clone,
    {
        model.theta = self.theta;


        let mut u = initial_condition;
        let mut t = self.t_start;
        let mut dt = self.dt;

        let mut solver = NewtonArmijoSolver::default();
        solver.convergence.tolerance = Tolerance::Combined(self.tolerance, 1e-9);

        // Initialize history in functional
        model.prepare_time_step(mesh, u.clone(), dt);

        println!(
            "Starting Transient Simulation: T={:.2} -> {:.2}",
            self.t_start, self.t_end
        );

        let mut step = 0;
        while t < self.t_end {
            step += 1;

            model.prepare_time_step(mesh, u.clone(), dt);
            model.current_time = Some(t + dt);

            match solver.solve(model, mesh, u.clone(), false) {
                Ok(result) => {
                    // Accept step
                    t += dt;
                    u = result.solution;
                    // dt *= 1.2; // Simple timestep increase on success

                    println!(
                        "Step {:>4} | t = {:.4e} | dt = {:.3e} | iters = {}",
                        step, t, dt, result.iterations
                    );

                    // User callback (e.g., writing to file)
                    callback(t, &u);
                }
                Err(e) => {
                    println!(
                        "Step {:>4} | t = {:.4e} | dt = {:.3e} | FAILED: {}",
                        step, t, dt, e
                    );
                    dt *= 0.5; // Simple timestep reduction on failure
                }
            }
        }
    }
}
