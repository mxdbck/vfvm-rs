use crate::discretization::mesh::Mesh;
use crate::discretization::fvm::FvmDiscretizer;
use crate::numerics::nonlinear::{NonlinearSolver, SolverResult};
use crate::numerics::transient::TransientSolver;
use crate::physics::bc::BCRegistry;
use crate::physics::functional::FunctionalPhysics;
use crate::processing::csv_writer;
use log::info;
use nalgebra::DVector;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SystemError {
    #[error("Simulation failed: {0}")]
    SimulationFailed(String),
    #[error("Output error: {0}")]
    OutputError(String),
}

// Map NonlinearSolver error to SystemError
impl From<crate::numerics::nonlinear::SolverError> for SystemError {
    fn from(err: crate::numerics::nonlinear::SolverError) -> Self {
        SystemError::SimulationFailed(err.to_string())
    }
}

// Map IO error to SystemError
impl From<std::io::Error> for SystemError {
    fn from(err: std::io::Error) -> Self {
        SystemError::OutputError(err.to_string())
    }
}

pub type HistoryHandler = Box<dyn FnMut(u32, &DVector<f64>, f64) -> Result<(), Box<dyn std::error::Error>>>;

pub enum InitialCondition {
    Vector(DVector<f64>),
    Closure(Box<dyn Fn(&[f64; 3]) -> Vec<f64>>),
}

pub struct Geometry {
    pub mesh: Mesh,
}

#[derive(Clone)]
pub struct OutputConfig {
    pub dir: PathBuf,
    pub file_pattern: String,
    pub save_initial: bool,
    pub save_final: bool,
    pub save_transient_interval: Option<usize>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("output"),
            file_pattern: "solution.csv".to_string(),
            save_initial: false,
            save_final: true,
            save_transient_interval: None,
        }
    }
}

impl OutputConfig {
    pub fn save_snapshot(
        &self,
        mesh: &Mesh,
        fields: &[crate::physics::bc::Field],
        step: usize,
        _time: f64,
        u: &DVector<f64>,
        length_scale: f64,
        state_scales: &[f64],
    ) -> Result<(), SystemError> {
        std::fs::create_dir_all(&self.dir)?;

        let filename = if self.file_pattern.contains("{}") {
            self.file_pattern.replace("{}", &format!("{:05}", step))
        } else {
            self.file_pattern.clone()
        };

        let path = self.dir.join(filename);

        // Prepare data with scaling
        let x: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[0] * length_scale).collect();
        let y: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[1] * length_scale).collect();
        let z: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[2] * length_scale).collect();

        let num_vars = fields.len();
        let mut columns = vec![x, y, z];
        let mut headers = vec!["x", "y", "z"];

        // Extract fields
        for (i, field) in fields.iter().enumerate() {
            let scale = state_scales.get(i).cloned().unwrap_or(1.0);
            let col: Vec<f64> = u.iter().skip(i).step_by(num_vars).map(|&val| val * scale).collect();
            columns.push(col);
            headers.push(&field.0);
        }

        csv_writer::write_csv(&path, &headers, &columns)?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Preconditioner {
    None,
    Ilu0,
}

pub struct SolverConfig {
    pub tolerance: f64,
    pub max_iterations: u32,
    pub preconditioner: Preconditioner,
    pub history_handler: Option<HistoryHandler>,
    pub min_step_size: f64,
    pub armijo_param: f64,
    pub max_step: Option<f64>,
    pub forcing_term: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 100,
            preconditioner: Preconditioner::None,
            history_handler: None,
            min_step_size: 1e-6,
            armijo_param: 1e-4,
            max_step: None,
            forcing_term: 0.1,
        }
    }
}

pub struct TransientConfig {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
    pub theta: f64,
    pub step_handler: Option<Box<dyn FnMut(usize, f64, &DVector<f64>)>>,
}

pub struct System<P> {
    pub physics: FunctionalPhysics<P>,
    pub geometry: Geometry,
    pub solver_config: SolverConfig,
    pub transient_config: Option<TransientConfig>,
    pub output_config: Option<OutputConfig>,
    pub initial_condition: DVector<f64>,
    pub bc_registry: BCRegistry,
    pub length_scale: f64,
    pub state_scales: Vec<f64>,
    pub summary_handler: Option<Box<dyn Fn(&System<P>, &SolverResult)>>,
}

impl<P: 'static + Clone + Sync> System<P> {
    pub fn new(
        physics: FunctionalPhysics<P>,
        geometry: Geometry,
        solver_config: SolverConfig,
        initial_condition: InitialCondition,
        bc_registry: BCRegistry,
    ) -> Self {
        // Resolve Initial Condition
        let initial_condition_vec = match initial_condition {
            InitialCondition::Vector(v) => v,
            InitialCondition::Closure(closure) => {
                let mut values = Vec::with_capacity(geometry.mesh.cells.len() * 3);
                for cell in &geometry.mesh.cells {
                    let vals = closure(&cell.centroid);
                    values.extend(vals);
                }
                DVector::from_vec(values)
            }
        };

        let num_fields = physics.field_names.len();

        System {
            physics,
            geometry,
            solver_config,
            transient_config: None,
            output_config: None,
            initial_condition: initial_condition_vec,
            bc_registry,
            length_scale: 1.0,
            state_scales: vec![1.0; num_fields],
            summary_handler: None,
        }
    }

    pub fn solve(&mut self) -> Result<SolverResult, SystemError> {
        let result = self.solve_internal()?;

        if let Some(handler) = &self.summary_handler {
            handler(self, &result);
        } else {
            self.default_summary(&result);
        }

        Ok(result)
    }

    fn solve_internal(&mut self) -> Result<SolverResult, SystemError> {
        // Handle initial condition saving
        if let Some(cfg) = &self.output_config {
            if cfg.save_initial {
                // Determine step and time for IC
                // For transient, t_start is known. For steady, assume 0.0?
                let t_start = self.transient_config.as_ref().map(|tc| tc.t_start).unwrap_or(0.0);
                self.save_snapshot(0, t_start, &self.initial_condition)?;
            }
        }

        if let Some(tc) = &mut self.transient_config {
            let solver = TransientSolver {
                t_start: tc.t_start,
                t_end: tc.t_end,
                dt: tc.dt,
                theta: tc.theta,
                tolerance: self.solver_config.tolerance,
            };

            let fields_clone = self.physics.field_names.clone();
            let mesh_clone = self.geometry.mesh.clone();
            let length_scale = self.length_scale;
            let state_scales = self.state_scales.clone();

            let mut discretizer = FvmDiscretizer::new(&mut self.physics, &self.geometry.mesh, &self.bc_registry);

            // Wrap handler
            let mut user_handler = tc.step_handler.take().unwrap_or_else(|| Box::new(|_,_,_| {}));
            let output_config = self.output_config.clone();

            let wrapped_handler = Box::new(move |step: usize, t: f64, u: &DVector<f64>| {
                user_handler(step, t, u);

                if let Some(cfg) = &output_config {
                    if let Some(interval) = cfg.save_transient_interval {
                        if step > 0 && step % interval == 0 {
                            // Call save logic via config
                            let _ = cfg.save_snapshot(&mesh_clone, &fields_clone, step, t, u, length_scale, &state_scales);
                        }
                    }
                }
            });

            let result = solver.solve(&mut discretizer, self.initial_condition.clone(), wrapped_handler)?;

            // Save final
            if let Some(cfg) = &self.output_config {
                if cfg.save_final {
                     // Save "final" with high index
                     let _ = cfg.save_snapshot(&self.geometry.mesh, &self.physics.field_names, 999999, result.final_residual, &result.solution, self.length_scale, &self.state_scales);
                }
            }
            return Ok(result);
        }

        // Steady State Solver
        let mut solver = NonlinearSolver::new(
            {
                SolverConfig {
                    tolerance: self.solver_config.tolerance,
                    max_iterations: self.solver_config.max_iterations,
                    preconditioner: self.solver_config.preconditioner,
                    history_handler: self.solver_config.history_handler.take(),
                    min_step_size: self.solver_config.min_step_size,
                    armijo_param: self.solver_config.armijo_param,
                    max_step: self.solver_config.max_step,
                    forcing_term: self.solver_config.forcing_term,
                }
            }
        );

        let discretizer = FvmDiscretizer::new(&mut self.physics, &self.geometry.mesh, &self.bc_registry);
        let result = solver.solve(&discretizer, self.initial_condition.clone());
        self.solver_config.history_handler = solver.config.history_handler;

        let result = result?;

        if let Some(cfg) = &self.output_config {
            if cfg.save_final {
                self.save_snapshot(1, 0.0, &result.solution)?;
            }
        }

        Ok(result)
    }

    pub fn save_snapshot(&self, step: usize, time: f64, u: &DVector<f64>) -> Result<(), SystemError> {
        if let Some(cfg) = &self.output_config {
            cfg.save_snapshot(&self.geometry.mesh, &self.physics.field_names, step, time, u, self.length_scale, &self.state_scales)?;
        }
        Ok(())
    }

    pub fn default_summary(&self, result: &SolverResult) {
        info!("{}", "=".repeat(60));
        info!("SIMULATION SUMMARY");
        info!("{}", "=".repeat(60));
        info!("Geometry:");
        info!("  Cells:       {}", self.geometry.mesh.cells.len());
        info!("  Faces:       {}", self.geometry.mesh.faces.len());
        info!("  Nodes:       {}", self.geometry.mesh.nodes.len());
        info!("Solver Performance:");
        info!("  Time taken:  {:.4} s", result.solve_time.as_secs_f64());
        info!("  Iterations:  {}", result.iterations);
        info!("  Steps:       {}", result.step_count);
        info!("  Final Res.:  {:.4e}", result.final_residual);
        info!("Scaling:");
        info!("  Length scale: {:.4e}", self.length_scale);
        info!("  State scales: {:?}", self.state_scales);
        info!("{}", "=".repeat(60));
    }
  
    pub fn update_mesh(&mut self, new_mesh: Geometry, interpolated_u: DVector<f64>) {
        self.geometry = new_mesh;
        self.initial_condition = interpolated_u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::bc::Field;
    use crate::physics::functional::FunctionalPhysics;
    use num_dual::DualDVec64;
    use crate::discretization::mesh::{Face, Cell};

    #[derive(Debug, Clone, Copy)]
    struct DummyParams;

    fn create_dummy_physics() -> FunctionalPhysics<DummyParams> {
        let flux = Box::new(|_: &mut [DualDVec64], _: &[DualDVec64], _: &[DualDVec64], _: &Face, _: &DummyParams| {});
        let reaction = Box::new(|_: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &DummyParams| {});
        let storage = Box::new(|_: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &DummyParams| {});
        FunctionalPhysics::new(vec![Field::from("u")], DummyParams, flux, reaction, storage)
    }

    fn create_dummy_mesh() -> Mesh {
        Mesh {
            cells: vec![],
            faces: vec![],
            nodes: vec![],
            cell_face_ids: vec![],
        }
    }

    #[test]
    fn test_system_new_success() {
        let system = System::new(
            create_dummy_physics(),
            Geometry { mesh: create_dummy_mesh() },
            SolverConfig::default(),
            InitialCondition::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0])),
            BCRegistry::default(),
        );

        assert_eq!(system.initial_condition.len(), 3);
    }

    #[test]
    fn test_system_closure_ic() {
        let mut mesh = create_dummy_mesh();
        mesh.cells.push(crate::discretization::mesh::Cell {
            id: 0,
            volume: 1.0,
            centroid: [0.0, 0.0, 0.0],
            face_start: 0,
            face_end: 0,
        });

        let system = System::new(
            create_dummy_physics(),
            Geometry { mesh },
            SolverConfig {
                tolerance: 1e-6,
                max_iterations: 100,
                preconditioner: Preconditioner::None,
                history_handler: None,
                min_step_size: 1e-6,
                armijo_param: 1e-4,
                max_step: None,
                forcing_term: 0.1,
            },
            InitialCondition::Closure(Box::new(|_p| vec![1.0, 2.0])),
            BCRegistry::default(),
        );

        assert_eq!(system.initial_condition.len(), 2);
        assert_eq!(system.initial_condition[0], 1.0);
    }

    #[test]
    fn test_system_output_config() {
        let mut system = System::new(
            create_dummy_physics(),
            Geometry { mesh: create_dummy_mesh() },
            SolverConfig::default(),
            InitialCondition::Vector(DVector::from_vec(vec![1.0])),
            BCRegistry::default(),
        );

        system.output_config = Some(OutputConfig::default());
        assert!(system.output_config.is_some());
    }
}
