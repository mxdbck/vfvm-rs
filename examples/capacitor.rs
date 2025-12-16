use nalgebra::DVector;
use num_dual::DualDVec64;
use std::fs;
use vfvm_rs::discretization::generator::{create_flat_3d_mesh, create_regular_2d_grid};
use vfvm_rs::discretization::mesh::{Cell, Face, Mesh};
use vfvm_rs::numerics::sparse_aramijo::NewtonArmijoSolver;
use vfvm_rs::physics::PhysicsModel;
use vfvm_rs::physics::bc::Field;
use vfvm_rs::physics::functional::FunctionalPhysics;
use vfvm_rs::processing::csv_writer;

#[derive(Clone)]
struct PlateConfig {
    v_top: f64,
    v_bottom: f64,
    plate_dimensions: (f64, f64),
    separation: f64,
}

#[derive(Clone)]
struct CapacitorParams {
    epsilon: f64,
    plates: PlateConfig,
    penalty: f64, // Stiffness for forcing voltage (e.g., 1e6)
}

struct CapacitorModel {
    physics: FunctionalPhysics<DualDVec64, CapacitorParams>,
}

impl CapacitorModel {
    fn new(params: CapacitorParams) -> Self {
        Self {
            physics: setup_electrostatics(params),
        }
    }
}

impl PhysicsModel<DualDVec64> for CapacitorModel {
    fn num_variables(&self) -> usize {
        self.physics.num_vars_per_cell
    }

    fn calculate_residual(&self, mesh: &Mesh, u: DVector<DualDVec64>) -> DVector<DualDVec64> {
        self.physics.calculate_residual(mesh, u)
    }

    fn initial_condition(&self, mesh: &Mesh) -> DVector<f64> {
        DVector::zeros(mesh.cells.len())
    }

    fn apply_boundary_conditions(&mut self, _mesh: &Mesh, _u: &mut Vec<f64>) {
        // No boundary conditions on the outer walls. Defaults to zero flux -> isolated box.
        // Plate voltages are enforced internally by the reaction term.
    }
}

fn setup_electrostatics(params: CapacitorParams) -> FunctionalPhysics<DualDVec64, CapacitorParams> {
    // −ε∇ϕ
    let flux = Box::new(
        |f: &mut [DualDVec64],
         u_k: &[DualDVec64],
         u_l: &[DualDVec64],
         _face: &Face,
         data: &CapacitorParams| {
            f[0] = (u_k[0].clone() - u_l[0].clone()) * data.epsilon;
        },
    );

    let reaction = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], cell: &Cell, data: &CapacitorParams| {
            let x = cell.centroid[0];
            let y = cell.centroid[1];
            let p = &data.plates;

            let in_x = x.abs() <= p.plate_dimensions.0 / 2.0;

            let top_y = p.separation / 2.0;
            let in_top = in_x && (y - top_y).abs() <= p.plate_dimensions.1 / 2.0;

            let bot_y = -p.separation / 2.0;
            let in_bot = in_x && (y - bot_y).abs() <= p.plate_dimensions.1 / 2.0;

            if in_top {
                // Force u -> v_top using high penalty
                f[0] = (u[0].clone() - p.v_top) * data.penalty;
            } else if in_bot {
                // Force u -> v_bottom
                f[0] = (u[0].clone() - p.v_bottom) * data.penalty;
            } else {
                // Empty space (source = 0)
                f[0] = DualDVec64::from_re(0.0);
            }
        },
    );

    let storage = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], _cell: &Cell, _data: &CapacitorParams| {
            f[0] = u[0].clone();
        },
    );

    FunctionalPhysics::new(vec![Field::from("phi")], params, flux, reaction, storage)
}

fn main() {
    let output_dir = "output/capacitor";
    fs::create_dir_all(output_dir).unwrap();

    let domain = [3.0, 3.0];
    let (nx, ny) = (500, 500);

    let points = create_regular_2d_grid(domain, nx, ny);
    let mesh = create_flat_3d_mesh(&points, domain, 0.1);
    println!("Mesh: {} cells", mesh.cells.len());

    let plates = PlateConfig {
        v_top: 100.0,
        v_bottom: -100.0,
        plate_dimensions: (1.5, 0.2),
        separation: 1.0,
    };

    let params = CapacitorParams {
        epsilon: 1.0,
        plates,
        penalty: 1e8, // High stiffness factor
    };

    let mut model = CapacitorModel::new(params);

    model.physics.calibrate_tolerances(&mesh);

    println!("Solving with Penalty Method...");
    let solver = NewtonArmijoSolver::default();
    match solver.solve(&model.physics, &mesh, model.initial_condition(&mesh), true) {
        Ok(res) => {
            println!("Converged! Residual: {:.2e}", res.final_residual);

            let x: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[0]).collect();
            let y: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[1]).collect();
            let phi: Vec<f64> = res.solution.iter().cloned().collect();

            csv_writer::write_csv(
                &format!("{}/potential.csv", output_dir),
                &["x", "y", "phi"],
                &[x, y, phi],
            )
            .unwrap();
        }
        Err(e) => eprintln!("Solver Error: {}", e),
    }
}
