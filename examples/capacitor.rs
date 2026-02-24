use log::{error, info};
use nalgebra::DVector;
use num_dual::DualDVec64;
use std::path::PathBuf;
use vfvm_rs::discretization::generator::create_flat_3d_mesh;
use vfvm_rs::discretization::generator::create_regular_2d_grid;
use vfvm_rs::discretization::mesh::{Cell, Face};
use vfvm_rs::physics::bc::{Field, BCRegistry};
use vfvm_rs::physics::functional::FunctionalPhysics;
use vfvm_rs::system::{System, InitialCondition, Geometry, SolverConfig, OutputConfig};

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

fn setup_electrostatics(params: CapacitorParams) -> FunctionalPhysics<CapacitorParams> {
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
    env_logger::builder().format_timestamp(None).init();
    let output_dir = "output/capacitor";

    let domain = [3.0, 3.0];
    let (nx, ny) = (500, 500);

    let points = create_regular_2d_grid(domain, nx, ny);
    let mesh = create_flat_3d_mesh(&points, domain, 0.1);
    info!("Mesh: {} cells", mesh.cells.len());

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

    let mut physics = setup_electrostatics(params);

    physics.calibrate_tolerances(&mesh);

    let bc_registry = BCRegistry::default();
    let init = DVector::zeros(mesh.cells.len());

    info!("Solving with Penalty Method...");

    let mut system = System::new(
        physics,
        Geometry { mesh },
        SolverConfig::default(),
        InitialCondition::Vector(init),
        bc_registry,
    );

    system.output_config = Some(OutputConfig {
        dir: PathBuf::from(output_dir),
        file_pattern: "potential.csv".to_string(),
        save_initial: false,
        save_final: true,
        save_transient_interval: None,
    });

    match system.solve() {
        Ok(res) => info!("Converged! Residual: {:.2e}", res.final_residual),
        Err(e) => error!("System Error: {}", e),
    }
}
