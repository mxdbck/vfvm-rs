// examples/particle.rs
use glam::DVec3;
use nalgebra::DVector;
use num_dual::DualDVec64;
use std::fs;

use vfvm_rs::discretization::generator::create_voronoi_mesh;
use vfvm_rs::discretization::mesh::{Cell, Face};
use vfvm_rs::numerics::transient::TransientSolver;
use vfvm_rs::physics::bc::Field;
use vfvm_rs::physics::functional::FunctionalPhysics;
use vfvm_rs::processing::csv_writer;

// Scaling Constants
const L_SCALE: f64 = 1.0e-9;          // Length scale: 1 nm
const E_SCALE: f64 = 1.60218e-19;     // Energy scale: 1 eV
const HBAR_SI: f64 = 1.05457e-34;     // Planck constant (J.s)
const MASS_SI: f64 = 9.10938e-31;     // Electron mass (kg)

const T_SCALE: f64 = HBAR_SI / E_SCALE;

#[derive(Clone)]
pub struct SchrodingerParams {
    pub hbar: f64,
    pub mass: f64,
    pub potential: Vec<f64>,
}

pub fn setup_schrodinger_physics(
    params: SchrodingerParams,
) -> FunctionalPhysics<DualDVec64, SchrodingerParams> {
    let reaction = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], cell: &Cell, data: &SchrodingerParams| {
            let (psi_r, psi_i) = (&u[0], &u[1]);
            let v_term = data.potential[cell.id] / data.hbar;
            f[0] = -psi_i.clone() * v_term;
            f[1] = psi_r.clone() * v_term;
        },
    );

    let flux = Box::new(
        |f: &mut [DualDVec64],
         u_k: &[DualDVec64],
         u_l: &[DualDVec64],
         _face: &Face,
         data: &SchrodingerParams| {
            let (r_k, i_k) = (&u_k[0], &u_k[1]);
            let (r_l, i_l) = (&u_l[0], &u_l[1]);
            let coeff = data.hbar / (2.0 * data.mass);
            f[0] = (i_l - i_k) * coeff;
            f[1] = (r_k - r_l) * coeff;
        },
    );

    let storage = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], _cell: &Cell, _data: &SchrodingerParams| {
            f[0] = u[0].clone();
            f[1] = u[1].clone();
        },
    );

    let fields = vec![Field::from("psi_r"), Field::from("psi_i")];
    FunctionalPhysics::new(fields, params, flux, reaction, storage)
}

fn main() {
    fs::create_dir_all("output/quantum").expect("Failed to create output directory");

    let width_sim = [200.0, 100.0, 1.0];
    let (nx, ny) = (200, 100);

    let mut generators = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            let x = (i as f64 / nx as f64 - 0.5) * width_sim[0];
            let y = (j as f64 / ny as f64 - 0.5) * width_sim[1];
            generators.push(DVec3::new(x, y, 0.0));
        }
    }
    let mesh = create_voronoi_mesh(&generators, width_sim);

    let x_sim: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[0]).collect();
    let y_sim: Vec<f64> = mesh.cells.iter().map(|c| c.centroid[1]).collect();

    let slit_width = 20.0;
    let wall_thick = 40.0;
    let v_wall_si = 1.0 * E_SCALE;

    let potential_sim: Vec<f64> = mesh.cells.iter().map(|c| {
        let x = c.centroid[0]; // scaled
        let y = c.centroid[1]; // scaled

        if x.abs() < wall_thick / 2.0 {
            let in_slit = y.abs() < slit_width / 2.0;
            if in_slit { 0.0 } else { v_wall_si / E_SCALE } // Scaled V = 1.0
        } else {
            0.0
        }
    }).collect();

    let k0_si = 1.5e9;
    let k_sim = k0_si * L_SCALE;

    let sigma_si = 8.0e-9;
    let sigma_sim = sigma_si / L_SCALE;

    let x_start_sim = -50.0;

    let mut init_vals = Vec::with_capacity(mesh.cells.len() * 2);
    for i in 0..mesh.cells.len() {
        let x = x_sim[i];
        let y = y_sim[i];

        let envelope = (-((x - x_start_sim).powi(2) + y.powi(2)) / (2.0 * sigma_sim.powi(2))).exp();
        init_vals.push(envelope * (k_sim * x).cos());
        init_vals.push(envelope * (k_sim * x).sin());
    }
    let u_init = DVector::from_vec(init_vals);

    // Effective Diffusion Coeff D_eff = hbar^2 / (2 * m * L^2 * E)
    let d_eff = (HBAR_SI.powi(2)) / (2.0 * MASS_SI * L_SCALE.powi(2) * E_SCALE);

    // In our code: coeff = hbar / (2*mass).
    // We set hbar_sim = 1.0.
    // We need 1.0 / (2*mass_sim) = d_eff  =>  mass_sim = 1.0 / (2*d_eff).
    let mass_sim = 1.0 / (2.0 * d_eff);

    let params = SchrodingerParams {
        hbar: 1.0,
        mass: mass_sim,
        potential: potential_sim
    };

    let mut model = setup_schrodinger_physics(params);

    let t_end_si = 1200.0e-15;
    let t_end_sim = t_end_si / T_SCALE;

    let dt_si = 4.0e-15; // Step size: 0.5 fs
    let dt_sim = dt_si / T_SCALE;

    let solver = TransientSolver {
        t_start: 0.0,
        t_end: t_end_sim,
        dt: dt_sim,
        tolerance: 1e-6,
        theta: 0.5,
    };

    println!("Starting Quantum Simulation (Scaled)...");
    println!("  L_scale: {:.2e} m", L_SCALE);
    println!("  T_scale: {:.2e} s", T_SCALE);
    println!("  D_eff:   {:.4}", d_eff);

    let mut frame_idx = 0;
    save_frame(frame_idx, 0.0, &u_init, &x_sim, &y_sim);

    solver.solve(&mut model, &mesh, u_init, |t_sim, u| {
        let t_fs = t_sim * T_SCALE * 1e15;
        let step = (t_sim / dt_sim).round() as u32;

        if step % 4 == 0 { // Save every 4 steps
            frame_idx += 1;
            save_frame(frame_idx, t_fs, u, &x_sim, &y_sim);
        }
    });
}

fn save_frame(idx: u32, t_fs: f64, u: &DVector<f64>, x_sim: &[f64], y_sim: &[f64]) {
    // Convert positions back to meters for plotting consistency
    let x_phys: Vec<f64> = x_sim.iter().map(|&x| x * L_SCALE).collect();
    let y_phys: Vec<f64> = y_sim.iter().map(|&y| y * L_SCALE).collect();

    let density: Vec<f64> = u
        .iter()
        .step_by(2)
        .zip(u.iter().skip(1).step_by(2))
        .map(|(r, i)| r * r + i * i)
        .collect();

    let filename = format!("output/quantum/wave_{:05}fs.csv", idx);

    csv_writer::write_csv(
        &filename,
        &["x", "y", "density"],
        &[x_phys, y_phys, density],
    )
    .expect("Failed to write output");

    println!("Saved frame {} at {:.2} fs", idx, t_fs);
}
