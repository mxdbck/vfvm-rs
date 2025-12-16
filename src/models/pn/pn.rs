use super::semiconductor::setup_semiconductor_physics;
use crate::discretization::generator::create_voronoi_mesh;
use crate::discretization::mesh::Mesh;
use crate::physics::PhysicsModel;
use crate::physics::bc::{BCRule, BoundarySelector, DirichletStyle, Field, GeneralizedBC};
use crate::physics::functional::FunctionalPhysics;
use glam::DVec3;
use nalgebra::DVector;
use num_dual::DualDVec64;

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct PnJunctionParams {
    pub dn: f64,            // Normalized electron diffusion coefficient
    pub dp: f64,            // Normalized hole diffusion coefficient
    pub ni: f64,            // Normalized intrinsic carrier concentration
    pub c_doping: Vec<f64>, // Doping profile (negative for p-type, positive for n-type)
    pub tau_n: f64,         // Normalized electron lifetime
    pub tau_p: f64,         // Normalized hole lifetime

    // Additional parameters useful for calculations
    pub q: f64,       // Elementary charge
    pub na_norm: f64, // Normalized acceptor concentration
    pub nd_norm: f64, // Normalized donor concentration
    pub ni_norm: f64, // Normalized intrinsic concentration
    pub v_scale: f64, // Voltage scaling (thermal voltage)
    pub l_scale: f64, // Length scaling (Debye length)
    pub n_scale: f64, // Concentration scaling
    pub eps_s: f64,   // Silicon permittivity
}

pub fn pn_problem_def(
    power_scale: f64,
    num_points: u32,
    logging: bool,
) -> (Mesh, PnJunctionParams) {
    // Physical parameters in cm-based units
    let ni_val = 1e10; // [cm^-3] Intrinsic carrier concentration
    let na_val = 2.0 * 10.0_f64.powf(16.0 * power_scale); // [cm^-3] Acceptor concentration (p-side)
    let nd_val = 1.0 * 10.0_f64.powf(16.0 * power_scale); // [cm^-3] Donor concentration (n-side)
    let eps_si = 11.68 * 8.854e-14; // [F/cm] Permittivity of Silicon
    let q = 1.602176634e-19; // [C] Elementary charge
    let t = 300.0; // [K] Temperature
    let kb = 1.380649e-23; // [J/K] Boltzmann constant

    // Scales for normalization
    // https://nanohub.org/resources/1545/download/ddmodel_introductory_part_word.pdf
    let v_scale = (kb * t) / q; // Thermal Voltage (V_T)
    let n_scale = na_val.max(nd_val); // Characteristic concentration
    let l_scale = ((eps_si * v_scale) / (q * n_scale)).sqrt(); // Extrinsic Debye Length
    let d_scale = 35.0; // Max diffusion coeff [cm^2/s]
    let time_scale = l_scale.powi(2) / d_scale;

    if logging {
        println!("--- Scaling Constants ---");
        println!("Potential Scale (V_T): {:.4} V", v_scale);
        println!("Concentration Scale (N_max): {:.2e} cm^-3", n_scale);
        println!("Length Scale (L_D): {:.4e} cm", l_scale);
        println!("Time Scale: {:.4e} s", time_scale);
        println!("-------------------------\n");
    }

    // Domain definition for a 1D PN junction
    let domain_width = 1.0e-4; // [cm]
    let norm_width = domain_width / l_scale; // Normalized width
    let spacing = domain_width / (num_points as f64 - 1.0); // Physical spacing
    let h = spacing / l_scale; // Normalized min mesh spacing

    let mut x_left = vec![];
    let mut current_x = 0.0;
    while current_x > -norm_width / 2.0 {
        x_left.push(current_x);
        current_x -= h;
    }
    x_left.reverse();

    let mut x_right = vec![];
    current_x = 0.0;
    while current_x < norm_width / 2.0 {
        x_right.push(current_x);
        current_x += h;
    }

    let mut coordinates = x_left;
    coordinates.extend(x_right.into_iter().skip(1)); // Skip the duplicate point at x=0

    // Make them three dimensional with zero second and third coordinates
    let mut coordinates_3d = Vec::new();
    for &x in &coordinates {
        coordinates_3d.push(DVec3::new(x, 0.0, 0.0));
    }

    let width = [norm_width, norm_width / 2.0, norm_width / 2.0];
    let mesh = create_voronoi_mesh(&coordinates_3d, width);

    // Physics setup (normalized)
    let ni_norm = ni_val / n_scale;
    let na_norm = na_val / n_scale;
    let nd_norm = nd_val / n_scale;

    // Doping profile: p-side for x<0, n-side for x>0
    let c_doping: Vec<f64> = mesh
        .cells
        .iter()
        .map(|cell| {
            let x_coord = cell.centroid[0]; // Assuming 1D mesh with x-coordinates
            if x_coord < 0.0 { -na_norm } else { nd_norm }
        })
        .collect();

    let params = PnJunctionParams {
        dn: 35.0 / d_scale,
        dp: 12.0 / d_scale,
        ni: ni_norm,
        c_doping,
        tau_n: 1e-7 / time_scale,
        tau_p: 1e-7 / time_scale,
        q,
        na_norm,
        nd_norm,
        ni_norm,
        v_scale,
        l_scale,
        n_scale,
        eps_s: eps_si,
    };

    (mesh, params)
}

/// A complete PN junction model that implements PhysicsModel.
pub struct PnJunctionModel {
    pub functional: FunctionalPhysics<DualDVec64, PnJunctionParams>,
    pub v_applied: f64, // Applied voltage in physical units (V)
    pub logging: bool,
    pub(crate) bcs_configured: bool,
}

impl PnJunctionModel {
    pub fn new(params: PnJunctionParams, v_applied: f64, logging: bool) -> Self {
        let functional = setup_semiconductor_physics(params);
        Self {
            functional,
            v_applied,
            logging,
            bcs_configured: false,
        }
    }

    /// Initialize model with mesh-calibrated tolerances.
    pub fn with_mesh(mut self, mesh: &Mesh) -> Self {
        self.functional.calibrate_tolerances(mesh);
        self
    }
}

/// Calculate equilibrium potential from charge neutrality.
fn calculate_equilibrium_psi(c: f64, ni: f64) -> f64 {
    let a = ni;
    let b = -c;
    let cc = -ni;
    let disc = b * b - 4.0 * a * cc;
    let x = (-b + disc.sqrt()) / (2.0 * a);
    x.ln()
}

impl PhysicsModel<DualDVec64> for PnJunctionModel {
    fn num_variables(&self) -> usize {
        self.functional.num_vars
    }

    fn calculate_residual(&self, mesh: &Mesh, u: DVector<DualDVec64>) -> DVector<DualDVec64> {
        self.functional.calculate_residual(mesh, u)
    }

    fn initial_condition(&self, mesh: &Mesh) -> DVector<f64> {
        let num_vars = self.num_variables();
        let params = &self.functional.data;

        let v_app_norm = self.v_applied / params.v_scale;

        // Equilibrium potentials from charge neutrality
        let psi_l_eq = calculate_equilibrium_psi(-params.na_norm, params.ni_norm);
        let psi_r_eq = calculate_equilibrium_psi(params.nd_norm, params.ni_norm);

        if self.logging {
            println!("Initial Condition:");
            println!("  Left equilibrium potential: {}", psi_l_eq);
            println!("  Right equilibrium potential: {}", psi_r_eq);
            println!("  Applied voltage (normalized): {}\n", v_app_norm);
        }

        // Boundary values
        let psi_left = psi_l_eq + v_app_norm;
        let psi_right = psi_r_eq;
        let phi_left = v_app_norm;
        let phi_right = 0.0;

        // Get mesh bounds for interpolation
        let x_min = mesh
            .nodes
            .iter()
            .map(|n| n.position[0])
            .reduce(f64::min)
            .unwrap();
        let x_max = mesh
            .nodes
            .iter()
            .map(|n| n.position[0])
            .reduce(f64::max)
            .unwrap();
        let width = if x_max > x_min { x_max - x_min } else { 1.0 };

        // Create linear initial guess across nodes
        let n_nodes = mesh.nodes.len();
        let mut init = Vec::with_capacity(num_vars * n_nodes);

        for i in 0..n_nodes {
            let node_x = mesh.nodes[i].position[0];
            let t = (node_x / width) + 0.5;

            let psi_i = (1.0 - t) * psi_left + t * psi_right;
            let phi_n_i = (1.0 - t) * phi_left + t * phi_right;
            let phi_p_i = (1.0 - t) * phi_left + t * phi_right;

            init.push(psi_i);
            init.push(phi_n_i);
            init.push(phi_p_i);
        }

        DVector::from_vec(init)
    }

    fn apply_boundary_conditions(&mut self, mesh: &Mesh, _u: &mut Vec<f64>) {
        let phys = &mut self.functional;
        let params = &phys.data;

        let v_app_norm = self.v_applied / params.v_scale;

        // Equilibrium potentials
        let psi_l_eq = calculate_equilibrium_psi(-params.na_norm, params.ni_norm);
        let psi_r_eq = calculate_equilibrium_psi(params.nd_norm, params.ni_norm);

        if self.logging {
            println!("Configuring Boundary Conditions:");
            println!(
                "  Left contact: psi={}, phi_n={}, phi_p={}",
                psi_l_eq + v_app_norm,
                v_app_norm,
                v_app_norm
            );
            println!("  Right contact: psi={}, phi_n=0, phi_p=0\n", psi_r_eq);
        }

        // Register left contact BCs
        phys.bc_registry.add(BCRule {
            field: Field::from("psi"),
            on: BoundarySelector::Label("left_contact".into()),
            bc: GeneralizedBC::dirichlet(psi_l_eq + v_app_norm),
            style: DirichletStyle::Strong,
        });
        phys.bc_registry.add(BCRule {
            field: Field::from("phi_n"),
            on: BoundarySelector::Label("left_contact".into()),
            bc: GeneralizedBC::dirichlet(v_app_norm),
            style: DirichletStyle::Strong,
        });
        phys.bc_registry.add(BCRule {
            field: Field::from("phi_p"),
            on: BoundarySelector::Label("left_contact".into()),
            bc: GeneralizedBC::dirichlet(v_app_norm),
            style: DirichletStyle::Strong,
        });

        // Register right contact BCs
        phys.bc_registry.add(BCRule {
            field: Field::from("psi"),
            on: BoundarySelector::Label("right_contact".into()),
            bc: GeneralizedBC::dirichlet(psi_r_eq),
            style: DirichletStyle::Strong,
        });
        phys.bc_registry.add(BCRule {
            field: Field::from("phi_n"),
            on: BoundarySelector::Label("right_contact".into()),
            bc: GeneralizedBC::dirichlet(0.0),
            style: DirichletStyle::Strong,
        });
        phys.bc_registry.add(BCRule {
            field: Field::from("phi_p"),
            on: BoundarySelector::Label("right_contact".into()),
            bc: GeneralizedBC::dirichlet(0.0),
            style: DirichletStyle::Strong,
        });

        // Identify and tag boundary faces
        let mut id_left = 0;
        let mut id_right = 0;

        for (i, face) in mesh.faces.iter().enumerate() {
            if face.neighbor_cell_ids.1.is_none() {
                if face.normal[0] > 0.0 {
                    if self.logging {
                        println!(
                            "Left boundary face: {:?}, normal: {:?}",
                            face.centroid, face.normal
                        );
                    }
                    id_left = i;
                } else if face.normal[0] < 0.0 {
                    if self.logging {
                        println!(
                            "Right boundary face: {:?}, normal: {:?}\n",
                            face.centroid, face.normal
                        );
                    }
                    id_right = i;
                }
            }
        }

        phys.face_tags.insert(id_left, "left_contact".into());
        phys.face_tags.insert(id_right, "right_contact".into());

        self.bcs_configured = true;
    }
}
