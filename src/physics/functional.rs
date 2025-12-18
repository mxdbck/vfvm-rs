use crate::discretization::mesh::{Cell, Face, Mesh};
use crate::physics::bc::{BCRegistry, Field, Normal, Point, robin_ghost_val};
use nalgebra::DVector;
use num_dual::DualNum;
use std::collections::HashMap;

// Type aliases for our function signatures to keep things clean.
// Note the generic `T` for automatic differentiation.

// Flux function: f(flux_vector, u_left, u_right, face_geometry, user_data)
type FluxFn<T, D> = Box<dyn Fn(&mut [T], &[T], &[T], &Face, &D)>;

// Reaction/Source function: f(source_vector, u, cell_geometry, user_data)
type ReactionFn<T, D> = Box<dyn Fn(&mut [T], &[T], &Cell, &D)>;

// Storage function (for time-dependent term): f(storage_vector, u, cell_geometry, user_data)
type StorageFn<T, D> = Box<dyn Fn(&mut [T], &[T], &Cell, &D)>;

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub struct NumericalTolerances {
    pub min_distance: f64,
    pub min_face_area: f64,
    pub eps_diagonal: f64,
}

impl Default for NumericalTolerances {
    fn default() -> Self {
        Self {
            min_distance: 1e-14,
            min_face_area: 1e-20,
            eps_diagonal: 1e-12,
        }
    }
}

/// A generic PhysicsModel configured by user-defined functions (closures).
/// `T` is the numeric type (e.g., `f64` or `DualDVec64`) and `D` is a generic
/// type for any user-defined data/parameters struct.
pub struct FunctionalPhysics<T, D> {
    pub num_vars_per_cell: usize,
    pub data: D,
    pub(crate) flux: FluxFn<T, D>,
    pub reaction: ReactionFn<T, D>,
    #[allow(unused)]
    pub storage: StorageFn<T, D>,
    pub bc_registry: BCRegistry,
    pub face_tags: HashMap<usize, String>,
    pub field_names: Vec<Field>,
    pub current_time: Option<f64>,
    pub tolerances: NumericalTolerances,

    pub dt: Option<f64>,
    pub theta: f64, // 1.0 = Backward Euler, 0.5 = Crank-Nicolson, 0.0 = Forward Euler (ah heeeell nah)
    pub s_old_cache: Option<DVector<T>>,
    pub spatial_old_cache: Option<DVector<T>>,
}

impl<T, D> FunctionalPhysics<T, D>
where
    T: nalgebra::Scalar + DualNum<f64> + num_traits::Zero,
    D: 'static,
{
    pub fn new(
        field_names: Vec<Field>,
        data: D,
        flux: FluxFn<T, D>,
        reaction: ReactionFn<T, D>,
        storage: StorageFn<T, D>,
    ) -> Self {
        let num_vars = field_names.len();
        Self {
            num_vars_per_cell: num_vars,
            data,
            flux,
            reaction,
            storage,
            bc_registry: BCRegistry::default(),
            face_tags: HashMap::new(),
            field_names,
            current_time: None,
            tolerances: NumericalTolerances::default(),
            dt: None,
            theta: 1.0, // Default to Implicit Euler
            s_old_cache: None,
            spatial_old_cache: None,
        }
    }

    /// Set numerical tolerances explicitly for the physics model.
    #[allow(unused)]
    pub fn with_tolerances(mut self, tol: NumericalTolerances) -> Self {
        self.tolerances = tol;
        self
    }

    /// Calibrate tolerances based on mesh characteristics.
    /// Should be called after mesh generation and before solving.
    pub fn calibrate_tolerances(&mut self, mesh: &Mesh) {
        // Compute minimum cell spacing
        let mut min_spacing: f64 = f64::INFINITY;
        let mut max_spacing: f64 = 0.0;
        let mut count = 0;

        for face in &mesh.faces {
            if let (k, Some(l)) = face.neighbor_cell_ids {
                let d = Self::raw_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                min_spacing = min_spacing.min(d);
                max_spacing = max_spacing.max(d);
                count += 1;
            }
        }

        if count == 0 {
            eprintln!("WARNING: No interior faces found for tolerance calibration");
            return;
        }

        // Set min_distance as a small fraction of minimum mesh spacing
        // This ensures we don't artificially merge nodes while catching degenerate cases
        let old_tol = self.tolerances.min_distance;
        self.tolerances.min_distance = min_spacing * 1e-8;

        println!("Tolerance calibration:");
        println!(
            "  Mesh spacing: min={:.3e}, max={:.3e}",
            min_spacing, max_spacing
        );
        println!("  Old min_distance: {:.3e}", old_tol);
        println!(
            "  New min_distance: {:.3e} (= {:.2e} × min_spacing)\n",
            self.tolerances.min_distance, 1e-8
        );
    }

    /// Raw distance calculation without clamping.
    #[inline]
    fn raw_distance(p1: [f64; 3], p2: [f64; 3]) -> f64 {
        (0..=2).map(|i| (p2[i] - p1[i]).powi(2)).sum::<f64>().sqrt()
    }

    /// Calculate distance between two points with a minimum threshold.
    /// Issues a warning in debug builds if clamping occurs.
    #[inline]
    pub fn safe_distance(&self, p1: [f64; 3], p2: [f64; 3]) -> f64 {
        let d = Self::raw_distance(p1, p2);
        let safe_d = d.max(self.tolerances.min_distance);

        #[cfg(debug_assertions)]
        if d < self.tolerances.min_distance {
            eprintln!(
                "WARNING: Clamped distance {:.3e} → {:.3e} between {:?} and {:?}",
                d, safe_d, p1, p2
            );
        }

        safe_d
    }

    /// Calculate the geometric scale factor for a face (area/distance).
    #[inline]
    pub fn face_scale(face: &Face, d: f64) -> f64 {
        face.area / d
    }

    /// Create Point and Normal from face geometry.
    #[inline]
    pub fn face_geometry(face: &Face) -> (Point, Normal) {
        let p = Point {
            x: face.centroid[0],
            y: face.centroid[1],
            z: face.centroid[2],
        };
        let n = Normal {
            nx: face.normal[0],
            ny: face.normal[1],
            nz: face.normal[2],
        };
        (p, n)
    }

    /// Calculate delta (normal dot distance vector) for BC application.
    #[inline]
    pub fn bc_delta(face_centroid: [f64; 3], cell_centroid: [f64; 3], normal: Normal) -> f64 {
        normal.nx * (face_centroid[0] - cell_centroid[0])
            + normal.ny * (face_centroid[1] - cell_centroid[1])
            + normal.nz * (face_centroid[2] - cell_centroid[2])
    }

    /// Compute ghost values for boundary conditions.
    /// Returns a vector of ghost values for all variables at a boundary face.
    #[inline]
    pub fn compute_ghost_values(
        &self,
        u_interior: &[T],
        face: &Face,
        cell_centroid: [f64; 3],
        label: &str,
    ) -> Vec<T> {
        let (p, n) = Self::face_geometry(face);
        let delta = Self::bc_delta(face.centroid, cell_centroid, n);
        let t = self.current_time.unwrap_or(0.0);

        (0..self.num_vars_per_cell)
            .map(|j| {
                let field = &self.field_names[j];
                if let Some(rule) = self.bc_registry.find_for(field.0.as_ref(), label, p, n) {
                    let alpha = (rule.bc.alpha)(t, p, n);
                    let beta = (rule.bc.beta)(t, p, n);
                    let gamma = (rule.bc.gamma)(t, p, n);
                    robin_ghost_val(u_interior[j].clone(), alpha, beta, gamma, delta)
                } else {
                    u_interior[j].clone()
                }
            })
            .collect()
    }

    /// Compute the residual contribution from all fluxes across faces.
    fn flux_contribution(&self, mesh: &Mesh, u: &DVector<T>) -> DVector<T> {
        let mut residual = DVector::zeros(mesh.cells.len() * self.num_vars_per_cell);
        let mut f_flux = DVector::from_vec(vec![T::zero(); self.num_vars_per_cell]);

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    // Interior cell
                    let u_k = u.rows(k * self.num_vars_per_cell, self.num_vars_per_cell);
                    // Exterior cell
                    let u_l = u.rows(l * self.num_vars_per_cell, self.num_vars_per_cell);

                    f_flux.fill(T::zero());
                    (self.flux)(
                        f_flux.as_mut_slice(),
                        u_k.as_slice(),
                        u_l.as_slice(),
                        face,
                        &self.data,
                    );

                    let d = self.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let scale = Self::face_scale(face, d);

                    for i in 0..self.num_vars_per_cell {
                        let flux_val = f_flux[i].clone() * scale;
                        residual[k * self.num_vars_per_cell + i] += flux_val.clone();
                        residual[l * self.num_vars_per_cell + i] -= flux_val;
                    }
                }
                // Boundary cell
                (k, None) => {
                    let Some(label) = self.face_tags.get(&face_idx) else {
                        continue;
                    };
                    let u_k = u.rows(k * self.num_vars_per_cell, self.num_vars_per_cell);

                    let ghost = self.compute_ghost_values(
                        u_k.as_slice(),
                        face,
                        mesh.cells[k].centroid,
                        label,
                    );

                    f_flux.fill(T::zero());
                    (self.flux)(
                        f_flux.as_mut_slice(),
                        u_k.as_slice(),
                        &ghost,
                        face,
                        &self.data,
                    );

                    let d = self.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let scale = Self::face_scale(face, d);

                    for i in 0..self.num_vars_per_cell {
                        residual[k * self.num_vars_per_cell + i] += f_flux[i].clone() * scale;
                    }
                }
            }
        }

        residual
    }

    /// Compute the residual contribution from reactions/sources within each cell.
    fn reaction_contribution(&self, mesh: &Mesh, u: &DVector<T>) -> DVector<T> {
        let mut residual = DVector::zeros(mesh.cells.len() * self.num_vars_per_cell);
        let mut f_reaction = DVector::from_vec(vec![T::zero(); self.num_vars_per_cell]);

        for cell in &mesh.cells {
            let u_cell = u.rows(cell.id * self.num_vars_per_cell, self.num_vars_per_cell);

            f_reaction.fill(T::zero());

            (self.reaction)(
                f_reaction.as_mut_slice(),
                u_cell.as_slice(),
                cell,
                &self.data,
            );

            for i in 0..self.num_vars_per_cell {
                residual[cell.id * self.num_vars_per_cell + i] += f_reaction[i].clone() * cell.volume;
            }
        }

        residual
    }

    pub fn storage_contribution(&self, mesh: &Mesh, u: &DVector<T>) -> DVector<T> {
        let mut s_vec = DVector::zeros(mesh.cells.len() * self.num_vars_per_cell);
        let mut f_storage = DVector::from_vec(vec![T::zero(); self.num_vars_per_cell]);

        for cell in &mesh.cells {
            let u_cell = u.rows(cell.id * self.num_vars_per_cell, self.num_vars_per_cell);

            f_storage.fill(T::zero());

            (self.storage)(
                f_storage.as_mut_slice(),
                u_cell.as_slice(),
                cell,
                &self.data,
            );

            for i in 0..self.num_vars_per_cell {
                s_vec[cell.id * self.num_vars_per_cell + i] += f_storage[i].clone() * cell.volume;
            }
        }
        s_vec
    }

    /// Prepare the physics functional for a transient time step.
    /// This pre-calculates S(u_old) so it doesn't need to be recomputed during Newton iterations.
    pub fn prepare_time_step(&mut self, mesh: &Mesh, u_old: DVector<f64>, dt: f64) {
        self.dt = Some(dt);

        let u_old_t = u_old.map(|x| {
            T::from_f64(x).expect("Oopsie, u_old should be convertible to the AD numerical type")
        });

        let s_old_t = self.storage_contribution(mesh, &u_old_t);

        self.s_old_cache = Some(s_old_t);

        if self.theta < 1.0 {
            let spation_old_t = self.flux_contribution(mesh, &u_old_t)
                + self.reaction_contribution(mesh, &u_old_t);
            self.spatial_old_cache = Some(spation_old_t);
        } else {
            self.spatial_old_cache = None;
        }
    }

    /// Calculate the full residual vector.
    pub fn calculate_residual(&self, mesh: &Mesh, u: DVector<T>) -> DVector<T> {
        let mut spatial_current = self.flux_contribution(mesh, &u)
            + self.reaction_contribution(mesh, &u);
        let theta_t = T::from_f64(self.theta)
            .expect("Oopsie, theta should be convertible to the AD numerical type");

        let mut residual = spatial_current * theta_t.clone();

        if let Some(spation_old) = &self.spatial_old_cache {
            residual += spation_old * (T::one() - theta_t);
        }

        if let (Some(dt), Some(s_old)) = (self.dt, &self.s_old_cache) {
            let dt_t = T::from_f64(dt)
                .expect("Oopsie, dt should be convertible to the AD numerical type");
            residual += (self.storage_contribution(mesh, &u) - s_old.clone()) / dt_t;
        }
        residual
    }
}
