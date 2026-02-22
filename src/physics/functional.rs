use crate::discretization::mesh::{Cell, Face, Mesh};
use crate::physics::bc::{BCRegistry, Field, Normal, Point};
use log::{info, warn};
use num_dual::DualDVec64;
use std::collections::HashMap;

// Type aliases for our function signatures to keep things clean.
// Using concrete DualDVec64 type for automatic differentiation.

// Flux function: f(flux_vector, u_left, u_right, face_geometry, user_data)
pub type FluxFn<D> = Box<dyn Fn(&mut [DualDVec64], &[DualDVec64], &[DualDVec64], &Face, &D) + Sync + Send>;

// Reaction/Source function: f(source_vector, u, cell_geometry, user_data)
pub type ReactionFn<D> = Box<dyn Fn(&mut [DualDVec64], &[DualDVec64], &Cell, &D) + Sync + Send>;

// Storage function (for time-dependent term): f(storage_vector, u, cell_geometry, user_data)
pub type StorageFn<D> = Box<dyn Fn(&mut [DualDVec64], &[DualDVec64], &Cell, &D) + Sync + Send>;

/// Action to take at a boundary for a specific variable.
pub enum BoundaryAction {
    /// Dirichlet: Fix the value at the boundary face.
    Dirichlet(DualDVec64),
    /// InjectedFlux: Apply an exact flux (Neumann/Robin) directly to the residual.
    InjectedFlux(DualDVec64),
}

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

/// A PhysicsModel configured by user-defined functions (closures).
/// Uses concrete `DualDVec64` type for automatic differentiation.
/// `D` is a generic type for any user-defined data/parameters struct.
pub struct FunctionalPhysics<D> {
    pub num_vars_per_cell: usize,
    pub data: D,
    pub flux: FluxFn<D>,
    pub reaction: ReactionFn<D>,
    #[allow(unused)]
    pub storage: StorageFn<D>,
    pub face_tags: HashMap<usize, String>,
    pub field_names: Vec<Field>,
    pub current_time: Option<f64>,
    pub tolerances: NumericalTolerances,
}

impl<D> FunctionalPhysics<D>
where
    D: 'static,
{
    pub fn new(
        field_names: Vec<Field>,
        data: D,
        flux: FluxFn<D>,
        reaction: ReactionFn<D>,
        storage: StorageFn<D>,
    ) -> Self {
        let num_vars = field_names.len();
        Self {
            num_vars_per_cell: num_vars,
            data,
            flux,
            reaction,
            storage,
            face_tags: HashMap::new(),
            field_names,
            current_time: None,
            tolerances: NumericalTolerances::default(),
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
            warn!("No interior faces found for tolerance calibration");
            return;
        }

        // Set min_distance as a small fraction of minimum mesh spacing
        // This ensures we don't artificially merge nodes while catching degenerate cases
        let old_tol = self.tolerances.min_distance;
        self.tolerances.min_distance = min_spacing * 1e-8;

        info!("Tolerance calibration:");
        info!(
            "  Mesh spacing: min={:.3e}, max={:.3e}",
            min_spacing, max_spacing
        );
        info!("  Old min_distance: {:.3e}", old_tol);
        info!(
            "  New min_distance: {:.3e} (= {:.2e} × min_spacing)",
            self.tolerances.min_distance,
            1e-8
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
            warn!(
                "Clamped distance {:.3e} → {:.3e} between {:?} and {:?}",
                d,
                safe_d,
                p1,
                p2
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

    /// Compute boundary actions (Dirichlet values or Injected Fluxes).
    #[inline]
    pub fn compute_boundary_values(
        &self,
        u_interior: &[DualDVec64],
        face: &Face,
        cell_centroid: [f64; 3],
        label: &str,
        bc_registry: &BCRegistry,
    ) -> Vec<BoundaryAction> {
        let (p, mut n) = Self::face_geometry(face);
        let delta = Self::bc_delta(face.centroid, cell_centroid, n);

        // Ensure normal points outward from the domain
        if delta < 0.0 {
            n.nx = -n.nx;
            n.ny = -n.ny;
            n.nz = -n.nz;
        }

        let t = self.current_time.unwrap_or(0.0);

        (0..self.num_vars_per_cell)
            .map(|j| {
                let field = &self.field_names[j];
                if let Some(rule) = bc_registry.find_for(field.0.as_ref(), label, p, n) {
                    let alpha = (rule.bc.alpha)(t, p, n);
                    let beta = (rule.bc.beta)(t, p, n);
                    let gamma = (rule.bc.gamma)(t, p, n);

                    if beta.abs() < 1e-14 {
                        // Pure Dirichlet: u = gamma / alpha
                        BoundaryAction::Dirichlet(DualDVec64::from(gamma / alpha))
                    } else {
                        // Neumann/Robin: Flux = (gamma - alpha * u_face) / beta
                        // Approximation: u_face = u_k + gradient * d = u_k - Flux * d
                        // Flux = (gamma - alpha * (u_k - Flux * d)) / beta
                        // Flux * (beta + alpha * d) = gamma - alpha * u_k
                        // Flux = (gamma - alpha * u_k) / (beta + alpha * d)

                        let d = delta.abs();
                        let denom = beta + alpha * d;

                        let u_k = &u_interior[j];
                        let val = (DualDVec64::from(gamma) - u_k.clone() * alpha) / denom;

                        // We compute the total flux (density * area) to inject into the residual.
                        // NOTE: GeneralizedBC defines BCs on the primitive variable u (and its derivative).
                        // Since Flux = - Gradient, we negate the computed gradient value.
                        BoundaryAction::InjectedFlux(-val * face.area)
                    }
                } else {
                    // No BC specified: homogeneous Neumann (zero flux)
                    BoundaryAction::InjectedFlux(DualDVec64::from(0.0))
                }
            })
            .collect()
    }
}
