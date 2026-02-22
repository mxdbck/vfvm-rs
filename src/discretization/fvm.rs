use crate::discretization::mesh::Mesh;
use crate::physics::bc::BCRegistry;
use crate::physics::functional::{BoundaryAction, FunctionalPhysics};
use crate::physics::DiscreteModel;
use kryst::matrix::sparse::CsrMatrix;
use nalgebra::{DVector, Dyn, U1};
use num_dual::{Derivative, DualDVec64};
use rayon::prelude::*;

pub struct SparsityPattern {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
}

pub struct FvmDiscretizer<'a, D> {
    pub model: &'a mut FunctionalPhysics<D>,
    pub mesh: &'a Mesh,
    pub bc_registry: &'a BCRegistry,
    pub dt: Option<f64>,
    pub theta: f64,
    pub s_old_cache: Option<DVector<DualDVec64>>,
    pub spatial_old_cache: Option<DVector<DualDVec64>>,
    pub sparsity_pattern: SparsityPattern,
}

impl<'a, D> FvmDiscretizer<'a, D>
where
    D: 'static,
{
    pub fn new(model: &'a mut FunctionalPhysics<D>, mesh: &'a Mesh, bc_registry: &'a BCRegistry) -> Self {
        let sparsity_pattern = Self::build_sparsity_pattern(mesh, model.num_vars_per_cell);
        Self {
            model,
            mesh,
            bc_registry,
            dt: None,
            theta: 1.0,
            s_old_cache: None,
            spatial_old_cache: None,
            sparsity_pattern,
        }
    }

    fn build_sparsity_pattern(mesh: &Mesh, num_vars_per_cell: usize) -> SparsityPattern {
        let n_cells = mesh.cells.len();
        let m = num_vars_per_cell;
        let n_dofs = n_cells * m;

        let mut indptr = Vec::with_capacity(n_dofs + 1);
        let mut indices = Vec::with_capacity(n_dofs * 7); // Rough estimate

        indptr.push(0);

        let mut neighbors = Vec::with_capacity(16);

        for cell_idx in 0..n_cells {
            neighbors.clear();
            neighbors.push(cell_idx);

            let cell = &mesh.cells[cell_idx];
            for &face_idx in &mesh.cell_face_ids[cell.face_start..cell.face_end] {
                let face = &mesh.faces[face_idx];
                match face.neighbor_cell_ids {
                    (k, Some(l)) => {
                        if k == cell_idx { neighbors.push(l); }
                        else if l == cell_idx { neighbors.push(k); }
                    }
                    _ => {}
                }
            }

            neighbors.sort_unstable();
            neighbors.dedup();

            // Same pattern for all variables in the cell
            for _ in 0..m {
                for &nb in &neighbors {
                    for w in 0..m {
                        indices.push(nb * m + w);
                    }
                }
                indptr.push(indices.len());
            }
        }

        SparsityPattern { indptr, indices }
    }

    /// Calculate the full residual vector.
    pub fn calculate_residual(&self, u: DVector<DualDVec64>) -> DVector<DualDVec64> {
        let spatial_current = self.flux_contribution(&u)
            + self.reaction_contribution(&u);
        let theta_t = DualDVec64::from(self.theta);

        let mut residual = spatial_current * theta_t.clone();

        if let Some(spation_old) = &self.spatial_old_cache {
            residual += spation_old * (DualDVec64::from(1.0) - theta_t);
        }

        if let (Some(dt), Some(s_old)) = (self.dt, &self.s_old_cache) {
            let dt_t = DualDVec64::from(dt);
            residual += (self.storage_contribution(&u) - s_old.clone()) / dt_t;
        }
        residual
    }

    /// Prepare the physics functional for a transient time step.
    pub fn prepare_time_step(&mut self, u_old: DVector<f64>, dt: f64) {
        self.dt = Some(dt);

        let u_old_t = u_old.map(|x| {
            DualDVec64::from(x)
        });

        let s_old_t = self.storage_contribution(&u_old_t);

        self.s_old_cache = Some(s_old_t);

        if self.theta < 1.0 {
            let spation_old_t = self.flux_contribution(&u_old_t)
                + self.reaction_contribution(&u_old_t);
            self.spatial_old_cache = Some(spation_old_t);
        } else {
            self.spatial_old_cache = None;
        }
    }

    /// Compute the residual contribution from all fluxes across faces.
    fn flux_contribution(&self, u: &DVector<DualDVec64>) -> DVector<DualDVec64> {
        let mesh = self.mesh;
        let num_vars = self.model.num_vars_per_cell;
        let mut residual = DVector::zeros(mesh.cells.len() * num_vars);
        let mut f_flux = DVector::from_vec(vec![DualDVec64::from_re(0.0); num_vars]);

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    let u_k = u.rows(k * num_vars, num_vars);
                    let u_l = u.rows(l * num_vars, num_vars);

                    f_flux.fill(DualDVec64::from_re(0.0));
                    (self.model.flux)(
                        f_flux.as_mut_slice(),
                        u_k.as_slice(),
                        u_l.as_slice(),
                        face,
                        &self.model.data,
                    );

                    let d = self.model.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let scale = FunctionalPhysics::<D>::face_scale(face, d);

                    for i in 0..num_vars {
                        let flux_val = f_flux[i].clone() * scale;
                        residual[k * num_vars + i] += flux_val.clone();
                        residual[l * num_vars + i] -= flux_val;
                    }
                }
                (k, None) => {
                    let Some(label) = self.model.face_tags.get(&face_idx) else {
                        continue;
                    };
                    let u_k = u.rows(k * num_vars, num_vars);

                    let actions = self.model.compute_boundary_values(
                        u_k.as_slice(),
                        face,
                        mesh.cells[k].centroid,
                        label,
                        self.bc_registry,
                    );

                    // Construct dummy boundary state for flux closure
                    let u_boundary: Vec<DualDVec64> = actions
                        .iter()
                        .zip(u_k.as_slice().iter())
                        .map(|(action, uk_val)| match action {
                            BoundaryAction::Dirichlet(val) => val.clone(),
                            BoundaryAction::InjectedFlux(_) => uk_val.clone(),
                        })
                        .collect();

                    f_flux.fill(DualDVec64::from_re(0.0));
                    (self.model.flux)(
                        f_flux.as_mut_slice(),
                        u_k.as_slice(),
                        &u_boundary,
                        face,
                        &self.model.data,
                    );

                    let d = self.model.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let scale = FunctionalPhysics::<D>::face_scale(face, d);

                    // Splice the flux
                    for i in 0..num_vars {
                        if let BoundaryAction::InjectedFlux(flux) = &actions[i] {
                            f_flux[i] = flux.clone() / scale;
                        }
                    }

                    for i in 0..num_vars {
                        residual[k * num_vars + i] += f_flux[i].clone() * scale;
                    }
                }
            }
        }
        residual
    }

    fn reaction_contribution(&self, u: &DVector<DualDVec64>) -> DVector<DualDVec64> {
        let mesh = self.mesh;
        let num_vars = self.model.num_vars_per_cell;
        let mut residual = DVector::zeros(mesh.cells.len() * num_vars);
        let mut f_reaction = DVector::from_vec(vec![DualDVec64::from_re(0.0); num_vars]);

        for cell in &mesh.cells {
            let u_cell = u.rows(cell.id * num_vars, num_vars);
            f_reaction.fill(DualDVec64::from_re(0.0));
            (self.model.reaction)(
                f_reaction.as_mut_slice(),
                u_cell.as_slice(),
                cell,
                &self.model.data,
            );
            for i in 0..num_vars {
                residual[cell.id * num_vars + i] += f_reaction[i].clone() * cell.volume;
            }
        }
        residual
    }

    pub fn storage_contribution(&self, u: &DVector<DualDVec64>) -> DVector<DualDVec64> {
        let mesh = self.mesh;
        let num_vars = self.model.num_vars_per_cell;
        let mut s_vec = DVector::zeros(mesh.cells.len() * num_vars);
        let mut f_storage = DVector::from_vec(vec![DualDVec64::from_re(0.0); num_vars]);

        for cell in &mesh.cells {
            let u_cell = u.rows(cell.id * num_vars, num_vars);
            f_storage.fill(DualDVec64::from_re(0.0));
            (self.model.storage)(
                f_storage.as_mut_slice(),
                u_cell.as_slice(),
                cell,
                &self.model.data,
            );
            for i in 0..num_vars {
                s_vec[cell.id * num_vars + i] += f_storage[i].clone() * cell.volume;
            }
        }
        s_vec
    }

    // Sparse Assembly Methods
    pub fn residual_component_row(&self, u: &[DualDVec64], r: usize) -> DualDVec64 {
        let mesh = self.mesh;
        let m = self.model.num_vars_per_cell;
        let cell_id = r / m;
        let var = r % m;

        let mut acc_spatial = DualDVec64::from_re(0.0);

        // (A) reaction
        {
            let u_cell: &[DualDVec64] = &u[(cell_id * m)..(cell_id * m + m)];
            let mut f_reaction = vec![DualDVec64::from_re(0.0); m];
            (self.model.reaction)(&mut f_reaction, u_cell, &mesh.cells[cell_id], &self.model.data);
            acc_spatial += f_reaction[var].clone() * mesh.cells[cell_id].volume;
        }

        // (B) flux
        let mut f_flux = vec![DualDVec64::from_re(0.0); m];
        let cell = &mesh.cells[cell_id];
        for face_idx in &mesh.cell_face_ids[cell.face_start..cell.face_end] {
            let face = &mesh.faces[*face_idx];
            match face.neighbor_cell_ids {
                (k, Some(l)) => { // internal face
                    let u_k: &[DualDVec64] = &u[(k * m)..(k * m + m)];
                    let u_l: &[DualDVec64] = &u[(l * m)..(l * m + m)];
                    for x in &mut f_flux { *x = DualDVec64::from_re(0.0); }
                    (self.model.flux)(&mut f_flux, u_k, u_l, face, &self.model.data);
                    let d = self.model.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let scale = FunctionalPhysics::<D>::face_scale(face, d);
                    if cell_id == k {
                        acc_spatial += f_flux[var].clone() * scale;
                    } else if cell_id == l {
                        acc_spatial -= f_flux[var].clone() * scale;
                    }
                }
                (k, None) => { // boundary face
                    let Some(label) = self.model.face_tags.get(face_idx) else { continue; };
                    let u_k: &[DualDVec64] = &u[(k * m)..(k * m + m)];
                    let actions = self.model.compute_boundary_values(u_k, face, mesh.cells[k].centroid, label, self.bc_registry);

                    let u_boundary: Vec<DualDVec64> = actions.iter().zip(u_k.iter()).map(|(action, uk_val)| {
                        match action {
                            BoundaryAction::Dirichlet(val) => val.clone(),
                            BoundaryAction::InjectedFlux(_) => uk_val.clone(),
                        }
                    }).collect();

                    for x in &mut f_flux { *x = DualDVec64::from_re(0.0); }
                    (self.model.flux)(&mut f_flux, u_k, &u_boundary, face, &self.model.data);
                    let d = self.model.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let scale = FunctionalPhysics::<D>::face_scale(face, d);

                    // Splice the flux
                    for i in 0..m {
                        if let BoundaryAction::InjectedFlux(flux) = &actions[i] {
                             f_flux[i] = flux.clone() / scale;
                        }
                    }

                    acc_spatial += f_flux[var].clone() * scale;
                }
            }
        }

        let theta_t = DualDVec64::from(self.theta);
        let mut total_residual = acc_spatial * theta_t;

        if let Some(spatial_old) = &self.spatial_old_cache {
            let one_minus_theta = DualDVec64::from(1.0 - self.theta);
            total_residual += spatial_old[r].clone() * one_minus_theta;
        }

        if let (Some(dt), Some(s_old)) = (self.dt, &self.s_old_cache) {
            let u_cell: &[DualDVec64] = &u[(cell_id * m)..(cell_id * m + m)];
            let mut f_storage = vec![DualDVec64::from_re(0.0); m];
            (self.model.storage)(&mut f_storage, u_cell, &mesh.cells[cell_id], &self.model.data);
            let s_new = f_storage[var].clone() * mesh.cells[cell_id].volume;
            let s_old_val = s_old[r].clone();
            total_residual += (s_new - s_old_val) / DualDVec64::from(dt);
        }

        total_residual
    }

    pub fn jacobian_row_locals(
        &self,
        u: &[f64],
        r: usize,
        row_indices: &[usize],
        row_data: &mut [f64],
        diag_accumulator: &mut Vec<f64>,
    ) {
        let mesh = self.mesh;
        let m = self.model.num_vars_per_cell;
        let cell_id = r / m;
        let var = r % m;

        // Helper to find the starting index for a neighbor's block of variables
        let get_block_offset = |neighbor_id: usize| -> usize {
            let start_col = neighbor_id * m;
            row_indices.binary_search(&start_col).expect("Column block not found in sparsity pattern")
        };

        // (A) reaction
        {
            let u_cell = self.seed_cell_dual(u, cell_id);
            let mut f = vec![DualDVec64::from_re(0.0); m];
            (self.model.reaction)(&mut f, &u_cell, &mesh.cells[cell_id], &self.model.data);
            let rd = f[var].clone() * mesh.cells[cell_id].volume;
            let deriv = rd.eps.unwrap_generic(Dyn(m), U1);
            for j in 0..m {
                diag_accumulator[j] += deriv[(j, 0)];
            }
        }

        // (B) flux
        let cell = &mesh.cells[cell_id];
        for &face_idx in &mesh.cell_face_ids[cell.face_start..cell.face_end] {
            let face = &mesh.faces[face_idx];
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    if k != cell_id && l != cell_id { continue; }
                    let (uk, ul) = self.seed_face_dual(u, k, l);
                    let mut f = vec![DualDVec64::from_re(0.0); m];
                    (self.model.flux)(&mut f, &uk, &ul, face, &self.model.data);
                    let d = self.model.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let mut rd = f[var].clone() * FunctionalPhysics::<D>::face_scale(face, d);
                    if cell_id == l { rd = -rd; }
                    let d_eps = rd.eps.unwrap_generic(Dyn(2 * m), U1);

                    if cell_id == k {
                        for j in 0..m { diag_accumulator[j] += d_eps[(j, 0)]; }
                        let block = d_eps.rows(m, m);
                        let block_idx = get_block_offset(l);
                        for j in 0..m {
                            row_data[block_idx + j] += block[(j, 0)] * self.theta;
                        }
                    } else if cell_id == l {
                        for j in 0..m { diag_accumulator[j] += d_eps[(m + j, 0)]; }
                        let block = d_eps.rows(0, m);
                        let block_idx = get_block_offset(k);
                        for j in 0..m {
                            row_data[block_idx + j] += block[(j, 0)] * self.theta;
                        }
                    }
                }
                (k, None) => {
                    if k != cell_id { continue; }
                    let Some(label) = self.model.face_tags.get(&face_idx) else { continue; };
                    let uk = self.seed_cell_dual(u, k);
                    let actions = self.model.compute_boundary_values(&uk, face, mesh.cells[k].centroid, label, self.bc_registry);

                    let ubc: Vec<DualDVec64> = actions.iter().zip(uk.iter()).map(|(action, uk_val)| {
                        match action {
                            BoundaryAction::Dirichlet(val) => val.clone(),
                            BoundaryAction::InjectedFlux(_) => uk_val.clone(),
                        }
                    }).collect();

                    let mut f = vec![DualDVec64::from_re(0.0); m];
                    (self.model.flux)(&mut f, &uk, &ubc, face, &self.model.data);
                    let d = self.model.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let scale = FunctionalPhysics::<D>::face_scale(face, d);

                    // Splice the flux
                    for i in 0..m {
                        if let BoundaryAction::InjectedFlux(flux) = &actions[i] {
                            f[i] = flux.clone() / scale;
                        }
                    }

                    let rd = f[var].clone() * scale;
                    let deriv = rd.eps.unwrap_generic(Dyn(m), U1);
                    for j in 0..m { diag_accumulator[j] += deriv[(j, 0)]; }
                }
            }
        }

        for j in 0..m { diag_accumulator[j] *= self.theta; }

        if let Some(dt) = self.dt {
            let u_cell = self.seed_cell_dual(u, cell_id);
            let mut f = vec![DualDVec64::from_re(0.0); m];
            (self.model.storage)(&mut f, &u_cell, &mesh.cells[cell_id], &self.model.data);
            let deriv = f[var].clone().eps.unwrap_generic(Dyn(m), U1);
            let factor = mesh.cells[cell_id].volume / dt;
            for j in 0..m { diag_accumulator[j] += deriv[(j, 0)] * factor; }
        }

        for j in 0..m {
            if diag_accumulator[j] != 0.0 {
                let col = cell_id * m + j;
                if let Ok(idx) = row_indices.binary_search(&col) {
                    row_data[idx] += diag_accumulator[j];
                } else {
                    panic!("Column {} not found in sparsity pattern for row {}", col, r);
                }
            }
        }
    }

    // Helpers
    fn seed_cell_dual(&self, u: &[f64], cell: usize) -> Vec<DualDVec64> {
        let m = self.model.num_vars_per_cell;
        (0..m).map(|j| {
            let eps = Derivative::derivative_generic(Dyn(m), U1, j);
            DualDVec64::new(u[cell * m + j], eps)
        }).collect()
    }

    fn seed_face_dual(&self, u: &[f64], left: usize, right: usize) -> (Vec<DualDVec64>, Vec<DualDVec64>) {
        let m = self.model.num_vars_per_cell;
        let mut ul = Vec::with_capacity(m);
        let mut ur = Vec::with_capacity(m);
        for j in 0..m {
            let el = Derivative::derivative_generic(Dyn(2 * m), U1, j);
            let er = Derivative::derivative_generic(Dyn(2 * m), U1, m + j);
            ul.push(DualDVec64::new(u[left * m + j], el));
            ur.push(DualDVec64::new(u[right * m + j], er));
        }
        (ul, ur)
    }
}

impl<'a, D> DiscreteModel for FvmDiscretizer<'a, D>
where
    D: 'static + Sync,
{
    fn num_variables(&self) -> usize {
        self.model.num_vars_per_cell
    }

    fn calculate_residual(&self, u: DVector<DualDVec64>) -> DVector<DualDVec64> {
        self.calculate_residual(u)
    }

    fn compute_jacobian_and_residual(&self, u: &DVector<f64>) -> (DVector<f64>, CsrMatrix<f64>) {
        let n = u.len();

        // Use pre-computed sparsity pattern
        let indptr = self.sparsity_pattern.indptr.clone();
        let indices = self.sparsity_pattern.indices.clone();

        let m = self.model.num_vars_per_cell;

        // dual representation for residual evaluation
        let u_dual: Vec<DualDVec64> = u.iter().map(|&x| DualDVec64::from_re(x)).collect();
        let u_slice = u.as_slice();

        // Parallel calculation of residuals and Jacobian rows
        let (residual_vec, data_chunks): (Vec<f64>, Vec<Vec<f64>>) = (0..n)
            .into_par_iter()
            .map(|r| {
                // Local scratch space
                let mut diag_reuse: Vec<f64> = vec![0.0; m];

                // 1. Residual Component
                let res_val = self.residual_component_row(&u_dual, r).re;

                // 2. Jacobian Row
                let row_start = indptr[r];
                let row_end = indptr[r + 1];
                let row_indices = &indices[row_start..row_end];

                // Initialize row data (local vector)
                let mut row_data = vec![0.0; row_end - row_start];

                self.jacobian_row_locals(
                    u_slice,
                    r,
                    row_indices,
                    &mut row_data,
                    &mut diag_reuse,
                );

                (res_val, row_data)
            })
            .unzip();

        // Assemble final structures
        let residual = DVector::from_vec(residual_vec);
        let data: Vec<f64> = data_chunks.into_iter().flatten().collect();

        let jacobian = CsrMatrix::from_csr(n, n, indptr, indices, data);
        (residual, jacobian)
    }
}
