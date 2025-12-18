use crate::discretization::mesh::Mesh;
use crate::physics::functional::FunctionalPhysics;
use nalgebra::{Dyn, Matrix, Storage, U1};
use num_dual::{Derivative, DualDVec64, DualNum};

impl<T, D> FunctionalPhysics<T, D>
where
    T: nalgebra::Scalar + DualNum<f64> + num_traits::Zero,
    D: 'static,
{
    /// Compute only the r-th residual component (row kernel).
    /// r corresponds to (cell_id, var) with var in [0..num_vars).
    pub fn residual_component_row(&self, mesh: &Mesh, u: &[T], r: usize) -> T {
        let m = self.num_vars_per_cell;
        let cell_id = r / m;
        let var = r % m;

        let mut acc_spatial = T::zero();

        // (A) reaction/source for this cell only
        {
            let u_cell: &[T] = &u[(cell_id * m)..(cell_id * m + m)];
            let mut f_reaction = vec![T::zero(); m];
            (self.reaction)(&mut f_reaction, u_cell, &mesh.cells[cell_id], &self.data);
            acc_spatial += f_reaction[var].clone() * mesh.cells[cell_id].volume;
        }

        // (B) flux terms on faces touching this cell
        let mut f_flux = vec![T::zero(); m];
        for face_idx in &mesh.cells[cell_id].face_ids {
            let face = &mesh.faces[*face_idx];
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    let u_k: &[T] = &u[(k * m)..(k * m + m)];
                    let u_l: &[T] = &u[(l * m)..(l * m + m)];
                    for x in &mut f_flux {
                        *x = T::zero();
                    }
                    (self.flux)(&mut f_flux, u_k, u_l, face, &self.data);
                    let d = self.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let scale = Self::face_scale(face, d);
                    if cell_id == k {
                        acc_spatial += f_flux[var].clone() * scale; // leaving k, positive
                    } else if cell_id == l {
                        acc_spatial -= f_flux[var].clone() * scale; // entering l, negative
                    }
                }
                (k, None) => {
                    let Some(label) = self.face_tags.get(face_idx) else {
                        continue;
                    };
                    let u_k: &[T] = &u[(k * m)..(k * m + m)];
                    let ghost = self.compute_ghost_values(u_k, face, mesh.cells[k].centroid, label);
                    for x in &mut f_flux {
                        *x = T::zero();
                    }
                    (self.flux)(&mut f_flux, u_k, &ghost, face, &self.data);
                    let d = self.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let scale = Self::face_scale(face, d);
                    acc_spatial += f_flux[var].clone() * scale;
                }
            }
        }

        // Transient Simulation Residual Assembly
        // If not using a trasient simulation, the
        // following reduces to just the spatial
        // residual.

        // Add theta * Acc_Spatial
        let theta_t = T::from_f64(self.theta).unwrap();
        let mut total_residual = acc_spatial * theta_t;

        // Add (1-theta) * Old_Spatial
        if let Some(spatial_old) = &self.spatial_old_cache {
            let one_minus_theta = T::from_f64(1.0 - self.theta).unwrap();
            total_residual += spatial_old[r].clone() * one_minus_theta;
        }

        // Add Storage Term
        if let (Some(dt), Some(s_old)) = (self.dt, &self.s_old_cache) {
            let u_cell: &[T] = &u[(cell_id * m)..(cell_id * m + m)];
            let mut f_storage = vec![T::zero(); m];
            (self.storage)(&mut f_storage, u_cell, &mesh.cells[cell_id], &self.data);
            let s_new = f_storage[var].clone() * mesh.cells[cell_id].volume;
            let s_old_val = s_old[r].clone();

            total_residual += (s_new - s_old_val) / T::from_f64(dt).unwrap();
        }

        total_residual
    }
}

#[inline]
fn push_block_view<S: Storage<f64, Dyn, U1>>(
    cols: &mut Vec<usize>,
    vals: &mut Vec<f64>,
    base: usize,
    g: &Matrix<f64, Dyn, U1, S>,
    m: usize,
) {
    for j in 0..m {
        cols.push(base * m + j);
        vals.push(g[(j, 0)]);
    }
}

impl<D> FunctionalPhysics<DualDVec64, D>
where
    D: 'static,
{
    /// Seed dual numbers for a single cell (local dimension m).
    #[inline]
    fn seed_cell_dual(&self, u: &[f64], cell: usize) -> Vec<DualDVec64> {
        let m = self.num_vars_per_cell;
        (0..m)
            .map(|j| {
                let eps = Derivative::derivative_generic(Dyn(m), U1, j);
                DualDVec64::new(u[cell * m + j], eps)
            })
            .collect()
    }

    /// Seed dual numbers for an interior face (local dimension 2m).
    /// Returns (left_cell_duals, right_cell_duals).
    #[inline]
    fn seed_face_dual(
        &self,
        u: &[f64],
        left: usize,
        right: usize,
    ) -> (Vec<DualDVec64>, Vec<DualDVec64>) {
        let m = self.num_vars_per_cell;
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

    /// Combine duplicate column indices by summing their values.
    #[inline]
    fn combine_duplicates(cols: &mut Vec<usize>, vals: &mut Vec<f64>) {
        if cols.len() <= 1 {
            return;
        }

        let mut p: Vec<usize> = (0..cols.len()).collect();
        p.sort_unstable_by_key(|&i| cols[i]);

        let sorted_cols: Vec<usize> = p.iter().map(|&i| cols[i]).collect();
        let sorted_vals: Vec<f64> = p.iter().map(|&i| vals[i]).collect();

        cols.clear();
        vals.clear();

        if sorted_cols.is_empty() {
            return;
        }

        let mut curr_col = sorted_cols[0];
        let mut curr_sum = sorted_vals[0];

        for i in 1..sorted_cols.len() {
            if sorted_cols[i] == curr_col {
                curr_sum += sorted_vals[i];
            } else {
                cols.push(curr_col);
                vals.push(curr_sum);
                curr_col = sorted_cols[i];
                curr_sum = sorted_vals[i];
            }
        }
        cols.push(curr_col);
        vals.push(curr_sum);
    }

    /// Build Jacobian row r = (cell_id,var) as (cols, vals) with local AD.
    pub fn jacobian_row_locals(
        &self,
        mesh: &Mesh,
        u: &[f64],
        r: usize,
        cols: &mut Vec<usize>,
        vals: &mut Vec<f64>,
        diag_accumulator: &mut Vec<f64>,
    ) {
        let m = self.num_vars_per_cell;
        let cell_id = r / m;
        let var = r % m;

        // (A) reaction/source contribution
        {
            let u_cell = self.seed_cell_dual(u, cell_id);
            let mut f = vec![DualDVec64::from_re(0.0); m];
            (self.reaction)(&mut f, &u_cell, &mesh.cells[cell_id], &self.data);
            let rd = f[var].clone() * mesh.cells[cell_id].volume;
            let deriv = rd.eps.unwrap_generic(Dyn(m), U1);
            for j in 0..m {
                diag_accumulator[j] += deriv[(j, 0)];
            }
        }

        // (B) flux contributions on faces touching this cell
        for &face_idx in &mesh.cells[cell_id].face_ids {
            let face = &mesh.faces[face_idx];
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    if k != cell_id && l != cell_id {
                        continue;
                    }
                    let (uk, ul) = self.seed_face_dual(u, k, l);
                    let mut f = vec![DualDVec64::from_re(0.0); m];
                    (self.flux)(&mut f, &uk, &ul, face, &self.data);
                    let d = self.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let mut rd = f[var].clone() * Self::face_scale(face, d);
                    if cell_id == l {
                        rd = -rd;
                    }
                    let d_eps = rd.eps.unwrap_generic(Dyn(2 * m), U1);

                    if cell_id == k {
                        for j in 0..m {
                            diag_accumulator[j] += d_eps[(j, 0)];
                        }
                        // Off-diagonals must be scaled by theta NOW before pushing
                        // (only relevant for transient simulations)
                        let block = d_eps.rows(m, m);
                        for j in 0..m {
                            cols.push(l * m + j);
                            vals.push(block[(j, 0)] * self.theta);
                        }
                    } else if cell_id == l {
                        for j in 0..m {
                            diag_accumulator[j] += d_eps[(m + j, 0)];
                        }
                        let block = d_eps.rows(0, m);
                        for j in 0..m {
                            cols.push(k * m + j);
                            vals.push(block[(j, 0)] * self.theta);
                        }
                    }
                }
                (k, None) => {
                    if k != cell_id {
                        println!("This is kinda weird");
                        continue;
                    }
                    let Some(label) = self.face_tags.get(&face_idx) else {
                        continue;
                    };
                    let uk = self.seed_cell_dual(u, k);
                    let ubc = self.compute_ghost_values(&uk, face, mesh.cells[k].centroid, label);
                    let mut f = vec![DualDVec64::from_re(0.0); m];
                    (self.flux)(&mut f, &uk, &ubc, face, &self.data);
                    let d = self.safe_distance(face.centroid, mesh.cells[k].centroid);
                    let rd = f[var].clone() * Self::face_scale(face, d);
                    let deriv = rd.eps.unwrap_generic(Dyn(m), U1);
                    for j in 0..m {
                        diag_accumulator[j] += deriv[(j, 0)];
                    }
                }
            }
        }

        // Apply Theta Scaling to accumulated Spatial Diagonal
        // (only relevant for transient simulations)
        for j in 0..m {
            diag_accumulator[j] *= self.theta;
        }

        // (C) Storage / Time Term (Unscaled by theta)
        if let Some(dt) = self.dt {
            let u_cell = self.seed_cell_dual(u, cell_id);
            let mut f = vec![DualDVec64::from_re(0.0); m];
            (self.storage)(&mut f, &u_cell, &mesh.cells[cell_id], &self.data);
            let deriv = f[var].clone().eps.unwrap_generic(Dyn(m), U1);
            let factor = mesh.cells[cell_id].volume / dt;
            for j in 0..m {
                diag_accumulator[j] += deriv[(j, 0)] * factor;
            }
        }

        for j in 0..m {
            if diag_accumulator[j] != 0.0 {
                cols.push(cell_id * m + j);
                vals.push(diag_accumulator[j]);
            }
        }
        Self::combine_duplicates(cols, vals)
    }
}
