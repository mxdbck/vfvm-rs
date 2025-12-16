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
        let m = self.num_vars;
        let cell_id = r / m;
        let var = r % m;

        let mut acc = T::zero();

        // (A) reaction/source for this cell only
        {
            let u_cell: &[T] = &u[(cell_id * m)..(cell_id * m + m)];
            let mut f_reaction = vec![T::zero(); m];
            (self.reaction)(&mut f_reaction, u_cell, &mesh.cells[cell_id], &self.data);
            // volume scaling like reaction_contribution
            acc += f_reaction[var].clone() * mesh.cells[cell_id].volume;
        }

        // (B) flux terms on faces touching this cell
        let mut f_flux = vec![T::zero(); m];
        for face_idx in &mesh.cells[cell_id].face_ids {
            let face = &mesh.faces[*face_idx];
            match face.neighbor_cell_ids {
                (k, Some(l)) => {
                    // Left/right slices
                    let u_k: &[T] = &u[(k * m)..(k * m + m)];
                    let u_l: &[T] = &u[(l * m)..(l * m + m)];

                    // compute flux
                    for x in &mut f_flux {
                        *x = T::zero();
                    }
                    (self.flux)(&mut f_flux, u_k, u_l, face, &self.data);

                    let d = self.safe_distance(mesh.cells[k].centroid, mesh.cells[l].centroid);
                    let scale = Self::face_scale(face, d);

                    if cell_id == k {
                        acc += f_flux[var].clone() * scale; // leaving k, positive
                    } else if cell_id == l {
                        acc -= f_flux[var].clone() * scale; // entering l, negative
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

                    acc += f_flux[var].clone() * scale;
                }
            }
        }

        acc
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
        let m = self.num_vars;
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
        let m = self.num_vars;
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
    /// Returns deduplicated (cols, vals) sorted by column index.
    #[inline]
    fn combine_duplicates(cols: Vec<usize>, vals: Vec<f64>) -> (Vec<usize>, Vec<f64>) {
        if cols.len() <= 1 {
            return (cols, vals);
        }

        let mut idx: Vec<usize> = (0..cols.len()).collect();
        idx.sort_unstable_by_key(|&i| cols[i]);

        let mut result_cols = Vec::with_capacity(cols.len());
        let mut result_vals = Vec::with_capacity(vals.len());

        let mut i = 0;
        while i < idx.len() {
            let col = cols[idx[i]];
            let mut sum = vals[idx[i]];
            i += 1;

            while i < idx.len() && cols[idx[i]] == col {
                sum += vals[idx[i]];
                i += 1;
            }

            result_cols.push(col);
            result_vals.push(sum);
        }

        (result_cols, result_vals)
    }

    /// Build Jacobian row r = (cell_id,var) as (cols, vals) with local AD.
    pub fn jacobian_row_locals(&self, mesh: &Mesh, u: &[f64], r: usize) -> (Vec<usize>, Vec<f64>) {
        let m = self.num_vars;
        let cell_id = r / m;
        let var = r % m;

        let mut cols: Vec<usize> = Vec::with_capacity(8 * m);
        let mut vals: Vec<f64> = Vec::with_capacity(8 * m);

        // (A) reaction/source contribution
        {
            let u_cell = self.seed_cell_dual(u, cell_id);
            let mut f = vec![DualDVec64::from_re(0.0); m];
            (self.reaction)(&mut f, &u_cell, &mesh.cells[cell_id], &self.data);

            let rd = f[var].clone() * mesh.cells[cell_id].volume;
            let deriv = rd.eps.unwrap_generic(Dyn(m), U1);
            push_block_view(&mut cols, &mut vals, cell_id, &deriv, m);
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
                    push_block_view(&mut cols, &mut vals, k, &d_eps.rows(0, m), m);
                    push_block_view(&mut cols, &mut vals, l, &d_eps.rows(m, m), m);
                }
                (k, None) => {
                    if k != cell_id {
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

                    push_block_view(&mut cols, &mut vals, k, &deriv, m);
                }
            }
        }

        Self::combine_duplicates(cols, vals)
    }
}
