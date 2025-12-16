use crate::discretization::mesh::Mesh;
use crate::models::pn::pn::{PnJunctionModel, PnJunctionParams};
use nalgebra::DVector;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub struct SimulationSummary {
    // Mesh info
    pub num_cells: usize,
    pub num_faces: usize,
    pub num_nodes: usize,
    pub domain_extent: (f64, f64),
    pub min_cell_spacing: f64,
    pub max_cell_spacing: f64,
    pub avg_cell_volume: f64,

    // Physics info
    pub builtin_voltage: f64,
    pub left_n: f64,
    pub left_p: f64,
    pub right_n: f64,
    pub right_p: f64,

    // Tolerances
    pub min_distance_tol: f64,
    pub eps_diagonal: f64,

    // Solver info
    pub dense_iterations: Option<u32>,
    pub dense_final_residual: Option<f64>,
    pub sparse_iterations: Option<u32>,
    pub sparse_final_residual: Option<f64>,
    pub max_solution_diff: Option<f64>,
    pub mean_solution_diff: Option<f64>,

    // Normalization scales
    pub v_scale: f64,
    pub l_scale: f64,
    pub n_scale: f64,
}

impl SimulationSummary {
    pub fn from_problem(mesh: &Mesh, params: &PnJunctionParams, model: &PnJunctionModel) -> Self {
        let num_cells = mesh.cells.len();
        let num_faces = mesh.faces.len();
        let num_nodes = mesh.nodes.len();

        let x_coords: Vec<f64> = mesh.nodes.iter().map(|n| n.position[0]).collect();
        let x_min = x_coords.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut spacings = Vec::new();
        for face in &mesh.faces {
            if let (k, Some(l)) = face.neighbor_cell_ids {
                let d = ((mesh.cells[k].centroid[0] - mesh.cells[l].centroid[0]).powi(2)
                    + (mesh.cells[k].centroid[1] - mesh.cells[l].centroid[1]).powi(2)
                    + (mesh.cells[k].centroid[2] - mesh.cells[l].centroid[2]).powi(2))
                .sqrt();
                spacings.push(d);
            }
        }

        let min_spacing = spacings.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_spacing = spacings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_volume = mesh.cells.iter().map(|c| c.volume).sum::<f64>() / num_cells as f64;

        let psi_l_eq = calculate_equilibrium_psi(-params.na_norm, params.ni_norm);
        let psi_r_eq = calculate_equilibrium_psi(params.nd_norm, params.ni_norm);
        let builtin_v = (psi_l_eq - psi_r_eq) * params.v_scale;

        // Boundary carrier concentrations (physical units)
        let left_n = params.ni_norm * (-psi_l_eq).exp() * params.n_scale;
        let left_p = params.na_norm * params.n_scale;
        let right_n = params.nd_norm * params.n_scale;
        let right_p = params.ni_norm * psi_r_eq.exp() * params.n_scale;

        Self {
            num_cells,
            num_faces,
            num_nodes,
            domain_extent: (x_min, x_max),
            min_cell_spacing: min_spacing,
            max_cell_spacing: max_spacing,
            avg_cell_volume: avg_volume,
            builtin_voltage: builtin_v,
            left_n,
            left_p,
            right_n,
            right_p,
            min_distance_tol: model.functional.tolerances.min_distance,
            eps_diagonal: model.functional.tolerances.eps_diagonal,
            dense_iterations: None,
            dense_final_residual: None,
            sparse_iterations: None,
            sparse_final_residual: None,
            max_solution_diff: None,
            mean_solution_diff: None,
            v_scale: params.v_scale,
            l_scale: params.l_scale,
            n_scale: params.n_scale,
        }
    }

    pub fn add_dense_solver_info(&mut self, iterations: u32, final_residual: f64) {
        self.dense_iterations = Some(iterations);
        self.dense_final_residual = Some(final_residual);
    }

    pub fn add_sparse_solver_info(&mut self, iterations: u32, final_residual: f64) {
        self.sparse_iterations = Some(iterations);
        self.sparse_final_residual = Some(final_residual);
    }

    pub fn add_comparison(&mut self, dense: &DVector<f64>, sparse: &DVector<f64>) {
        let diffs: Vec<f64> = dense
            .iter()
            .zip(sparse.iter())
            .map(|(d, s)| (d - s).abs())
            .collect();

        self.max_solution_diff = Some(diffs.iter().cloned().fold(0.0, f64::max));
        self.mean_solution_diff = Some(diffs.iter().sum::<f64>() / diffs.len() as f64);
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "{}", "=".repeat(60))?;
        writeln!(file, "PN JUNCTION SIMULATION SUMMARY")?;
        writeln!(file, "{}", "=".repeat(60))?;
        writeln!(file)?;

        writeln!(file, "MESH STATISTICS")?;
        writeln!(file, "{}", "-".repeat(60))?;
        writeln!(file, "Number of cells:     {}", self.num_cells)?;
        writeln!(file, "Number of faces:     {}", self.num_faces)?;
        writeln!(file, "Number of nodes:     {}", self.num_nodes)?;
        writeln!(
            file,
            "Domain extent:       {:.6e} to {:.6e} (normalized)",
            self.domain_extent.0, self.domain_extent.1
        )?;
        writeln!(
            file,
            "Physical extent:     {:.6e} to {:.6e} cm",
            self.domain_extent.0 * self.l_scale,
            self.domain_extent.1 * self.l_scale
        )?;
        writeln!(
            file,
            "Min cell spacing:    {:.6e} (normalized)",
            self.min_cell_spacing
        )?;
        writeln!(
            file,
            "Max cell spacing:    {:.6e} (normalized)",
            self.max_cell_spacing
        )?;
        writeln!(file, "Avg cell volume:     {:.6e}", self.avg_cell_volume)?;
        writeln!(file)?;

        writeln!(file, "NORMALIZATION SCALES")?;
        writeln!(file, "{}", "-".repeat(60))?;
        writeln!(file, "Voltage scale (V_T): {:.6e} V", self.v_scale)?;
        writeln!(file, "Length scale (L_D):  {:.6e} cm", self.l_scale)?;
        writeln!(file, "Conc. scale (N_max): {:.6e} cm⁻³", self.n_scale)?;
        writeln!(file)?;

        writeln!(file, "NUMERICAL TOLERANCES")?;
        writeln!(file, "{}", "-".repeat(60))?;
        writeln!(
            file,
            "Min distance:        {:.6e} (normalized)",
            self.min_distance_tol
        )?;
        writeln!(
            file,
            "  = {:.6e} × min_cell_spacing",
            self.min_distance_tol / self.min_cell_spacing
        )?;
        writeln!(
            file,
            "  = {:.6e} cm (physical)",
            self.min_distance_tol * self.l_scale
        )?;
        writeln!(file, "Diagonal epsilon:    {:.6e}", self.eps_diagonal)?;
        writeln!(file)?;

        writeln!(file, "PHYSICS RESULTS")?;
        writeln!(file, "{}", "-".repeat(60))?;
        writeln!(file, "Built-in voltage:    {:.4} V", self.builtin_voltage)?;
        writeln!(file)?;
        writeln!(file, "Left boundary (p-side):")?;
        writeln!(file, "  n = {:.2e} cm⁻³", self.left_n)?;
        writeln!(file, "  p = {:.2e} cm⁻³", self.left_p)?;
        writeln!(file)?;
        writeln!(file, "Right boundary (n-side):")?;
        writeln!(file, "  n = {:.2e} cm⁻³", self.right_n)?;
        writeln!(file, "  p = {:.2e} cm⁻³", self.right_p)?;
        writeln!(file)?;

        if self.dense_iterations.is_some() || self.sparse_iterations.is_some() {
            writeln!(file, "SOLVER PERFORMANCE")?;
            writeln!(file, "{}", "-".repeat(60))?;

            if let (Some(iter), Some(res)) = (self.dense_iterations, self.dense_final_residual) {
                writeln!(file, "Dense solver:")?;
                writeln!(file, "  Iterations:        {}", iter)?;
                writeln!(file, "  Final residual:    {:.6e}", res)?;
            }

            if let (Some(iter), Some(res)) = (self.sparse_iterations, self.sparse_final_residual) {
                writeln!(file, "Sparse solver:")?;
                writeln!(file, "  Iterations:        {}", iter)?;
                writeln!(file, "  Final residual:    {:.6e}", res)?;
            }
            writeln!(file)?;
        }

        // Solution comparison
        if let (Some(max_diff), Some(mean_diff)) = (self.max_solution_diff, self.mean_solution_diff)
        {
            writeln!(file, "SOLVER COMPARISON")?;
            writeln!(file, "{}", "-".repeat(60))?;
            writeln!(file, "Max difference:      {:.6e}", max_diff)?;
            writeln!(file, "Mean difference:     {:.6e}", mean_diff)?;
            writeln!(file)?;
        }

        writeln!(file, "{}", "=".repeat(60))?;

        Ok(())
    }

    pub fn print_to_console(&self) {
        println!("\n{}", "=".repeat(60));
        println!("SIMULATION SUMMARY");
        println!("{}", "=".repeat(60));
        println!(
            "Mesh:          {} cells, {} nodes",
            self.num_cells, self.num_nodes
        );
        println!("Built-in V:    {:.4} V", self.builtin_voltage);
        println!(
            "Tolerance:     {:.3e} (= {:.2e} × spacing)",
            self.min_distance_tol,
            self.min_distance_tol / self.min_cell_spacing
        );
        if let (Some(d_iter), Some(s_iter)) = (self.dense_iterations, self.sparse_iterations) {
            println!("Iterations:    dense={}, sparse={}", d_iter, s_iter);
        }
        if let Some(max_diff) = self.max_solution_diff {
            println!("Max diff:      {:.3e}", max_diff);
        }
        println!("{}\n", "=".repeat(60));
    }
}

fn calculate_equilibrium_psi(c: f64, ni: f64) -> f64 {
    let a = ni;
    let b = -c;
    let cc = -ni;
    let disc = b * b - 4.0 * a * cc;
    let x = (-b + disc.sqrt()) / (2.0 * a);
    x.ln()
}
