use glam::DVec3;
use nalgebra::DVector;
use num_dual::DualDVec64;

use vfvm_rs::discretization::generator::create_voronoi_mesh;
use vfvm_rs::discretization::mesh::{Cell, Face, Mesh};
use vfvm_rs::numerics::sparse_aramijo::NewtonArmijoSolver;
use vfvm_rs::physics::bc::{BCRule, BoundarySelector, DirichletStyle, Field, GeneralizedBC};
use vfvm_rs::physics::functional::FunctionalPhysics;
use vfvm_rs::physics::PhysicsModel;

#[derive(Clone)]
struct DiffusionParams { k: f64 }


// The PDE
// -k * d2T/dx2 = 0
fn setup_diffusion(params: DiffusionParams) -> FunctionalPhysics<DualDVec64, DiffusionParams> {
    let flux = Box::new(|f: &mut [DualDVec64], u_k: &[DualDVec64], u_l: &[DualDVec64], _: &Face, data: &DiffusionParams| {
        f[0] = (u_l[0].clone() - u_k[0].clone()) * data.k;
    });
    let reaction = Box::new(|f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &DiffusionParams| f[0] = DualDVec64::from_re(0.0));
    let storage = Box::new(|f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &DiffusionParams| f[0] = DualDVec64::from_re(0.0));
    FunctionalPhysics::new(vec![Field::from("T")], params, flux, reaction, storage)
}

#[derive(Clone)]
struct PoissonParams { source: f64 }

// The PDE
// -d2u/dx2 = source
fn setup_poisson(params: PoissonParams) -> FunctionalPhysics<DualDVec64, PoissonParams> {
    let flux = Box::new(|f: &mut [DualDVec64], u_k: &[DualDVec64], u_l: &[DualDVec64], _: &Face, _: &PoissonParams| {
        f[0] = -(u_l[0].clone() - u_k[0].clone());
    });
    let reaction = Box::new(|f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, data: &PoissonParams| {
        f[0] = DualDVec64::from_re(-data.source);
    });
    let storage = Box::new(|f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &PoissonParams| f[0] = DualDVec64::from_re(0.0));
    FunctionalPhysics::new(vec![Field::from("u")], params, flux, reaction, storage)
}

struct SimpleModel<D: Clone + 'static> { physics: FunctionalPhysics<DualDVec64, D> }
impl<D: Clone + 'static> PhysicsModel<DualDVec64> for SimpleModel<D> {
    fn num_variables(&self) -> usize { self.physics.num_vars_per_cell }
    fn calculate_residual(&self, mesh: &Mesh, u: DVector<DualDVec64>) -> DVector<DualDVec64> { self.physics.calculate_residual(mesh, u) }
    fn apply_boundary_conditions(&mut self, _: &Mesh, _: &mut Vec<f64>) {}
}

#[test]
fn basic_verif() {
    let width = 1.0;
    let num_points = 101;
    let dx = width / num_points as f64;
    let mut generators = Vec::new();
    for i in (-(num_points) / 2)..=((num_points) / 2) {
        let x = (i as f64) * dx;
        generators.push(DVec3::new(x, 0.0, 0.0));
    }
    let mesh = create_voronoi_mesh(&generators, [width, 0.1, 0.1]);

    let mut left_face = 0; let mut right_face = 0;
    let tol = 1e-10;
    for (i, face) in mesh.faces.iter().enumerate() {
        if face.neighbor_cell_ids.1.is_none() {
            if face.centroid[0] < tol - 0.5 { left_face = i; }
            else if face.centroid[0] > 0.5 - tol { right_face = i; }
        }
    }

    let solver = NewtonArmijoSolver::default();

    println!("Test 1: Linear Diffusion");
    let mut model1 = SimpleModel { physics: setup_diffusion(DiffusionParams { k: 1.0 }) };
    model1.physics.face_tags.insert(left_face, "left".to_string());
    model1.physics.face_tags.insert(right_face, "right".to_string());
    model1.physics.bc_registry.add(BCRule { field: Field::from("T"), on: BoundarySelector::Label("left".to_string()), bc: GeneralizedBC::dirichlet(0.0), style: DirichletStyle::Strong });
    model1.physics.bc_registry.add(BCRule { field: Field::from("T"), on: BoundarySelector::Label("right".to_string()), bc: GeneralizedBC::dirichlet(100.0), style: DirichletStyle::Strong });

    let init = DVector::zeros(mesh.cells.len());
    let result1 = solver.solve(&model1.physics, &mesh, init, false).expect("Solved");

    let mut max_err = 0.0;
    for (i, val) in result1.solution.iter().enumerate() {
        let x = mesh.cells[i].centroid[0];
        let exact = 100.0 * (x + 0.5);
        if (val - exact).abs() > max_err { max_err = (val - exact).abs(); }
    }
    println!("Max Absolute Error: {:.2e}", max_err);
    // Solver is exact for linear problems, so very low threshold
    if max_err < 1e-8 { println!("  -> [PASSED]"); } else { println!("  -> [FAILED]"); }
    println!();

    println!("Test 2: Poisson Equation");
    let source_s = 10.0;
    let mut model2 = SimpleModel { physics: setup_poisson(PoissonParams { source: source_s }) };
    model2.physics.face_tags.insert(left_face, "left".to_string());
    model2.physics.face_tags.insert(right_face, "right".to_string());
    model2.physics.bc_registry.add(BCRule { field: Field::from("u"), on: BoundarySelector::Label("left".to_string()), bc: GeneralizedBC::dirichlet(0.0), style: DirichletStyle::Strong });
    model2.physics.bc_registry.add(BCRule { field: Field::from("u"), on: BoundarySelector::Label("right".to_string()), bc: GeneralizedBC::dirichlet(0.0), style: DirichletStyle::Strong });

    let init = DVector::zeros(mesh.cells.len());
    let result2 = solver.solve(&model2.physics, &mesh, init, false).expect("Solved");

    let mut max_err = 0.0;
    for (i, val) in result2.solution.iter().enumerate() {
        let x = mesh.cells[i].centroid[0];
        let exact = (source_s / 2.0) * (0.25 - x * x);
        if (val - exact).abs() > max_err { max_err = (val - exact).abs(); }
    }
    println!("Max Absolute Error: {:.2e}", max_err);
    if max_err < 5e-3 { println!("  -> [PASSED]"); } else { println!("  -> [FAILED]"); }
}
