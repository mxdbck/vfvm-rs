use glam::DVec3;
use nalgebra::DVector;
use num_dual::DualDVec64;

use vfvm_rs::discretization::generator::create_voronoi_mesh;
use vfvm_rs::discretization::mesh::{Cell, Face, Mesh};
use vfvm_rs::discretization::fvm::FvmDiscretizer;
use vfvm_rs::numerics::nonlinear::NonlinearSolver;
use vfvm_rs::physics::bc::{BCRegistry, BCRule, BoundarySelector, DirichletStyle, Field, GeneralizedBC};
use vfvm_rs::physics::functional::{FluxFn, FunctionalPhysics, ReactionFn, StorageFn};
use vfvm_rs::system::SolverConfig;
use vfvm_rs::system::Preconditioner;

#[derive(Clone)]
struct LinearParams;

fn setup_linear(params: LinearParams) -> FunctionalPhysics<LinearParams, impl FluxFn<LinearParams>, impl ReactionFn<LinearParams>, impl StorageFn<LinearParams>> {
    let flux = 
        |f: &mut [DualDVec64], u_k: &[DualDVec64], u_l: &[DualDVec64], _face: &Face, _: &LinearParams| {
            f[0] = u_k[0].clone() - u_l[0].clone();
        };
    let reaction = |f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &LinearParams| f[0] = DualDVec64::from_re(0.0);
    let storage = |f: &mut [DualDVec64], _: &[DualDVec64], _: &Cell, _: &LinearParams| f[0] = DualDVec64::from_re(0.0);
    FunctionalPhysics::new(vec![Field::from("u")], params, flux, reaction, storage)
}

fn setup_1d_mesh(width: f64, num_points: usize) -> (Mesh, usize, usize) {
    let dx = width / num_points as f64;
    let mut generators = Vec::new();
    // Shift points to cell centers to avoid placing generators on the boundary
    let start = -(width / 2.0) + dx / 2.0;
    for i in 0..num_points {
        let x = start + (i as f64) * dx;
        generators.push(DVec3::new(x, 0.0, 0.0));
    }
    let mesh = create_voronoi_mesh(&generators, [width, 0.1, 0.1]);
    let mut left_face = 0; let mut right_face = 0;
    let tol = 1e-4;
    for (i, face) in mesh.faces.iter().enumerate() {
        if face.neighbor_cell_ids.1.is_none() {
            if face.centroid[0] < tol - 0.5 { left_face = i; }
            else if face.centroid[0] > 0.5 - tol { right_face = i; }
        }
    }
    (mesh, left_face, right_face)
}

#[test]
fn verify_neumann() {
    println!("Test: Neumann BC (Fixed Flux)");
    let (mesh, left_face, right_face) = setup_1d_mesh(1.0, 50);
    let q_flux = 5.0;

    let mut physics = setup_linear(LinearParams);
    physics.face_tags.insert(left_face, "left".to_string());
    physics.face_tags.insert(right_face, "right".to_string());

    let mut bc_registry = BCRegistry::default();
    bc_registry.add(BCRule {
        field: Field::from("u"), on: BoundarySelector::Label("left".to_string()),
        bc: GeneralizedBC::dirichlet(0.0), style: DirichletStyle::Strong,
    });

    // Right: Neumann (Flux) = 5.0 => du/dx = 5.0
    bc_registry.add(BCRule {
        field: Field::from("u"), on: BoundarySelector::Label("right".to_string()),
        bc: GeneralizedBC::neumann(q_flux), style: DirichletStyle::Strong,
    });

    let config = SolverConfig {
        tolerance: 1e-12,
        max_iterations: 100,
        preconditioner: Preconditioner::None,
        history_handler: None,
        min_step_size: 1e-6,
        armijo_param: 1e-4,
        max_step: None,
        forcing_term: 1e-3,
    };
    let mut solver = NonlinearSolver::new(config);
    let init = DVector::zeros(mesh.cells.len());
    let discretizer = FvmDiscretizer::new(&mut physics, &mesh, &bc_registry);

    let result = solver.solve(&discretizer, init).expect("Solved");

    let mut max_err = 0.0;
    for (i, val) in result.solution.iter().enumerate() {
        let x = mesh.cells[i].centroid[0];
        let exact = q_flux * (x + 0.5);
        if (val - exact).abs() > max_err { max_err = (val - exact).abs(); }
    }
    println!("Max Absolute Error: {:.2e}", max_err);
    if max_err < 1e-4 { println!("  -> [PASSED]"); } else { println!("  -> [FAILED]"); panic!("Neumann test failed"); }
}

#[test]
fn verify_robin() {
    println!("\nTest: Robin BC");
    // Domain: [-0.5, 0.5]
    // Equation: -u'' = 0 -> u(x) = Ax + B
    // BC Left:  u(-0.5) = 10.0  => -0.5A + B = 10
    // BC Right: u'(0.5) + h * u(0.5) = 0
    //           A + h(0.5A + B) = 0
    //           A(1 + 0.5h) + hB = 0

    let h_coeff = 2.0;

    // Solving the linear system for A and B:
    // 1) B = 10 + 0.5A
    // 2) A(1 + 0.5*2) + 2B = 0 => 2A + 2B = 0 => A = -B
    // Subst: B = 10 + 0.5(-B) => 1.5B = 10 => B = 20/3
    // A = -20/3

    let b_const = 10.0 / 1.5;
    let slope = -b_const;

    let (mesh, left_face, right_face) = setup_1d_mesh(1.0, 50);

    let mut physics = setup_linear(LinearParams);
    physics.face_tags.insert(left_face, "left".to_string());
    physics.face_tags.insert(right_face, "right".to_string());

    let mut bc_registry = BCRegistry::default();
    bc_registry.add(BCRule {
        field: Field::from("u"), on: BoundarySelector::Label("left".to_string()),
        bc: GeneralizedBC::dirichlet(10.0), style: DirichletStyle::Strong,
    });

    bc_registry.add(BCRule {
        field: Field::from("u"), on: BoundarySelector::Label("right".to_string()),
        bc: GeneralizedBC::robin(h_coeff, 0.0), style: DirichletStyle::Strong,
    });

    let config = SolverConfig {
        tolerance: 1e-12,
        max_iterations: 100,
        preconditioner: Preconditioner::None,
        history_handler: None,
        min_step_size: 1e-6,
        armijo_param: 1e-4,
        max_step: None,
        forcing_term: 0.1,
    };
    let mut solver = NonlinearSolver::new(config);
    let init = DVector::from_element(mesh.cells.len(), 5.0);
    let discretizer = FvmDiscretizer::new(&mut physics, &mesh, &bc_registry);

    let result = solver.solve(&discretizer, init).expect("Solved");

    let mut max_err = 0.0;
    for (i, val) in result.solution.iter().enumerate() {
        let x = mesh.cells[i].centroid[0];
        let exact = slope * x + b_const;
        if (val - exact).abs() > max_err { max_err = (val - exact).abs(); }
    }
    println!("Max Absolute Error: {:.2e}", max_err);

    if max_err < 1e-3 { println!("  -> [PASSED]"); } else { println!("  -> [FAILED]"); panic!("Robin test failed"); }
}
