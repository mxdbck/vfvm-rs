use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::DVec3;
use vfvm_rs::discretization::generator::{build_voronoi, parse_voronoi};
use vfvm_rs::models::pn::pn::{PnJunctionModel, pn_problem_def};
use vfvm_rs::numerics::solver::NewtonSolver;
use vfvm_rs::numerics::sparse::SparseNewtonSolver;
use vfvm_rs::physics::PhysicsModel;

fn problem_sizes() -> Vec<u32> {
    vec![300, 1000]
}

fn solver_sizes() -> Vec<u32> {
    vec![100, 300]
}

fn bench_dense_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_jacobian");
    for &size in &problem_sizes() {
        let (mesh, params) = pn_problem_def(1.0, size, false);
        let mut phys = PnJunctionModel::new(params, -1.0, false).with_mesh(&mesh);
        phys.apply_boundary_conditions(&mesh, &mut vec![]);
        let init = phys.initial_condition(&mesh);
        let solver = NewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter(|| {
                let (_res, jac) = solver.compute_residual_and_jacobian(&phys, &mesh, &init);
                std::hint::black_box(jac);
            });
        });
    }
    group.finish();
}

fn bench_sparse_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_jacobian");
    for &size in &problem_sizes() {
        let (mesh, params) = pn_problem_def(1.0, size, false);
        let mut phys = PnJunctionModel::new(params.clone(), -1.0, false).with_mesh(&mesh);
        phys.apply_boundary_conditions(&mesh, &mut vec![]);
        let init = phys.initial_condition(&mesh);
        let solver = SparseNewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter(|| {
                let (_res, jac) =
                    solver.compute_residual_and_jacobian(&phys.functional, &mesh, &init);
                std::hint::black_box(jac.nnz());
            });
        });
    }
    group.finish();
}

fn bench_voronoi_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi_build");
    for &size in &problem_sizes() {
        let generators: Vec<DVec3> = (0..size)
            .map(|i| {
                let half = size as f64 / 2.0;
                let x = i as f64 - half;
                let y = ((i * 7) % size) as f64 - half;
                let z = ((i * 13) % size) as f64 - half;
                DVec3::new(x, y, z)
            })
            .collect();
        let span = size as f64 + 1.0;
        let width = [span, span, span];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter(|| {
                let v = build_voronoi(std::hint::black_box(&generators), width);
                std::hint::black_box(v);
            });
        });
    }
    group.finish();
}

fn bench_voronoi_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi_parse");
    for &size in &problem_sizes() {
        let generators: Vec<DVec3> = (0..size)
            .map(|i| {
                let half = size as f64 / 2.0;
                let x = i as f64 - half;
                let y = ((i * 7) % size) as f64 - half;
                let z = ((i * 13) % size) as f64 - half;
                DVec3::new(x, y, z)
            })
            .collect();
        let span = size as f64 + 1.0;
        let width = [span, span, span];
        let voronoi = build_voronoi(&generators, width);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter(|| {
                let mesh = parse_voronoi(std::hint::black_box(&voronoi), &generators);
                std::hint::black_box(mesh);
            });
        });
    }
    group.finish();
}

fn bench_dense_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_solver");
    for &size in &solver_sizes() {
        let (mesh, params) = pn_problem_def(1.0, size, false);
        let mut phys = PnJunctionModel::new(params, -1.0, false).with_mesh(&mesh);
        phys.apply_boundary_conditions(&mesh, &mut vec![]);
        let init = phys.initial_condition(&mesh);
        let solver = NewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter_batched(
                || init.clone(),
                |u| {
                    let _ = solver.solve(&phys, &mesh, u, false);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_sparse_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_solver");
    for &size in &solver_sizes() {
        let (mesh, params) = pn_problem_def(1.0, size, false);
        let mut phys = PnJunctionModel::new(params, -1.0, false).with_mesh(&mesh);
        phys.apply_boundary_conditions(&mesh, &mut vec![]);
        let init = phys.initial_condition(&mesh);
        let solver = SparseNewtonSolver {
            tolerance: 1e-8,
            max_iterations: 1,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
            b.iter_batched(
                || init.clone(),
                |u| {
                    let _ = solver.solve(&phys.functional, &mesh, u, false);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dense_jacobian,
    bench_sparse_jacobian,
    bench_voronoi_build,
    bench_voronoi_parse,
    bench_dense_solver,
    bench_sparse_solver
);
criterion_main!(benches);
