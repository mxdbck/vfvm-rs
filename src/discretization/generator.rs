use super::mesh::{Cell, Face, Mesh, Node};
use glam::DVec3;
use meshless_voronoi::{Dimensionality, Voronoi};

/// Build the raw Voronoi diagram using the external library.
pub fn build_voronoi(generators: &[DVec3], width: [f64; 3]) -> Voronoi {
    Voronoi::build(
        generators,
        [-width[0] / 2.0, -width[1] / 2.0, -width[2] / 2.0].into(),
        width.into(),
        Dimensionality::ThreeD,
        false,
    )
}

/// Convert a `Voronoi` diagram into the library's [`Mesh`] representation.
pub fn parse_voronoi(voronoi: &Voronoi, generators: &[DVec3]) -> Mesh {
    let mut cells = Vec::new();
    let mut faces = Vec::new();
    let mut nodes = Vec::new();

    for (cell_id, cell) in voronoi.cells().into_iter().enumerate() {
        cells.push(Cell {
            id: cell_id,
            volume: cell.volume(),
            centroid: cell.centroid().to_array(),
            face_ids: cell.face_indices(voronoi).to_vec(),
        });
    }

    for face in voronoi.faces().into_iter() {
        faces.push(Face {
            area: face.area(),
            normal: face.normal().to_array(),
            neighbor_cell_ids: (face.left(), face.right()),
            centroid: face.centroid().to_array(),
        });
    }

    for node in generators.iter() {
        nodes.push(Node {
            position: node.to_array(),
        });
    }

    Mesh {
        cells,
        faces,
        nodes,
    }
}

/// Convenience wrapper that builds and immediately parses a Voronoi mesh.
pub fn create_voronoi_mesh(generators: &[DVec3], width: [f64; 3]) -> Mesh {
    let voronoi = build_voronoi(generators, width);
    parse_voronoi(&voronoi, generators)
}

/// Create a flat 3D Voronoi mesh from a 2D point cloud.
/// The mesh will have a single cell thickness in the z-direction.
pub fn create_flat_3d_mesh(points_2d: &[(f64, f64)], width: [f64; 2], thickness: f64) -> Mesh {
    let generators: Vec<DVec3> = points_2d
        .iter()
        .map(|(x, y)| DVec3::new(*x - width[0] / 2.0, *y - width[1] / 2.0, 0.0))
        .collect();

    let width_3d = [width[0], width[1], thickness];
    create_voronoi_mesh(&generators, width_3d)
}

/// Create a regular 2D grid of points for testing/simple cases.
/// Returns points in the range [0, width[0]] × [0, width[1]].
pub fn create_regular_2d_grid(width: [f64; 2], nx: usize, ny: usize) -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let dx = width[0] / nx as f64;
    let dy = width[1] / ny as f64;

    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 + 0.5) * dx;
            let y = (j as f64 + 0.5) * dy;
            points.push((x, y));
        }
    }

    points
}
