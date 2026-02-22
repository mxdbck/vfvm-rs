/// The complete computational grid.
#[derive(Clone)]
pub struct Mesh {
    pub cells: Vec<Cell>,
    pub faces: Vec<Face>,
    pub nodes: Vec<Node>,
    /// Stores all face indices for cells in a single contiguous array.
    /// Use `cell.face_start..cell.face_end` to slice into this array.
    pub cell_face_ids: Vec<usize>,
}

/// A single control volume (a Voronoi cell).
#[derive(Clone)]
pub struct Cell {
    pub id: usize,
    pub volume: f64,
    pub centroid: [f64; 3],
    /// Start index in `Mesh.cell_face_ids`
    pub face_start: usize,
    /// End index in `Mesh.cell_face_ids`
    pub face_end: usize,
}

/// An interface between two cells.
#[derive(Clone)]
pub struct Face {
    // pub id: usize,
    pub area: f64,
    pub normal: [f64; 3],
    /// Tuple of (cell1_id, optional cell2_id). `None` indicates a boundary face.
    /// In the meshless_voronoi crate, these are called the left and right faces.
    /// Only the right face can be None if it is a boundary face.
    pub neighbor_cell_ids: (usize, Option<usize>),
    pub centroid: [f64; 3],
}

#[derive(Clone)]
pub struct Node {
    // pub id: usize,
    pub position: [f64; 3],
}
