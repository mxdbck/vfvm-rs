/// The complete computational grid.
pub struct Mesh {
    pub cells: Vec<Cell>,
    pub faces: Vec<Face>,
    pub nodes: Vec<Node>,
}

/// A single control volume (a Voronoi cell).
pub struct Cell {
    pub id: usize,
    pub volume: f64,
    pub centroid: [f64; 3],
    pub face_ids: Vec<usize>,
}

/// An interface between two cells.
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

pub struct Node {
    // pub id: usize,
    pub position: [f64; 3],
}
