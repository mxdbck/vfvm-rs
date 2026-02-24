use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod discretization;
pub mod models;
pub mod numerics;
pub mod physics;
pub mod processing;
pub mod system;
