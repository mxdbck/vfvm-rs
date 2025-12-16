use std::env;

use nalgebra::DVector;
use num_dual::{DualDVec64, DualNum, jacobian};

fn m_to_n(m: usize, n: usize, vec: DVector<DualDVec64>) -> DVector<DualDVec64> {
    let mut result = DVector::zeros(n);
    for i in 0..n.max(m) {
        if i > 0 {
            result[i % n] += vec[i].powi((i + 1) as i32) * vec[i - 1].powi(i as i32);
            continue;
        }
        result[i] += vec[i].powi((i + 1) as i32);
    }
    result
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let m: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut init_vec = vec![1.0; n];
    for i in 0..n {
        init_vec[i] = (i + 1) as f64;
    }

    let xy = DVector::from(init_vec);
    let (f, jac) = jacobian(|arg| m_to_n(m, n, arg), xy);
    print!("{}\n", f);
    print!("{}\n", jac);
}
