use std::sync::Arc;

use num_dual::DualNum;

/// Field identifier stored as a runtime string.
#[derive(Clone, Debug)]
pub struct Field(pub Arc<str>);

impl Field {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }
}

impl<T: Into<Arc<str>>> From<T> for Field {
    fn from(name: T) -> Self {
        Field::new(name)
    }
}

#[allow(unused)]
/// Geometric point in space.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Outward pointing unit normal.
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
}

/// Shared function type for BC coefficients that may depend on time, position and normal.
pub type SFn = Arc<dyn Fn(f64, Point, Normal) -> f64 + Send + Sync>;

/// Local trait allowing convenient conversion into [`SFn`].
pub trait IntoSFn {
    fn into_sfn(self) -> SFn;
}

#[derive(Clone)]
pub struct GeneralizedBC {
    pub alpha: SFn,
    pub beta: SFn,
    pub gamma: SFn,
}

#[allow(unused)]
impl GeneralizedBC {
    pub fn dirichlet(g: impl IntoSFn) -> Self {
        Self {
            alpha: c(1.0),
            beta: c(0.0),
            gamma: g.into_sfn(),
        }
    }
    pub fn neumann(q: impl IntoSFn) -> Self {
        Self {
            alpha: c(0.0),
            beta: c(1.0),
            gamma: q.into_sfn(),
        }
    }
    pub fn robin(k: impl IntoSFn, g: impl IntoSFn) -> Self {
        let kf = k.into_sfn();
        let gf = g.into_sfn();
        Self {
            alpha: kf.clone(),
            beta: c(1.0),
            gamma: Arc::new(move |t, x, n| kf(t, x, n) * gf(t, x, n)),
        }
    }
}

/// move makes the closure capture val by value.
/// Without move, the closure would try to borrow val,
/// but val is a stack variable that goes out of scope
/// when c returns.
fn c(val: f64) -> SFn {
    Arc::new(move |_, _, _| val)
}

impl IntoSFn for f64 {
    fn into_sfn(self) -> SFn {
        c(self)
    }
}

impl<F> IntoSFn for F
where
    F: Fn(f64, Point, Normal) -> f64 + Send + Sync + 'static,
{
    fn into_sfn(self) -> SFn {
        Arc::new(self)
    }
}

#[derive(Clone)]
pub enum BoundarySelector {
    Label(String),
    #[allow(unused)]
    Id(u32),
    #[allow(unused)]
    Predicate(Arc<dyn Fn(Point, Normal) -> bool + Send + Sync>),
}

#[derive(Clone, Copy, Debug)]
pub enum DirichletStyle {
    Strong,
    #[allow(unused)]
    Weak,
}

impl Default for DirichletStyle {
    fn default() -> Self {
        DirichletStyle::Strong
    }
}

#[derive(Clone)]
pub struct BCRule {
    pub field: Field,
    pub on: BoundarySelector,
    pub bc: GeneralizedBC,
    #[allow(unused)]
    pub style: DirichletStyle,
}

#[derive(Default)]
pub struct BCRegistry {
    rules: Vec<BCRule>,
}

impl BCRegistry {
    pub fn add(&mut self, rule: BCRule) {
        self.rules.push(rule);
    }

    pub fn find_for<'a>(
        &'a self,
        field: impl AsRef<str>,
        label: &str,
        p: Point,
        n: Normal,
    ) -> Option<&'a BCRule> {
        self.rules.iter().rev().find(|r| {
            r.field.0.as_ref() == field.as_ref()
                && match &r.on {
                    BoundarySelector::Label(l) => l == label,
                    // WARNING: ID MATCHING NOT IMPLEMENTED
                    BoundarySelector::Id(_) => panic!("ID matching not implemented"),
                    BoundarySelector::Predicate(pred) => pred(p, n),
                }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_bc_by_field_and_label() {
        let mut reg = BCRegistry::default();
        reg.add(BCRule {
            field: Field::from("psi"),
            on: BoundarySelector::Label("left".into()),
            bc: GeneralizedBC::dirichlet(1.0),
            style: DirichletStyle::Strong,
        });

        let p = Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let n = Normal {
            nx: 1.0,
            ny: 0.0,
            nz: 0.0,
        };
        let rule = reg.find_for("psi", "left", p, n).expect("rule not found");
        assert_eq!(rule.field.0.as_ref(), "psi");
    }
}

#[inline]
pub fn robin_ghost_val<T: DualNum<f64>>(
    u_i: T,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> T {
    let two = T::from(2.0);
    let alpha_t = T::from(alpha);
    let beta_t = T::from(beta);
    let gamma_t = T::from(gamma);
    let delta_t = T::from(delta);
    if beta == 0.0 {
        two * (gamma_t / alpha_t) - u_i
    } else {
        let denom = alpha_t.clone() + beta_t.clone() / delta_t.clone();
        let numer = two * gamma_t - (alpha_t.clone() - beta_t.clone() / delta_t.clone()) * u_i;
        numer / denom
    }
}
