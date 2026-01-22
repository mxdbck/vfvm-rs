This doc is intended to explain the logic of simulating systems using vfvm-rs and also how the code takes the model you have defined and actually numerically simulates it to better be able to interpret the results and know the limitations.

The beginning is always a PDE, as an example, we can consider the heat equation :
$$
\frac{\partial u}{\partial t}=\Delta u+S
$$
To express this in a form the library understands we must consider the form that can be seen as the "canonical form" of the library input :
$$ \frac{ \partial (\text{storage}) }{ \partial t }+\nabla \cdot (\text{flux})=\text{source}-\text{reaction}   $$
(currently the source term isn't implemented and is just absorbed into the reaction term)
In this case we see that :
$$ \text{storage}=u\quad \text{flux}=-\nabla u\quad \text{reaction}=-S $$
fits the desired form.

Ok, great, now we have the equation in standard form. What actual code can we write to make sure the library understands this? Notice first that `source`, `flux`, `S` are functions, accordingly, the library expects functions, more specifically rust closures that take the current state as input and output the value of the `storage`, `flux` and `source` terms.

Here is an example which will be discussed below it  :
```rust
let flux = Box::new(
    |f: &mut [DualDVec64],
        u_k: &[DualDVec64],
        u_l: &[DualDVec64],
        _face: &Face,
        data: &HeatParams| {
        f[0] = (u_k[0].clone() - u_l[0].clone());
    },
);

// Reaction function implementing a heat source spreading over a region of 
// radius 0.5 around the origin.
let reaction = Box::new(
    |f: &mut [DualDVec64], u: &[DualDVec64], cell: &Cell, data: &HeatParams| {
        let x = cell.centroid[0];
        if x.abs() <= 0.5 {
	        f[0] = DualDVec64::from_re(10.0);
        } else {
            f[0] = DualDVec64::from_re(0.0);
        }
    },
);

let storage = Box::new(
    |f: &mut [DualDVec64], u: &[DualDVec64], _cell: &Cell, _data: &HeatParams| {
        f[0] = u[0].clone();
    },
);
```
Things to note :
- Do not worry about the specifics to of `DualDVec64` you may abstract them as floating point values. They are used here because they allow for automatic differentiation.
- Everything is always 3D but equivalent 2D or 1D problems are easy to implement by considering simple 3D geometries.
- 

At this stage, it's important to know a few things about what the library will do with your PDE. This doc is not intended as a full explanation of the finite volume method but the main thing to note is that, as noted in the README this library implements a **Cell-Centered Finite Volume Method** on Voronoi meshes.

The spatial discretization is based on a **Two-Point Flux Approximation**. The total flux $\Gamma_{kl}$ across a face $f$ separating two cells, $k$ and $l$, is computed by the library as:
$\Gamma_{kl} = \left( \frac{A_f}{d_{kl}} \right) \times f_{\text{user}}(u_k, u_l, \text{face}, \text{data})$
Where:
* **$A_f$** is the area of the shared face $f$.
* **$d_{kl}$** is the distance between the centroids of cell $k$ and cell $l$, calculated by the library.
* **$f_{\text{user}}(\dots)$** is the user-provided `FluxFn` closure, which defines the physical behavior of the flux based on the left-state $u_k$ and right-state $u_l$.

This explain why, as one may have wondered form the example, the flux function only returns the difference between the two cell centered-values and not the full two-point gradient approximation :
$$ u'(x)= \frac{u(x-h)+u(x+h)}{2h}+\mathcal{O}(h^{2}) $$
It's because the division by the distance is done internally by library! This was mostly done to emulate VoronoiFVM.jl (and also to make the closures simpler) and seems adapted to most of the cases I have faced though I am not completely convinced it is the best approach yet.

Things left to discuss :
- Domain creation
- Boundary condition application and internal workings, current ghost-cell approach and limitations for semiconductor simulations, need for strong dirichlet implementation.
- System assembly and sparse assembly
- Solver and convergence criterion selection
