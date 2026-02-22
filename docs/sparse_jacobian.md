The goal of the `jacobian_row_locals` function is to compute a single row of the Jacobian matrix for the solver. Because we are solving a non-linear system of equations $R(u) = 0$ using Newton's method, we need the exact partial derivatives of our residuals with respect to every state variable.

Following the library's conventions, the continuous physics equation:
$$
\frac{ \partial (\text{storage}) }{ \partial t }+\nabla \cdot (\text{flux})+\text{reaction}=0
$$
Is discretized over a control volume (cell) $K$ for a specific variable $v$. The total discrete residual $R_{K,v}$ at the new time step evaluates to:
$$
R_{K,v}(u) = \left( \frac{S_{K,v}(u) - S_{K,v}(u_{old})}{\Delta t} \right) |K| + \theta \left[ \sum_{L \in \mathcal{N}(K)} F_{KL,v}(u_K, u_L) \frac{|KL|}{d_{KL}} + Y_{K,v}(u) |K| \right] + (1-\theta)[...]
$$
Where:
- $S$, $F$, and $Y$ represent the user-defined `storage`, `flux`, and `reaction` closures.
- $|K|$ is the cell volume, $|KL|$ is the face area, and $d_{KL}$ is the distance between cell centroids.
- $\theta$ is the time-stepping weight (e.g., $1.0$ for Backward Euler).

To solve this, the Jacobian matrix $J$ requires the partial derivative of this residual row with respect to every variable $w$ in every cell $M$:
$$
J_{(K,v), (M,w)} = \frac{\partial R_{K,v}}{\partial u_{M,w}}
$$
Instead of forcing the user to derive these complex partial derivatives by hand, the library uses automatic differentiation via dual numbers ($x + \epsilon \frac{\partial f}{\partial x}$).

Here is how `jacobian_row_locals` does this in the code:

#### 1. Mapping the Matrix Row to the Mesh
The Jacobian is a flat 2D matrix, but our mesh has $N$ cells, each with $m$ variables. We first map the current matrix row `r` back to its physical meaning:

```rust
let cell_id = r / m; // The cell K
let var = r % m;     // The variable v
```

#### 2. The Reaction Contribution
$$
\frac{\partial}{\partial u_{K,w}} \left( Y_{K,v}(u) |K| \right)
$$
We "seed" the state variables of cell $K$ as dual numbers, meaning we attach an $\epsilon$ to them to track their derivatives. We pass these dual numbers into the user's `reaction` closure. The closure evaluates the reaction formula, and the output's $\epsilon$ part automatically contains the exact derivatives with respect to all $m$ variables in the cell. We extract these and add them to our `diag_accumulator`.

#### 3. The Flux Contribution
$$
\frac{\partial}{\partial u_{M,w}} \left( \sum_{L} F_{KL,v}(u_K, u_L) \frac{|KL|}{d_{KL}} \right)
$$
This is the most complex part because the flux across a face depends on the state of _both_ the left cell ($K$) and the right cell ($L$).

- We loop over all faces of cell $K$.
- For internal faces, we seed a combined dual array for both cells $K$ and $L$ so that the automatic differentiation tracks derivatives with respect to $2m$ variables simultaneously.
- We call the `flux` closure and multiply the result by the face's geometric scale factor $\frac{|KL|}{d_{KL}}$.
- The resulting derivatives are split: the derivatives with respect to $K$'s variables are added to the diagonal block, and the derivatives with respect to $L$'s variables are pushed into the off-diagonal columns representing the neighbor cell.

#### 4. The Storage Contribution (Transient)
$$
\frac{\partial}{\partial u_{K,w}} \left( \frac{S_{K,v}(u)}{\Delta t} |K| \right)
$$
If the simulation is transient ($\Delta t$ exists), we seed the cell variables again and pass them to the `storage` closure. The extracted derivatives are scaled by $\frac{|K|}{\Delta t}$ and added to the diagonal accumulator.

#### 5. Final Assembly
Finally, all the accumulated diagonal values (which had to be collected separately to account for the sum of all neighbor fluxes, reactions, and storage) are multiplied by the time-integration factor $\theta$ and injected into the CSR `row_data`.
