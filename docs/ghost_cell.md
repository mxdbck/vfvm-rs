Place a ghost cell mirrored across the face. Let:
- $u_i=$ interior cell value,
- $u_g=$ ghost value we'll choose,
- $\Delta=\mathbf{n} \cdot\left(\mathbf{x}_f-\mathbf{x}_i\right)$ (distance from interior center to face projected on the normal, assuming outward normal this should be positive),
- face value $u_f \approx \frac{1}{2}\left(u_i+u_g\right)$,
- normal derivative $\partial_n u \approx\left(u_g-u_i\right) /(2 \Delta)$.

Enforce $\alpha u_f+\beta \partial_n u=\gamma$ at the face and solve for $u_g$ :

$$
\begin{gathered}
\alpha \frac{u_i+u_g}{2}+\beta \frac{u_g-u_i}{2 \Delta}=\gamma \\
\Rightarrow \quad u_g=\frac{2 \gamma-(\alpha-\beta / \Delta) u_i}{\alpha+\beta / \Delta} .
\end{gathered}
$$
