The goal for this example was to visualize the evolution of a quantum particle through a grid which I have seen some people do before and tended to give some nice patterns. We start with the time-depended Schrödinger equation :

$$ i \hbar \frac{\partial \psi}{\partial t}=-\frac{\hbar^2}{2 m} \nabla^2 \psi+V(\mathbf{r}) \psi $$

The library requires all state variables to be real, so we can split the complex quantum wave into a real and imaginary part $\psi=u+iv$. Substituting :

$$ i\hbar (\dot{u}+i\dot{v})=-\frac{\hbar^{2}}{2m}\nabla^{2}(u+iv)+V(u+iv) $$

We split the real and imaginary parts giving us our two equations :

$$ \dot{u}\hbar+\frac{\hbar^{2}}{2m}\nabla^{2}v-Vv=0 $$

$$  -\dot{v}\hbar+\frac{\hbar^{2}}{2m}\nabla^{2}u-Vu=0  $$

The library follows VoronoiFVM.jl's conventions. Which I'll summarize here as:

$$ \frac{ \partial (\text{storage}) }{ \partial t }+\nabla \cdot (\text{flux})=\text{storage}-\text{reaction}   $$

(though we include the storage term into the reaction term).
Putting our equation in the required forms we get :

$$  \dot{u}+\nabla  \cdot \left( \frac{\hbar}{2m}\nabla v \right)=-(-Vv)  $$

$$  \dot{v}+\nabla  \cdot \left( -\frac{\hbar}{2m}\nabla u \right)=-(Vv)  $$
