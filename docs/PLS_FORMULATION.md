# PLS Cosserat formulation

This document uses the notation of *Piecewise Linear Strain Cosserat Model for Soft Slender Manipulator*. The integration rules below describe this implementation.

## Strain and coordinates

Let $X\in[0,L]$ denote the material coordinate. Section $n$ spans $[L_{n-1},L_n]$, with length $\ell_n=L_n-L_{n-1}$. The body strain is

$$
\xi(X)=\begin{bmatrix}K(X)\\Q(X)\end{bmatrix}.
$$

$K$ contains torsion and bending strains; $Q$ contains axial and shear strains. A straight, unstretched rod has $K=0$ and $Q=\mathbf e_1=[1,0,0]^T$. The nodal vector for $N$ sections is

$$
q=\begin{bmatrix}\xi_0^T&\xi_1^T&\cdots&\xi_N^T\end{bmatrix}^T.
$$

SOFA stores the Vec6 coordinates $z_i=[K_i^T,(Q_i-\mathbf e_1)^T]^T$. Thus $q=z+q_{\mathrm{ref}}$, where each block of $q_{\mathrm{ref}}$ is $[0,0,0,1,0,0]^T$, and $\dot q=\dot z$. The material rest strain $\xi^0$ is specified separately through rest data. Vec3 stores only $K$ and fixes $Q=\mathbf e_1$.

## Linear interpolation

For $X\in[L_{n-1},L_n]$,

$$
\alpha_n(X)=\frac{L_n-X}{\ell_n},\qquad
\beta_n(X)=\frac{X-L_{n-1}}{\ell_n},
$$

$$
\xi(X)=\alpha_n(X)\xi_{n-1}+\beta_n(X)\xi_n=\Phi(X)q.
$$

The two nonzero blocks of $\Phi$ are $\alpha_nI_6$ and $\beta_nI_6$. Stacking the sampling rows gives the sparse interpolation matrix $H$:

$$
\xi_s=Hq,\qquad z_s=Hz,\qquad
\dot z_s=H\dot z,\qquad\ddot z_s=H\ddot z.
$$

Sampling states introduce no independent degrees of freedom. For fixed material coordinates, $H$ is independent of the strain. Virtual work gives the accumulating force transpose:

$$
\delta W=f_s^T\delta z_s=(H^Tf_s)^T\delta z,\qquad
f_z\mathrel{+}=H^Tf_s.
$$

Constraint rows are multiplied by $H$ in the same way. `applyRestPosition` interpolates rest data using the same weights.

## Elastic energy and quadrature

Using the sectional stiffness $\Sigma$ and $\Delta\xi=\xi-\xi^0$,

$$
U_n=\frac12\int_{L_{n-1}}^{L_n}\Delta\xi^T\Sigma(X)\Delta\xi\,dX.
$$

For constant $\Sigma_n$, let $a=\xi_{n-1}-\xi^0_{n-1}$ and $b=\xi_n-\xi^0_n$. Then

$$
U_n=\frac{\ell_n}{6}(a^T\Sigma_na+a^T\Sigma_nb+b^T\Sigma_nb),
$$

$$
K_n^{\mathrm{el}}=\frac{\ell_n}{6}
\begin{bmatrix}2\Sigma_n&\Sigma_n\\\Sigma_n&2\Sigma_n\end{bmatrix},\qquad
f_n^{\mathrm{el}}=-K_n^{\mathrm{el}}\begin{bmatrix}a\\b\end{bmatrix}.
$$

$K_n^{\mathrm{el}}$ denotes element material stiffness, distinct from angular strain $K$. The derivative of the internal force is $-K_n^{\mathrm{el}}$.

`BeamHookeLawForceField` evaluates the material at two Gauss points per section:

$$
X_{n,\pm}=L_{n-1}+\ell_n\left(\frac12\pm\frac{1}{2\sqrt3}\right),\qquad
w_{n,\pm}=\frac{\ell_n}{2}.
$$

Weights are supplied through the force field's `length` data. Two-point quadrature exactly integrates this quadratic energy for constant sectional stiffness. Varying sections use numerical quadrature.

For a circular section of radius $r$,

$$
A=\pi r^2,\quad J_y=J_z=\frac{\pi r^4}{4},\quad J_x=J_y+J_z,\quad
G=\frac{E}{2(1+\nu)},
$$

$$
\Sigma=\operatorname{diag}(GJ_x,EJ_y,EJ_z,EA,GA,GA).
$$

Lowercase $r$ denotes radius to distinguish it from the rotation matrix $R$.

## Geometry and Jacobian

With centerline position $u$ and rotation $R$,

$$
g(X)=\begin{bmatrix}R(X)&u(X)\\0^T&1\end{bmatrix},\qquad
g'(X)=g(X)\widehat\xi(X),\qquad
\widehat\xi=\begin{bmatrix}\widetilde K&Q\\0^T&0\end{bmatrix},
$$

where $\widetilde K v=K\times v$. Each section is subdivided into geometry intervals. The implementation uses exponential midpoint integration:

$$
g(X_{j+1})=g(X_j)\exp\!\left(\Delta X_j\widehat\xi(X_{j+1/2})\right).
$$

`ELEMENT_COUNT` controls the number of PLS sections; `INTEGRATION_SUBSTEPS` controls the geometry intervals per section.

For a fixed base, the paper expresses body velocity as $\eta=[\Omega^T,V^T]^T=J(X)\dot q$. SOFA composes the geometry and interpolation mappings by the chain rule and expresses output velocities in its rigid-body convention. Base motion is handled by a separate rigid input.

The Jacobian is updated from the current geometry and used for velocity propagation and force transfer. The implicit solve does not include the complete geometric stiffness term $(\delta J)^Tf$. The PLS branch disables the original mapping's partial contribution so that force linearization and matrix assembly use a consistent approximation. A complete geometric Hessian is not implemented.
