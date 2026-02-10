# Mathematical Theory

This page summarizes the theory implemented in `Achedmy.jl` using the notation of the accompanying manuscript. It includes the chemical reaction network definition, the path-integral formulation, effective fields from (extended) Plefka expansion, and the mean/response/correlation update equations solved numerically in the package.

## 1. Chemical reaction networks and the stochastic dynamics

We consider `P` species with copy-number state

```math
\mathbf{n}(\tau) = (n_1(\tau),\ldots,n_P(\tau)).
```

A general reaction `\beta` is

```math
\sum_{i=1}^{P} r_i^\beta X_i \xrightarrow{k_\beta(\tau)} \sum_{i=1}^{P} s_i^\beta X_i,
```

with reactant stoichiometry `r_i^\beta`, product stoichiometry `s_i^\beta`, and stoichiometric matrix entries

```math
S_{i\beta}=s_i^\beta-r_i^\beta.
```

For well-mixed dynamics, the microscopic propensity is

```math
f_\beta(\mathbf{n},\tau)=k_\beta(\tau)\prod_i\frac{n_i!}{(n_i-r_i^\beta)!}.
```

The probability mass function `P(\mathbf{n},\tau)` obeys the chemical master equation (CME):

```math
\frac{\partial P(\mathbf{n},\tau)}{\partial \tau}
=\sum_\beta f_\beta(\mathbf{n}-\mathbf{s}^\beta+\mathbf{r}^\beta,\tau)P(\mathbf{n}-\mathbf{s}^\beta+\mathbf{r}^\beta,\tau)
-\sum_\beta f_\beta(\mathbf{n},\tau)P(\mathbf{n},\tau).
```

In the deterministic large-copy-number limit, this reduces to mass-action kinetics (MAK):

```math
\partial_\tau \mathbf{x}=\mathbf{S}\,\mathbf{f}^{\mathrm{MAK}},
\qquad
f_\beta^{\mathrm{MAK}}(\mathbf{x})=j_\beta\prod_i x_i^{r_i^\beta}.
```

Achedmy targets regimes where MAK is inaccurate because intrinsic fluctuations are large.

## 2. Doi-Peliti path integral

The CME can be mapped to a Doi-Peliti field theory with fields `\phi_i(\tau)` and conjugate fields `\tilde\phi_i(\tau)`. The Doi-shifted Hamiltonian is

```math
H[\tilde\phi,\phi]
=\sum_\beta k_\beta(\tau_-)
\left[\prod_i(1+\tilde\phi_i(\tau))^{s_i^\beta}-\prod_i(1+\tilde\phi_i(\tau))^{r_i^\beta}\right]
\prod_i\phi_i(\tau_-)^{r_i^\beta}.
```

The generating functional is

```math
\mathcal Z(\tilde\theta,\theta)=\int\mathcal D\tilde\phi\,\mathcal D\phi\;e^{S[\tilde\phi,\phi]},
```

with action

```math
\begin{aligned}
S[\tilde\phi,\phi]
&=\int_0^t d\tau\,H[\tilde\phi(\tau),\phi(\tau_-)]
+\sum_i\Bigg(n_{0i}\tilde\phi_i(0)-\phi_i(0)\tilde\phi_i(0) \\
&\quad+\int_0^t d\tau\big[-\tilde\phi_i\partial_\tau\phi_i+\tilde\theta_i\tilde\phi_i+\theta_i\phi_i\big]\Bigg).
\end{aligned}
```

### Observables and two-time functions

The key order parameters are:

```math
\mu_i(\tau)=\langle\phi_i(\tau)\rangle=\langle n_i(\tau)\rangle,
\qquad
\tilde\mu_i(\tau)=\langle\tilde\phi_i(\tau)\rangle=0,
```

```math
R_{ij}(\tau,\tau')=\langle\delta\phi_i(\tau)\,\delta\tilde\phi_j(\tau')\rangle,
\qquad
C_{ij}(\tau,\tau')=\langle\delta\phi_i(\tau)\,\delta\phi_j(\tau')\rangle.
```

The physical number correlator is

```math
N_{ij}(\tau,\tau')\equiv\langle\delta n_i(\tau)\delta n_j(\tau')\rangle.
```

For Gaussian closure (used by the Plefka-reduced dynamics),

```math
N_{ij}(\tau,\tau')=C_{ij}(\tau,\tau')+R_{ij}(\tau,\tau')\mu_j(\tau')\quad(\tau'<\tau).
```

with equal-time identities

```math
N_{ij}(\tau,\tau)=C_{ij}(\tau,\tau)\ (i\neq j),
\qquad
N_{ii}(\tau,\tau)=\mu_i(\tau)+C_{ii}(\tau,\tau).
```

## 3. Dynamical Plefka free energy and linear effective fields

We split the Hamiltonian as

```math
H_\alpha = H_0 + \alpha H_{\mathrm{int}},
```

where `H_0` is a quadratic baseline and `H_{\mathrm{int}}` contains higher-order reactions.

The effective action (Plefka free energy) is the Legendre transform

```math
\Gamma(\tilde\mu,\mu)=\operatorname*{extr}_{\tilde\theta,\theta}
\left\{
\log\int\mathcal D\tilde\phi\,\mathcal D\phi\,
\exp\left[S_\alpha-\sum_i\int d\tau\left(\tilde\mu_i\tilde\theta_i+\mu_i\theta_i\right)\right]
\right\}.
```

Conjugate fields follow from

```math
\theta_{i,\alpha}(\tau)=-\frac{\delta\Gamma}{\delta\mu_i(\tau)},
\qquad
\tilde\theta_{i,\alpha}(\tau)=-\frac{\delta\Gamma}{\delta\tilde\mu_i(\tau)}.
```

Physical dynamics corresponds to zero external fields:

```math
\frac{\delta\Gamma}{\delta\mu_i(\tau)}=0,
\qquad
\frac{\delta\Gamma}{\delta\tilde\mu_i(\tau)}=0.
```

### Plefka expansion and effective fields

We expand

```math
\Gamma_\alpha=\Gamma^0+\alpha\Gamma^1+\frac{\alpha^2}{2}\Gamma^2+\cdots,
```

and similarly for fields. The effective fields are

```math
\tilde\theta_i^{\mathrm{eff}}=-\alpha\tilde\theta_i^1-\frac{\alpha^2}{2}\tilde\theta_i^2+\cdots,
\qquad
\theta_i^{\mathrm{eff}}=-\alpha\theta_i^1-\frac{\alpha^2}{2}\theta_i^2+\cdots.
```

The mean update equations under the Gaussian effective action are

```math
\partial_\tau\mu_i(\tau)=k_{1i}-k_{2i}\mu_i(\tau)+\tilde\theta_i^{\mathrm{eff}}(\tau),
```

```math
-\partial_\tau\tilde\mu_i(\tau)=-k_{2i}\tilde\mu_i(\tau)+\theta_i^{\mathrm{eff}}(\tau),
```

with the physical Doi-shifted solution `\tilde\mu_i\equiv 0` and `\theta_i^{\mathrm{eff}}\equiv 0`.

### Stoichiometric coefficient tensor

A convenient representation of `H_{\mathrm{int}}` is

```math
H_{\mathrm{int}}(\tau)=\sum_{\bar m,\bar n}c_{\bar m,\bar n}(\tau)
\prod_i\delta\tilde\phi_i(\tau)^{m_i}\,\delta\phi_i(\tau_-)^{n_i},
```

with

```math
\begin{aligned}
c_{\bar m,\bar n}(\tau)
&=\sum_\beta k_\beta(\tau_-)
\left[\prod_i\binom{s_i^\beta}{m_i}(1+\tilde\mu_i)^{s_i^\beta-m_i}
-\prod_i\binom{r_i^\beta}{m_i}(1+\tilde\mu_i)^{r_i^\beta-m_i}
\right] \\
&\qquad\times\prod_i\binom{r_i^\beta}{n_i}\mu_i(\tau_-)^{r_i^\beta-n_i}.
\end{aligned}
```

At the physical solution (`\tilde\mu=0`), this simplifies to the expression implemented in `src/Cmn.jl`:

```math
c_{\bar m,\bar n}(\tau)=\sum_\beta k_\beta(\tau_-)
\left[\prod_i\binom{s_i^\beta}{m_i}-\prod_i\binom{r_i^\beta}{m_i}\right]
\prod_i\binom{r_i^\beta}{n_i}\mu_i(\tau_-)^{r_i^\beta-n_i}.
```

First-order Plefka gives

```math
-\tilde\theta_i^1(\tau)=c_{\bar e_i,\bar 0}(\tau),
\qquad
-\theta_i^1(\tau)=c_{\bar 0,\bar e_i}(\tau_+)=0,
```

which recovers MAK for interacting reactions:

```math
\partial_\tau\mu_i(\tau)=k_{1i}-k_{2i}\mu_i(\tau)+\sum_\beta k_\beta(\tau)(s_i^\beta-r_i^\beta)\prod_j\mu_j(\tau)^{r_j^\beta}.
```

Second order introduces explicit memory terms in `\tilde\theta_i^2` through past responses.

## 4. Extended Plefka free energy (means + two-time order parameters)

To go beyond linear order parameters, extended Plefka constrains

```math
Q=\{R,B,C\}=
\{\delta\phi\,\delta\tilde\phi,\ \delta\tilde\phi\,\delta\tilde\phi,\ \delta\phi\,\delta\phi\}
```

via conjugate fields

```math
\hat Q=\{\hat R,\hat B,\hat C\}.
```

The augmented action is

```math
\begin{aligned}
S^Q &= S
+\sum_{ij}\int d\tau\,d\tau'\Big[
\hat R_{ij}(\tau,\tau')\,\delta\tilde\phi_j(\tau)\delta\phi_i(\tau') \\
&\qquad\qquad\qquad+\tfrac12\hat B_{ij}(\tau,\tau')\,\delta\tilde\phi_i(\tau)\delta\tilde\phi_j(\tau')
+\tfrac12\hat C_{ij}(\tau,\tau')\,\delta\phi_i(\tau)\delta\phi_j(\tau')
\Big].
\end{aligned}
```

The extended free energy is

```math
G(\tilde\mu,\mu,Q)=\Gamma(\tilde\mu,\mu)
-\sum_{ij}\int d\tau\,d\tau'\left[
\hat R_{ij}R_{ji}+\tfrac12\hat B_{ij}B_{ij}+\tfrac12\hat C_{ij}C_{ij}
\right],
```

and the effective quadratic fields are expanded as

```math
\hat Q^{\mathrm{eff}}=-\alpha\hat Q^1-\frac{\alpha^2}{2}\hat Q^2+\cdots.
```

### Kadanoff-Baym update equations

With the effective Gaussian action, the coupled updates are:

```math
\partial_\tau\mu_i(\tau)=k_{1i}-k_{2i}\mu_i(\tau)+\tilde\theta_i^{\mathrm{eff}}(\tau),
```

```math
(\partial_\tau+k_{2i})R_{ij}(\tau,\tau')=
\delta_{ij}\delta(\tau-\tau')+
\int_{\tau'}^{\tau}d\tau''\sum_k \hat R^{\mathrm{eff}}_{ik}(\tau,\tau'')R_{kj}(\tau'',\tau'),
```

```math
\begin{aligned}
(\partial_\tau+k_{2i})C_{ij}(\tau,\tau')
&=\int_0^\tau d\tau''\sum_k \hat R^{\mathrm{eff}}_{ik}(\tau,\tau'')C_{kj}(\tau'',\tau') \\
&\quad+\int_0^\tau d\tau''\sum_k \hat B^{\mathrm{eff}}_{ik}(\tau,\tau'')R_{jk}(\tau',\tau'').
\end{aligned}
```

A numerically convenient non-differential form used in Achedmy is

```math
\mathbf C=(\Delta t)^2\,\mathbf R\,\mathbf{\hat B}^{\mathrm{eff}}\,\mathbf R^{\mathsf T}.
```

## 5. Effective fields used by MAK, MCA, SBR, and gSBR

### First-order extended fields

For at-most-binary reactions, first-order fields are

```math
-\tilde\theta_i^1(\tau)=c_{\bar e_i,\bar 0}(\tau)+\sum_{k\le l}c_{\bar e_i,\bar e_k+\bar e_l}(\tau)C_{kl}(\tau_-,\tau_-),
```

```math
-\hat R^1_{ij}(\tau,\tau')=\frac{\delta_{\tau',\tau_-}}{\Delta t}\,c_{\bar e_i,\bar e_j}(\tau),
```

```math
-\frac12\hat B^1_{ij}(\tau,\tau')=
\frac{\delta_{\tau',\tau}}{2\Delta t}
\left[c_{\bar e_i+\bar e_j,\bar 0}(\tau)+\sum_{k\le l}c_{\bar e_i+\bar e_j,\bar e_k+\bar e_l}(\tau)C_{kl}(\tau_-,\tau_-)
\right],
```

with `\hat C^1=0`.

### MCA (`O(\alpha^2)`) kernel

The second-order response kernel is

```math
-\hat R^2_{ij}(\tau,\tau')=
2\sum_{\bar n,\bar m\in\mathbb S}
 c_{\bar e_i,\bar n}(\tau)c_{\bar m,\bar e_j}(\tau'_+)
\Lambda^{\bar n,\bar m}(\tau_-,\tau'_+),
```

where `\Lambda^{\bar n,\bar m}` is the sum of all Wick pairings built from response functions.

### gSBR resummation

gSBR replaces the truncated `O(\alpha^2)` kernel by an infinite bubble-chain resummation. In continuous time,

```math
\begin{aligned}
-\hat R^{2,\mathrm{gSBR}}_{ij}(\tau,\tau')
&=2\sum_{\bar n,\bar m,\bar n',\bar m'\in\mathbb S}
 c_{\bar e_i,\bar n}(\tau)
\int_0^\tau d\tau'' \\
&\times\left(\delta_{\bar n,\bar m}\delta(\tau-\tau'')-\Lambda^{\bar n,\bar m}(\tau,\tau'')c_{\bar m,\bar n'}(\tau'')\right)^{-1}
\Lambda^{\bar n',\bar m'}(\tau'',\tau')c_{\bar m',\bar e_j}(\tau').
\end{aligned}
```

The effective memory kernel used in the response equation is

```math
\hat R^{\mathrm{eff}}=-\hat R^1-\frac12\hat R^{2,\mathrm{gSBR}}.
```

In the implementation, this inverse is computed as a causal block lower-triangular solve (`src/BlockOp.jl`), which stabilizes dynamics at large reaction rates.

## 6. Numerical update equations in Achedmy.jl

Achedmy solves the coupled mean/response dynamics with adaptive two-time Kadanoff-Baym integration (`KadanoffBaym.jl`). On a nonuniform time grid (`h_1` quadrature weights), updates are:

```math
\dot\mu_i(t)=k_{1i}-k_{2i}\mu_i(t)+\int_0^t d\tau\,\Sigma_\mu^i(t,\tau),
```

```math
\partial_t R_{ij}(t,t')=-k_{2i}R_{ij}(t,t')+\delta_{ij}\delta(t-t')+
\int_{t'}^t d\tau\sum_k\Sigma_R^{ik}(t,\tau)R_{kj}(\tau,t'),
```

```math
\mathbf C\approx\mathbf R\,(\mathbf\Sigma_B\odot \mathbf W)\,\mathbf R^{\mathsf T},
\qquad
N_{ij}(t,t')=C_{ij}(t,t')+\mu_j(t')R_{ij}(t,t').
```

Here `\Sigma_\mu`, `\Sigma_R`, and `\Sigma_B` are computed from `c_{\bar m,\bar n}` using one of four closures:

- `MAK`: first-order local terms only.
- `MCA`: second-order (`O(\alpha^2)`) truncation.
- `SBR`: single-species self-consistent bubble resummation.
- `gSBR`: full cross-species and cross-reaction bubble resummation.

## 7. Practical scope and limits

The formulation is designed for Markovian jump processes with polynomial propensities, especially CRNs with at-most-binary reactions where gSBR has the strongest empirical performance. It gives accurate means and two-time functions in regimes where MAK/LNA fail, while avoiding direct solution of the full CME state space.

For worked examples and API usage, see:

- `docs/src/tutorial.md`
- `docs/src/examples.md`
- `docs/src/api.md`
