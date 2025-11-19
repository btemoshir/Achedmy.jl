# Mathematical Theory

This page explains the theoretical framework underlying Achedmy.jl, from the master equation to the various approximation schemes.

## The Chemical Master Equation

ChemicaThese encode:
1. **Reaction rates** ($k_\alpha$)
2. **Stoichiometric structure** (binomial coefficients)
3. **Mean-field densities** ($\mu^{\mathbf{r}-\mathbf{n}}$)

See [`c_mnFULL`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Cmn.jl) for implementation details.tion networks with stochastic dynamics are governed by the **Master Equation**:

```math
\\frac{\\partial P(\\mathbf{n}, t)}{\\partial t} = \\sum_{\\alpha} k_\\alpha \\left[ E_{-\\mathbf{s}_\\alpha + \\mathbf{r}_\\alpha} - 1 \\right] \\left( \\prod_i n_i^{r_i^\\alpha} \\right) P(\\mathbf{n}, t)
```

where:
- \$P(\\mathbf{n}, t)\$ = probability of state \$\\mathbf{n}\$ at time \$t\$
- \$k_\\alpha\$ = rate of reaction \$\\alpha\$
- \$\\mathbf{s}_\\alpha\$ = product stoichiometry
- \$\\mathbf{r}_\\alpha\$ = reactant stoichiometry
- \$E_{\\mathbf{m}}\$ = step operator: \$E_{\\mathbf{m}} f(\\mathbf{n}) = f(\\mathbf{n} + \\mathbf{m})\$

### The Challenge

Solving the Master Equation exactly requires tracking \$\\mathcal{O}(N^M)\$ states (N = typical copy number, M = species). This becomes intractable for even modest systems.

## The Plefka Expansion

Achedmy uses the **Plefka expansion** (TAP approximation) to derive closed equations for:

1. **Mean densities**: \$\\mu_i(t) = \\langle n_i(t) \\rangle\$
2. **Response functions**: \$R_{ij}(t,t') = \\frac{\\delta \\mu_i(t)}{\\delta h_j(t')}\$
3. **Correlation functions**: \$N_{ij}(t,t') = \\langle \\Delta n_i(t) \\Delta n_j(t') \\rangle\$

where \$\\Delta n_i = n_i - \\mu_i\$.

### Effective Action

The Plefka expansion starts from the effective action:

```math
S[\\mu, R] = -\\log Z[h] - \\int dt \\sum_i h_i(t) \\mu_i(t) + \\frac{1}{2} \\int dt dt' \\sum_{ij} h_i(t) R_{ij}(t,t') h_j(t')
```

Extremizing this action gives **memory-corrected** equations of motion.

## Equations of Motion

### Response Function Equation

The response function satisfies a Kadanoff-Baym equation:

```math
\\left[ \\frac{\\partial}{\\partial t} \\delta_{ij} - A_{ij}(t) \\right] R_{jk}(t,t') = \\delta_{ik} \\delta(t-t') + \\int_{t'}^t dt'' \\, \\Sigma_R^{ij}(t,t'') R_{jk}(t'',t')
```

where:
- \$A_{ij}(t)\$ = Jacobian of mean-field dynamics
- \$\\Sigma_R^{ij}(t,t'')\$ = **self-energy** (memory kernel)

### Correlation Function Equation

Similarly for correlations:

```math
N_{ij}(t,t') = \\int_{t_0}^{\\min(t,t')} dt'' \\, R_{ik}(t,t'') \\Sigma_B^{kl}(t'',t'') R_{lj}(t',t'')
```

where:
- \$\\Sigma_B^{kl}(t,t)\$ = Born self-energy (noise kernel)

### Mean Density Equation

The mean evolves with a memory term:

```math
\\frac{d\\mu_i(t)}{dt} = \\sum_\\alpha k_\\alpha (s_i^\\alpha - r_i^\\alpha) \\mu^{\\mathbf{r}_\\alpha}(t) + \\int_{t_0}^t dt' \\, \\Sigma_\\mu^i(t,t')
```

where:
- \$\\Sigma_\\mu^i(t,t')\$ = mean-field correction from fluctuations

## The Self-Energy: Different Approximations

The **key challenge** is computing the self-energy \$\\Sigma\$. Achedmy implements four approximation schemes:

### 1. MAK (Mean-field Approximation with Kinetics)

**Assumption**: Neglect all memory effects (\$\\Sigma \\approx 0\$).

**Equations**:
```math
\\begin{aligned}
\\Sigma_R^{ij}(t,t') &= 0 \\\\
\\Sigma_\\mu^i(t,t') &= 0 \\\\
\\Sigma_B^{ij}(t,t) &= \\sum_\\alpha k_\\alpha (s_i^\\alpha + r_i^\\alpha) \\mu^{\\mathbf{r}_\\alpha}(t)
\\end{aligned}
```

**Pros**: 
- Fastest (no memory integrals)
- Simple analytic structure

**Cons**:
- Inaccurate for systems with strong fluctuations
- No memory effects

**When to use**: Quick estimates, mean-field dominated systems

### 2. MCA (Mode Coupling Approximation)

**Assumption**: Perturbative expansion to \$\\mathcal{O}(\\alpha^2)\$ in the "vertex" parameter \$\\alpha\$.

**Self-energy**:
```math
\\Sigma_R^{ij}(t,t') = \\sum_{\\alpha,\\beta} \\sum_{m,n} c_{mn}^{(\\alpha\\beta)}(t) \\prod_k R_{kk}(t,t')^{m_k+n_k}
```

where:
- \$c_{mn}^{(\\alpha\\beta)}\$ = combinatorial coefficients (see [`c_mnFULL`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Cmn.jl))
- Sum restricted to \$|m| + |n| < 3\$ (second-order)

**Pros**:
- Captures leading-order memory effects
- Moderate computational cost

**Cons**:
- Perturbative (fails for large \$\\alpha\$)
- No self-consistent resummation

**When to use**: Weak to moderate fluctuations

### 3. SBR (Single-species Bubble Resummation)

**Assumption**: Resum geometric series of "bubble" diagrams for **each species independently**.

**Self-energy**:
```math
\\Sigma_R^{ii}(t,t') = \\sum_{m,n} c_{mn,i}(t) \\left[ \\frac{1}{1 - \\chi_i} \\right] R_{ii}(t,t')^{m_i+n_i}
```

where:
- \$\\chi_i = c_{mn,i} R_{ii}^{m_i+n_i}\$ (bubble series for species \$i\$)
- \$\\frac{1}{1-\\chi_i}\$ = geometric series: \$1 + \\chi_i + \\chi_i^2 + \\cdots\$

**Pros**:
- Self-consistent (not perturbative)
- Captures multi-order memory effects
- Moderate cost (no matrix inversion)

**Cons**:
- Neglects cross-species correlations
- Less accurate than gSBR for coupled systems

**When to use**: Single-species dominant, moderate coupling

### 4. gSBR (Generalized SBR)

**Assumption**: Resum bubble series with **full cross-species** coupling.

**Self-energy**:
```math
\\Sigma_R^{ij}(t,t') = \\sum_{m,n} c_{mn}^{(\\alpha\\beta)}(t) \\left[ (I + \\Sigma_R \\cdot R)^{-1} \\right]_{ij} R_{ik}(t,t')^{m_k} R_{jl}(t,t')^{n_l}
```

where:
- \$I + \\Sigma_R \\cdot R\$ = full operator to invert (via [`block_tri_lower_inverse`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/BlockOp.jl))
- All cross-species terms \$R_{ij}\$ included

**Pros**:
- **Most accurate** approximation
- Captures cross-species correlations
- Self-consistent resummation

**Cons**:
- Most expensive (\$\\mathcal{O}(N^4 T^3)\$)
- Requires block matrix inversion

**When to use**: Strong coupling, cross-species correlations important

## Comparison of Methods

| Feature | MAK | MCA | SBR | gSBR |
|---------|-----|-----|-----|------|
| **Complexity** | \$\\mathcal{O}(N T)\$ | \$\\mathcal{O}(N T^2)\$ | \$\\mathcal{O}(N T^2)\$ | \$\\mathcal{O}(N^4 T^3)\$ |
| **Memory effects** | ✗ | ✓ (pert.) | ✓ (self-cons.) | ✓ (self-cons.) |
| **Cross-species** | ✗ | ✗ | ✗ | ✓ |
| **Self-consistent** | N/A | ✗ | ✓ | ✓ |
| **Accuracy** | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## The Coefficient Functions

A key component is the coefficient \$c_{mn}^\\alpha(t)\$:

```math
c_{mn}^\\alpha(t) = k_\\alpha \\left[ \\prod_i \\binom{s_i^\\alpha}{m_i} - \\prod_i \\binom{r_i^\\alpha}{m_i} \\right] \\prod_i \\binom{r_i^\\alpha}{n_i} \\mu^{\\mathbf{r}_\\alpha - \\mathbf{n}}(t)
```

These encode:
1. **Reaction rates** (\$k_\\alpha\$)
2. **Stoichiometric structure** (binomial coefficients)
3. **Mean-field densities** (\$\\mu^{\\mathbf{r}-\\mathbf{n}}\$)

See [`c_mnFULL`](@https://github.com/btemoshir/Achedmy.jl/blob/main/src/Cmn.jl) for implementation details.

## Block Matrix Structure (gSBR)

The gSBR self-energy requires inverting a **block lower-triangular** matrix:

```math
\\Xi = \\left[ I - \\chi \\right]^{-1}
```

where:
- **Block dimension**: Different \$(m,n)\$ pairs
- **Time dimension**: Lower-triangular causality structure

This is computed efficiently using forward elimination (see [`block_tri_lower_inverse`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/BlockOp.jl) rather than naive matrix inversion.

### Geometric Series Interpretation

The inversion implements the geometric series:

```math
\\Xi = I + \\chi + \\chi^2 + \\chi^3 + \\cdots
```

Each term represents higher-order bubble contributions:
- \$I\$: Mean-field
- \$\\chi\$: Single bubble
- \$\\chi^2\$: Two bubbles
- ...

## Time Integration: Adaptive Kadanoff-Baym

Achedmy uses the **KadanoffBaym.jl** package for adaptive time-stepping:

1. **Initial step**: Coarse grid with user-specified \$dt\$
2. **Error estimation**: Local truncation error from ODE solver
3. **Refinement**: Add points where error exceeds tolerance
4. **Two-time propagation**: Extend both \$t\$ and \$t'\$ axes

This ensures accuracy while minimizing computational cost.

## Comparison to Other Methods

### vs. Gillespie SSA

| Aspect | Gillespie | Achedmy |
|--------|-----------|---------|
| **Speed** (two-time) | ⚠️ Very slow | ✓ Fast |
| **Accuracy** | ✓ Exact | ≈ Excellent |
| **Memory** | Low | Moderate-High |
| **Scalability** | Poor | Good |

**Conclusion**: Achedmy is 10-1000× faster for two-time correlations, with comparable accuracy.

### vs. Linear Noise Approximation (LNA)

| Aspect | LNA | Achedmy |
|--------|-----|---------|
| **Regime** | Near deterministic | All regimes |
| **Accuracy** | Good (large N) | Good (all N) |
| **Memory effects** | ✗ | ✓ |
| **Two-time** | Limited | Full |

**Conclusion**: Achedmy extends beyond LNA's Gaussian regime and includes memory effects.

## Further Reading

For detailed derivations and benchmarks, see:

1. **Plefka Expansion**: Plefka (1982), Georges & Yedidia (1991)
2. **Chemical Reaction Networks**: Van Kampen (2007)
3. **Kadanoff-Baym Equations**: Kadanoff & Baym (1962), Balzer et al. (2013)

## See Also

- [Getting Started](tutorial.md) for practical usage
- [Examples](examples.md) for applications to real systems
- [API Reference](api.md) for implementation details

<!-- - [Getting Started](@ref) for practical usage
- [Examples](@ref) for applications to real systems
- [API Reference](@ref) for implementation details -->
