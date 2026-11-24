# API Reference

```@meta
CurrentModule = Achedmy
```

```@docs
Achedmy
```

This page documents all public functions and types in Achedmy.jl.

## Main Types

```@docs
ReactionStructure
ReactionVariables
Response
```

## Main Functions

```@docs
solve_dynamics!
```

## Symmetry Types

The two-time quantities in [`ReactionVariables`](@ref) are KadanoffBaym.jl
`GreenFunction`s. Only one triangle of the time plane is stored, and the
symmetry type sets the other half:

| Field | `"single"` | `"cross"` | Symmetry defined in |
|---|---|---|---|
| `R`, `N`, `Σ_R`, `Σ_μ` | `Response` | `Response` | Achedmy |
| `μ` | `OnePoint` | `OnePoint` | KadanoffBaym.jl |
| `C` | `Symmetrical` | plain `Array` | KadanoffBaym.jl |
| `Σ_B` | `Symmetrical` | `Symmetrical` | KadanoffBaym.jl |

- **`Response`**: causal, ``G(t,t') = 0`` for ``t' > t``.
- **`OnePoint`**: one-time quantities such as the means ``\mu_i(t)``.
- **`Symmetrical`**: ``G(t,t') = G(t',t)`` element by element.
- **`Correlation`**: the block symmetry of cross-species correlations,
  ``G(i,j,t,t') = G(j,i,t',t)``. **Not implemented yet:** the type is defined,
  but it doesn't fill in the other half (its symmetry is `nothing`) and no field
  uses it. The `"cross"` correlation `C` is stored as a plain array for now.
  This will be updated in a future release.

See [`src/Var.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Var.jl) for the definitions.

## Internal Functions

The following internal functions have detailed docstrings in the source code but are not part of the public API.

### Self-energies ([`src/SelfEnergy.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/SelfEnergy.jl))

[`solve_dynamics!`](@ref) picks the self-energy from `selfEnergy` and the
response type of the variables:

| `selfEnergy` | `"single"` | `"cross"` |
|---|---|---|
| `"MAK"` | `self_energy_mak_noC!` | `self_energy_mak_noC!` |
| `"MCA"` | `self_energy_alpha2!` | `self_energy_alpha2_cross!` |
| `"SBR"` | `self_energy_SBR!` | not available (use `"gSBR"`) |
| `"gSBR"` | `self_energy_SBR_mixed!` | `self_energy_SBR_mixed_cross_noC!` |

- `self_energy_mak_noC!`: mass-action kinetics; first-order fields only, no memory
- `self_energy_alpha2!`, `self_energy_alpha2_cross!`: second-order (``O(\alpha^2)``) MCA kernel
- `self_energy_SBR!`: SBR; self-consistent single-species fluctuations, with the bubble resummation done separately for each reaction
- `self_energy_SBR_mixed!`, `self_energy_SBR_mixed_cross_noC!`: gSBR, resumming jointly over reactions (and, for `"cross"`, over species)

### Coefficient functions ([`src/Cmn.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Cmn.jl))

- `mnList`: the (m, n) multi-index pairs that appear in the self-energies
- `c_mn_no_mu`: the coefficient ``c_{mn}`` of one reaction, without the mean-field factors
- `c_mn`: the coefficient ``c^\beta_{mn}(t)`` of one reaction at time index `t`
- `c_mnFULL`: ``c_{mn}(t)``, summed over all reactions; used by every self-energy
- `c_mnFULL_test`: a fast check of whether ``c_{mn}`` is non-zero (ignoring the μ factors), used to skip zero terms
- `create_c_mn_dict`: pre-computes the bare coefficients ``c^\beta_{mn}`` (without μ) of every reaction in a dictionary

### Block operations ([`src/BlockOp.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/BlockOp.jl))

- `block_tri_lower_inverse`: inverse of a block lower-triangular matrix (the causal resummation in gSBR)
- `block_mat_mul`: block matrix multiplication
- `block_mat_mix_mul`: product of a block matrix with a mixed-structure matrix (one multi-index, two time indices)
- `block_lower_shift`: shifts the blocks down one time step, giving a strictly lower-triangular block matrix
- `block_identity`: block identity matrix
- `block_vec_mat_mul_single_sp`: vector–matrix product for single species
- `response_combinations`: all permutations of the cross-species response products in the bubble corrections

For detailed mathematical descriptions and usage examples, see the comprehensive docstrings in each source file.

See the [source code](https://github.com/btemoshir/Achedmy.jl) for detailed docstrings of internal functions.