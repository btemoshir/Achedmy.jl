# API Reference

```@meta
CurrentModule = Achedmy
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

<!-- ## Internal Functions

The following internal functions are documented in the source code but are not part of the public API:

- **Self-Energy Functions**: `self_energy_mak_noC!`, `self_energy_alpha2!`, `self_energy_SBR!`, `self_energy_SBR_mixed!`, `self_energy_SBR_mixed_cross_noC!`, `self_energy_alpha2_cross!`
- **Coefficient Functions**: `mnList`, `c_mn_no_mu`, `c_mn`, `c_mnFULL`, `c_mnFULL_test`, `create_c_mn_dict`
- **Block Operations**: `block_tri_lower_inverse`, `block_mat_mul`, `block_mat_mix_mul`, `block_lower_shift`, `block_identity`, `block_vec_mat_mul_single_sp`, `response_combinations` -->

## Symmetry Types

Beyond `Response`, Achedmy defines additional symmetry types for Green's functions:

- **`Correlation`**: Enforces block symmetry for cross-species correlations
- **`OnePoint`**: For mean-field (one-point) functions

See the source code in [`src/Var.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Var.jl) for implementation details.


## Internal Functions

The following internal functions have detailed docstrings in the source code but are not part of the public API:

### Self-Energy Functions ([`src/SelfEnergy.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/SelfEnergy.jl))
- `self_energy_mak_noC!` - MAK approximation without correlations
- `self_energy_alpha2!` - α² expansion self-energy
- `self_energy_SBR!` - Single-species SBR approximation
- `self_energy_SBR_mixed!` - Mixed SBR for multiple species
- `self_energy_SBR_mixed_cross_noC!` - Cross-species SBR without initial correlations
- `self_energy_alpha2_cross!` - Cross-species α² expansion

### Coefficient Functions ([`src/Cmn.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/Cmn.jl))
- `mnList` - Generate (m,n) index pairs
- `c_mn_no_mu` - Coefficient calculation without mean-field
- `c_mn` - Full coefficient with mean-field
- `c_mnFULL` - Complete coefficient matrix
- `c_mnFULL_test` - Test version with validation
- `create_c_mn_dict` - Dictionary of coefficients

### Block Operations ([`src/BlockOp.jl`](https://github.com/btemoshir/Achedmy.jl/blob/main/src/BlockOp.jl))
- `block_tri_lower_inverse` - Inverse of block lower-triangular matrix
- `block_mat_mul` - Block matrix multiplication
- `block_mat_mix_mul` - Mixed block multiplication
- `block_lower_shift` - Shift operation for block matrices
- `block_identity` - Block identity matrix
- `block_vec_mat_mul_single_sp` - Vector-matrix product for single species
- `response_combinations` - Generate response function index combinations

For detailed mathematical descriptions and usage examples, see the comprehensive docstrings in each source file.

See the [source code](https://github.com/btemoshir/Achedmy.jl) for detailed docstrings of internal functions.