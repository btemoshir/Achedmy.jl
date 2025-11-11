# API Reference

Complete API documentation for Achedmy.jl.

## Core Types

```@docs
ReactionStructure
ReactionVariables
```

## Main Solver

```@docs
solve_dynamics!
```

## Self-Energy Functions

These functions compute the memory kernels (self-energies) for different approximation methods.

```@docs
self_energy_mak_noC!
self_energy_alpha2!
self_energy_SBR!
self_energy_SBR_mixed!
self_energy_SBR_mixed_cross_noC!
self_energy_alpha2_cross!
```

## Coefficient Functions

Functions for computing the \$c_{mn}\$ coefficients that appear in self-energies.

```@docs
mnList
c_mn_no_mu
c_mn
c_mnFULL
c_mnFULL_test
create_c_mn_dict
```

## Block Matrix Operations

Specialized linear algebra for gSBR bubble resummation.

```@docs
block_tri_lower_inverse
block_mat_mul
block_mat_mix_mul
block_lower_shift
block_identity
block_vec_mat_mul_single_sp
response_combinations
```

## Index

```@index
```

## Function Index

```@index
Pages = ["api.md"]
Order = [:function, :type]
```
