"""
    mnList(structure)

Generate index lists for self-energy coefficient calculations.

Creates comprehensive lists of multi-indices \$m\$ and \$n\$ that enumerate all possible
species count combinations for reaction terms. These indices are fundamental to the
(quasi-)Hamiltonian representation of Chemical reaction network in Doi-Peliti formalism 
and determine which coefficients \$c_{mn}\$ need to be computed.

# Arguments

- `structure::ReactionStructure`: Reaction network structure

# Returns

A tuple of 6 elements:
- `m_list`: Vector of m-lists per reaction (length = num_interactions)
- `n_list`: Vector of n-lists per reaction (length = num_interactions)
- `m_listFULL`: Full m-list across all reactions (may contain unused entries)
- `n_listFULL`: Full n-list across all reactions (may contain unused entries)
- `m_listNEW`: Union of all per-reaction m-lists (optimized, no duplicates)
- `n_listNEW`: Union of all per-reaction n-lists (optimized, no duplicates)

# Details

For each reaction α, generates:
- **m-indices**: \$m_i \\in [0, \\max(s_i^α, r_i^α)]\$ where \$s_i^α\$ = products, \$r_i^α\$ = reactants
- **n-indices**: \$n_i \\in [0, r_i^α]\$ (limited by reactant stoichiometry)

The m-indices enumerate fluctuation terms, while n-indices enumerate derivative terms
in the Plefka expansion.

# Example

```julia
rn = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end
structure = ReactionStructure(rn, [:S=>100, :E=>10, :SE=>0, :P=>0])
m_list, n_list, m_full, n_full, m_new, n_new = mnList(structure)

# m_new contains all unique multi-indices like [0,0,0,0], [1,0,0,0], [0,1,0,0], etc.
```

# See Also

- [`c_mnFULL`](@ref): Uses these lists to compute coefficients
- [`ReactionStructure`](@ref): Contains stoichiometry matrices
"""
function mnList(structure)
    
    num_int = structure.num_interactions
    num_species = structure.num_species
    s_i = structure.stochiometry_prod
    r_i = structure.stochiometry_react
    
    m_list = []
    n_list = []
    m_listFULL = []
    n_listFULL = []
    
    for i in 1:num_int
        
        x = vec(collect(product([0:Int(max(s_i[j,i], r_i[j,i])) for j in 1:num_species]...)))
        y = vec(collect(product([0:Int(r_i[j,i]) for j in 1:num_species]...)))
        
        #The above creates tuples(Int, Int ... , Int (len = num_species)) which needs to be converted to an array,
        # Otherwise it runs into issues later!
        
        x′ = []
        y′ = []      
        
        push!(x′,[[Int(j) for j in k] for k in x])
        push!(y′,[[Int(j) for j in k] for k in y])
        
        push!(m_list, x′[1])
        push!(n_list, y′[1])
    end
    
    #Construct the full lists here
    max1 = zeros(Int,num_species)
    max2 = zeros(Int,num_species)
    
    for j in 1:num_species
    
        max1[j] = maximum(cat(collect(s_i[j,i] for i in 1:num_int),collect(r_i[j,i] for i in 1:num_int),dims=1))
        max2[j] = maximum(collect(r_i[j,i] for i in 1:num_int))
    end
    
    
    a = vec(collect(product([0:max1[j] for j in 1:num_species]...)))
    b = vec(collect(product([0:max2[j] for j in 1:num_species]...)))

    push!(m_listFULL, a)
    push!(n_listFULL, b)
    
    #Also define the NEW lists which are just the intersection of the lists across the different reactions,
    #the FULL lists are overkills and have extra elements
    
    m_listNEW = union(collect.(m_list[j] for j in 1:num_int)...)
    n_listNEW = union(collect.(n_list[j] for j in 1:num_int)...)
    
    #create excluded lists as well:

    return m_list, n_list, m_listFULL, n_listFULL, m_listNEW, n_listNEW
    
end

"""
    c_mn_no_mu(structure, int_rxn_index, m, n)

Compute coefficient \$c_{mn}^β\$ for reaction β **without** mean-field factors.

This is the "bare" coefficient from the Plefka expansion, representing the contribution
of reaction β to the self-energy. The full coefficient includes μ factors (see [`c_mn`](@ref)).

# Arguments

- `structure::ReactionStructure`: Network structure
- `int_rxn_index::Int`: Reaction index β
- `m::Vector{Int}`: Multi-index for fluctuation terms (length = num_species)
- `n::Vector{Int}`: Multi-index for derivative terms (length = num_species)

# Returns

- `Float64`: The coefficient value (0 if n exceeds reactant stoichiometry)

# Mathematical Formula

```math
c_{m,n}^β(t) = k_\\beta(t_-) \\left[ \\prod_i \\binom{s_i^\\beta}{m_i} \\left( \\tilde{\\mu}_i(t) + 1 \\right)^{s_i^\\beta - m_i} - \\right. \\\\
\\left. \\prod_i \\binom{r_i^\\beta}{m_i} \\left( \\tilde{\\mu}_i(t) + 1  \\right)^{r_i^\\beta-m_i} \\right]
\\prod_i  \\binom{r_i^\\beta}{n_i}
```

where:
- \$k_β\$ = reaction rate
- \$s_i^β\$ = product stoichiometry
- \$r_i^β\$ = reactant stoichiometry

The first term vanishes if any \$n_i > r_i^β\$ (can't take more derivatives than reactants).

# Example

```julia
# For reaction: S + E --> SE with rate k1
# m = [1, 0, 0], n = [1, 0, 0] (one S fluctuation, one S derivative)
c = c_mn_no_mu(structure, 1, [1,0,0], [1,0,0])
# Returns: k1 * (binom(0,1) - binom(1,1)) * binom(1,1) = -k1
```

# See Also

- [`c_mn`](@ref): Full coefficient with μ factors
- [`c_mnFULL`](@ref): Sum over all reactions
"""
function c_mn_no_mu(structure,int_rxn_index,m,n)
    
    if any(n.-structure.stochiometry_react[:,int_rxn_index] .> 0.)
        c_mnBeta = 0.

    else
        c_mnBeta = structure.rate_interaction[int_rxn_index]*(prod(
        binomial.(structure.stochiometry_prod[:,int_rxn_index],m).*(1 .^(structure.stochiometry_prod[:,int_rxn_index].-m))
        ) .- prod(binomial.(structure.stochiometry_react[:,int_rxn_index],m).*(1 .^(structure.stochiometry_react[:,int_rxn_index].-m) )
        ))*prod(binomial.(structure.stochiometry_react[:,int_rxn_index],n))
    
    end
    
    return c_mnBeta
end

"""
    c_mn(structure, variables, int_rxn_index, m, n, time)

Compute full coefficient \$c_{mn}^β(t)\$ for reaction β **including** mean-field factors.

This extends [`c_mn_no_mu`](@ref) by multiplying with the time-dependent mean-field
density \$μ_i(t)\$ raised to appropriate powers. This is the coefficient actually used
in self-energy calculations.

# Arguments

- `structure::ReactionStructure`: Network structure
- `variables::ReactionVariables`: Variables container (contains μ history)
- `int_rxn_index::Int`: Reaction index β
- `m::Vector{Int}`: Multi-index for fluctuation terms
- `n::Vector{Int}`: Multi-index for derivative terms  
- `time::Int`: Time index t

# Returns

- `Float64`: The time-dependent coefficient (0 for t ≤ 1 or invalid n)

# Mathematical Formula

```math
\\begin{multline}
c_{m,n}^β(t) = k_\\beta(t_-) \\left[ \\prod_i \\binom{s_i^\\beta}{m_i} \\left( \\tilde{\\mu}_i(t) + 1 \\right)^{s_i^\\beta - m_i} - \\right. \\\\
\\left. \\prod_i \\binom{r_i^\\beta}{m_i} \\left( \\tilde{\\mu}_i(t) + 1  \\right)^{r_i^\\beta-m_i} \\right]
\\prod_i  \\binom{r_i^\\beta}{n_i} \\mu_i^{r_i^\\beta-n_i}(t-1) 
\\end{multline}
```

Note the **time shift** t → t-1 in μ factors, which is crucial for causality in the
memory integral.

# Time Index Convention

- `time = 1`: Returns 0 (no history available)
- `time > 1`: Uses μ[:,time-1] from previous timestep

# Example

```julia
# At t=10, for S + E --> SE reaction
c = c_mn(structure, variables, 1, [1,0,0], [0,1,0], 10)
# Includes factor: μ_S(t=9)^1 * μ_E(t=9)^0 = μ_S(t=9)
```

# See Also

- [`c_mn_no_mu`](@ref): Bare coefficient without μ
- [`c_mnFULL`](@ref): Sum over all reactions
"""
function c_mn(structure,variables,int_rxn_index,m,n,time)

    if any(n.-structure.stochiometry_react[:,int_rxn_index] .> 0.)
        
        c_mnBeta = 0.

    else
        if time>1
        
        c_mnBeta = structure.rate_interaction[int_rxn_index]*(prod(
        binomial.(structure.stochiometry_prod[:,int_rxn_index],m).*(1 .^(structure.stochiometry_prod[:,int_rxn_index].-m))
        ) .- prod(binomial.(structure.stochiometry_react[:,int_rxn_index],m).*(1 .^(structure.stochiometry_react[:,int_rxn_index].-m) )
        )).*prod(binomial.(structure.stochiometry_react[:,int_rxn_index],n)).*prod(variables.μ[:,time-1] .^(structure.stochiometry_react[:,int_rxn_index].-n))
        
        #Note that this cmn is defined with the time shift t- in \mu!
                
        else
            c_mnBeta = 0.
        end
        
    end
        
    return c_mnBeta
    
end

"""
    c_mnFULL(structure, variables, m, n, time)

Compute total coefficient \$c_{mn}(t)\$ summed over **all reactions**.

This is the primary function used in self-energy calculations. It sums the individual
reaction contributions \$c_{mn}^β(t)\$ from all reactions in the network.

# Arguments

- `structure::ReactionStructure`: Network structure
- `variables::ReactionVariables`: Variables container
- `m::Vector{Int}`: Multi-index for fluctuation terms (length = num_species)
- `n::Vector{Int}`: Multi-index for derivative terms (length = num_species)
- `time::Int`: Time index t

# Returns

- `Float64`: Total coefficient \$\\sum_β c_{mn}^β(t)\$

# Mathematical Formula

```math
c_{mn}(t) = \\sum_{β=1}^{N_{\\text{reactions}}} c_{mn}^β(t)
```

where each \$c_{mn}^β(t)\$ includes both combinatorial factors and mean-field densities.

# Usage in Self-Energies

This function appears repeatedly in all self-energy calculations:

```julia
# In self_energy_mak_noC!
Σ_R[i,t] += c_mnFULL(structure, variables, [1,0,...], [1,0,...], t)

# In self_energy_SBR!  
Σ_R[i,t] += sum(c_mnFULL(..., m, n, t) * R^m * R^n for m,n in lists)
```

# Performance

Called **many times per timestep** in iterative solvers. For large networks:
- Consider caching results if m,n are reused
- Use `c_mnFULL_test` to skip zero contributions

# Example

```julia
# Total contribution at t=10 for m=[1,0,0], n=[0,0,0]
c_total = c_mnFULL(structure, variables, [1,0,0], [0,0,0], 10)
# Sums over all reactions affecting species 1
```

# See Also

- [`c_mn`](@ref): Single reaction contribution
- [`c_mnFULL_test`](@ref): Test if coefficient is zero
- Self-energy functions that call this: [`self_energy_mak_noC!`](@ref), [`self_energy_SBR!`](@ref), etc.
"""
function c_mnFULL(structure,variables,m,n,time)

    retval = 0.
    
    for k in 1:structure.num_interactions
        retval += c_mn(structure,variables,k,m,n,time)
    end
    
    return retval
    
end

"""
    c_mnFULL_test(structure, variables, m, n)

Test if coefficient \$c_{mn}\$ is **non-zero** (ignoring μ factors).

This is a **fast screening function** used to skip zero-valued terms before expensive
calculations. It checks only the combinatorial structure, not the time-dependent μ values.

# Arguments

- `structure::ReactionStructure`: Network structure
- `variables::ReactionVariables`: Variables container (not actually used, but kept for API consistency)
- `m::Vector{Int}`: Multi-index for fluctuation terms
- `n::Vector{Int}`: Multi-index for derivative terms

# Returns

- `Float64`: Non-zero if any reaction contributes, 0 if all reactions give zero

# When to Use

Use this **before** calling `c_mnFULL` in tight loops:

```julia
# Inefficient:
for m in m_list, n in n_list
    c = c_mnFULL(structure, variables, m, n, t)
    # ... use c
end

# Efficient:
for m in m_list, n in n_list
    if c_mnFULL_test(structure, variables, m, n) != 0
        c = c_mnFULL(structure, variables, m, n, t)
        # ... use c
    end
end
```

# Mathematical Details

Sums \$c_{mn}^β\$ **without μ factors**:

```math
c_{mn,\\text{test}} = \\sum_β k_β \\left[\\prod_i \\binom{s_i^β}{m_i} - \\prod_i \\binom{r_i^β}{m_i}\\right] \\prod_i \\binom{r_i^β}{n_i}
```

If this is zero, then \$c_{mn}(t)\$ will be zero for **all times** t.

# Performance Impact

For large networks with many (m,n) pairs, pre-filtering with this function can
reduce computational cost by **orders of magnitude** in self-energy calculations.

# See Also

- [`c_mnFULL`](@ref): Full time-dependent coefficient
- [`c_mn_no_mu`](@ref): Per-reaction bare coefficient
"""
function c_mnFULL_test(structure,variables,m,n)

    retval = 0.
        
    for k in 1:structure.num_interactions
        retval += c_mn_no_mu(structure,k,m,n)
    end
    
    return retval
    
end

"""
    create_c_mn_dict(structure, variables, m_list, n_list)

Create dictionary for **pre-computed** bare coefficients \$c_{mn}^β\$.

This function pre-computes all \$c_{mn}^β\$ (without μ) for each reaction and stores
them in a dictionary for fast lookup. Useful for optimizing repeated calculations.

# Arguments

- `structure::ReactionStructure`: Network structure
- `variables::ReactionVariables`: Variables container (not used but kept for consistency)
- `m_list::Vector{Vector}`: List of m-indices per reaction
- `n_list::Vector{Vector}`: List of n-indices per reaction

# Returns

- `Dict{Tuple{Int, Vector{Int}, Vector{Int}}, Float64}`: Dictionary with keys `(rxn_index, m, n)` → coefficient value

# Structure

The dictionary is indexed by:
- **Key**: `(int_rxn_index, m, n)` where:
  - `int_rxn_index::Int`: Reaction β
  - `m::Vector{Int}`: Fluctuation multi-index
  - `n::Vector{Int}`: Derivative multi-index
- **Value**: `c_mn_no_mu(structure, int_rxn_index, m, n)`

# Example

```julia
m_list, n_list, _, _, _, _ = mnList(structure)
c_dict = create_c_mn_dict(structure, variables, m_list, n_list)

# Fast lookup
c_val = c_dict[(1, [1,0,0], [1,0,0])]  # Reaction 1, m=[1,0,0], n=[1,0,0]
```

# When to Use

- **Advantage**: O(1) lookup vs recomputing combinatorics
- **Disadvantage**: Memory overhead for large networks
- **Best for**: Small to medium networks with repeated (m,n) access patterns

# Memory Usage

Approximately:
- Number of entries = sum(length(m_list[i]) × length(n_list[i]) for i in reactions)
- Each entry: ~24 bytes (key + value)

For typical biochemical networks (3-10 species, 5-20 reactions): <1 MB.

# See Also

- [`c_mn_no_mu`](@ref): Function being memoized
- [`mnList`](@ref): Generates m_list and n_list inputs
"""
function create_c_mn_dict(structure,variables,m_list,n_list)

    c_mn_dict = Dict()

    for int_rxn_index in range(1,structure.num_interactions)
        
        #Note that for the m_list, n_list above, they should be the m-list, n-list 
        #corresponding to each reaction individually 
        for m in m_list[int_rxn_index]
            for n in n_list[int_rxn_index]
                
                #Can already NOT SAVE many of the c_mn which are zero at this point!
                c_mn_dict[(int_rxn_index,m,n)] = c_mn_no_mu(structure,int_rxn_index,m,n)
            
            end
        end
    end
    
    return c_mn_dict

end             