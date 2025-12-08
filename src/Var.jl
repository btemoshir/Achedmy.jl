"""
    Response

Defined as 
`` G(t,t') = 0 if t' > t ``
"""
struct Response <: KadanoffBaym.AbstractSymmetry end

@inline KadanoffBaym.symmetry(::Type{Response}) = zero

"""
    Correlation

Correlation function with block imposed symmetry since (cross) species correlations are 'block' symmetric in time. Defined as invariant under interchanging first two indices with last two indices:
``G(i,j,t,t') = G(j,i,t',t)``. The pre-defined Symmetrical GreenFunction can be used for this purpose!
To be updated in future releases!
"""
struct Correlation <: KadanoffBaym.AbstractSymmetry end

@inline KadanoffBaym.symmetry(::Type{Correlation}) = nothing

"""
    ReactionVariables

Storage container for all dynamical variables computed during the simulation.

# Fields

- `response_type::String`: Either "single" (diagonal only) or "cross" (full matrix)
- `μ::GreenFunction`: Mean molecular numbers, \$\\langle n_i(t) \\rangle\$
  - Size: `(num_species, num_timesteps)`
  
- `R::GreenFunction`: Response function, \$R_{ij}(t,t')\$
  - **Single**: Size `(num_species, num_timesteps, num_timesteps)`
  - **Cross**: Size `(num_species, num_species, num_timesteps, num_timesteps)`
  - Measures response to perturbations
  - Causal: \$R_{ij}(t,t') = 0\$ for \$t < t'\$
  
- `C::GreenFunction`: Connected correlation function, \$C_{ij}(t,t')\$
  - Same dimensions as `R`
  - \$C_{ij}(t,t') = \\langle \\delta \\phi_i(t) \\delta \\phi_j(t') \\rangle_c\$
  
- `N::GreenFunction`: Number-number correlator, \$N_{ij}(t,t')\$
  - Same dimensions as `R`
  - \$N_{ij}(t,t') = \\langle \\delta n_i(t) \\delta n_j(t') \\rangle\$
  - Related to C by: \$N = C + R \\otimes \\mu\$
  
- `Σ_R::GreenFunction`: Response self-energy (memory kernel)
- `Σ_μ::GreenFunction`: Mean self-energy (memory kernel)
- `Σ_B::GreenFunction`: Correlation self-energy (memory kernel)

# Response Types

## "single" Mode
- Tracks only diagonal elements (single-species correlations)
- Memory usage: O(N × T^n) where N = num_species, T = num_timesteps, n = depends on the approximation used
- Faster for large systems with weak cross-correlations

## "cross" Mode  
- Tracks full matrix (all cross-species correlations)
- Memory usage: O(N² × T^n) where N = num_species, T = num_timesteps, n = depends on the approximation used
- More accurate, captures all correlations

# Constructor

    ReactionVariables(structure::ReactionStructure, response_type::String="cross")

Initialize storage arrays for dynamics computation.

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `response_type::String`: Either "cross" (default) or "single"

# Example

```julia
using Achedmy, Catalyst

enzyme = @reaction_network begin
    @species S(t)=1.0 E(t)=0.9 C(t)=0.1 P(t)=0.0
    @parameters k_f=1.0 k_b=0.1 k_cat=1.0
    (k_f, k_b), S + E <--> C
    k_cat, C --> E + P
end

structure = ReactionStructure(enzyme)

# Full cross-correlations (recommended)
vars_cross = ReactionVariables(structure, "cross")
sol_cross = solve_dynamics!(structure, vars_cross, selfEnergy="gSBR", tmax=5.0)

# Single-species only (memory efficient)
vars_single = ReactionVariables(structure, "single")
sol_single = solve_dynamics!(structure, vars_single, selfEnergy="gSBR", tmax=5.0)

# Access results
mean_substrate = vars_cross.μ[1, :]              # Mean S(t)
variance_product = diag(vars_cross.N[4,4,:,:])  # Var(P) over time
response_SE = vars_cross.R[1,2,:,:]              # R_SE(t,t')
cross_corr_SP = vars_cross.N[1,4,:,:]            # ⟨δS(t)δP(t')⟩
```

# Performance Tips

- Use `"single"` for systems with >10 species to save memory
- For weakly coupled systems, `"single"` may be sufficient

# See Also

- [`ReactionStructure`](@ref)
- [`solve_dynamics!`](@ref)
- [`Response`](@ref): Symmetry types
"""
Base.@kwdef mutable struct ReactionVariables

    response_type = "cross"
    R = 0 
    μ = 0
    C = 0
    N = 0
    Σ_R = 0
    Σ_μ = 0
    Σ_B = 0
    
end

function ReactionVariables(reaction_system::ReactionStructure,response_type="cross")
    
    if response_type == "single"
        
        R = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response)
        R[:,1,1] = ones(reaction_system.num_species)
        
        μ = GreenFunction(zeros(Float64,reaction_system.num_species,1), OnePoint)
        μ[:,1] = reaction_system.initial_values
        
        C = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Symmetrical)
        C[:,1,1] = reaction_system.initial_C #Defines the initial correlations in the system if any
        
        N = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response)
        N[:,1,1] = μ[:,1] + C[:1,1]
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
            N = N,
            Σ_R = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response),
            Σ_B = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Symmetrical),
            Σ_μ = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response))
        
    elseif response_type == "cross"
        
        R = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response)
        R[:,:,1,1] = zeros(reaction_system.num_species,reaction_system.num_species)

        for i in 1:reaction_system.num_species
            R[i,i,1,1] = 1. #Only the diagonal responses take the equal time value of one!
        end
        
        μ = GreenFunction(zeros(Float64,reaction_system.num_species,1), OnePoint)
        μ[:,1] = reaction_system.initial_values
        
        #The cross response functions should not be Symmetric only in time but also in species!
        C = zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1)

        #TODO: In a future release, replace the correlation with the following 'Symmetrical' GreenFunction which is by default 'Block' Symmetrical
        # C = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Symmetrical)
        
        if any(reaction_system.initial_C .!= 0)
            @warn "Non-zero initial correlations are currently not supported but will be added in future versions. Please open a feature request or Github issue if you need this functionality."
        end

        if size(reaction_system.initial_C) == size(C[:,:,1,1])
            C[:,:,1,1] = reaction_system.initial_C #Defines the initial correlations in the system if any
        elseif size(reaction_system.initial_C) == size(C[:,1,1,])
            if any(reaction_system.initial_C .!= 0)
                @warn "Initial correlations are non-zero and correlation matrix size mismatch. Using diagonal elements only."
            end
            for i in 1:reaction_system.num_species
                C[i,i,1,1] = reaction_system.initial_C[i]
            end
        else
            @warn "Initial correlation matrix size mismatch. Using all zero initial correlations."
            C[:,:,1,1] = zeros(reaction_system.num_species,reaction_system.num_species)
        end

        N = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response)
        N[:,:,1,1] = C[:,:,1,1]
        for i in range(1,reaction_system.num_species)
            N[i,i,1,1] += μ[i,1]
        end
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
            N = N,
            Σ_R = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response),
            Σ_B = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Symmetrical),
            Σ_μ = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response))
    end

end