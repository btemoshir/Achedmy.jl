include("BlockOp.jl")
include("Cmn.jl")

"""
    self_energy_mak_noC!(structure, variables, times, h1, h2, t, t′)

Compute self-energy in the Mass Action Kinetics (MAK) approximation.

This is the simplest mean-field approximation with no memory corrections beyond
O(α). All fluctuations and correlations are neglected.

In the MAK approximation, the self-energies are diagonal and time-local.

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container for dynamical variables (modified in-place)
- `times::Vector{Float64}`: Time grid
- `h1::Vector{Float64}`: Integration weights for single time integrals
- `h2::Matrix{Float64}`: Integration weights for double time integrals  
- `t::Int`: Current time index
- `t′::Int`: Past time index (for causality: t′ ≤ t)

No memory integrals are computed—all corrections are instantaneous.

# Side Effects

- Modifies `variables.Σ_μ`, `variables.Σ_R`, `variables.Σ_B` in-place
- Only updates at `t′ == 1` since we don't have direct two-time integrals
- Automatically resizes self-energy arrays if needed

# Performance

- **Cost**: O(N) per time step where N = num_species
- **Memory**: O(NT) where T = number of time steps
- **Fastest** of all methods, suitable for quick baseline comparisons

# Example

```julia
# MAK is used internally by solve_dynamics!
sol = solve_dynamics!(structure, variables, selfEnergy="MAK", tmax=10.0)

# Self-energies are time-local (diagonal in time)
@assert all(variables.Σ_R[:, 5, 1:4] .≈ 0)  # Off-diagonal times are zero
```

# See Also

- [`self_energy_alpha2!`](@ref): MCA approximation (O(α²))
- [`self_energy_SBR!`](@ref): SBR approximation  
- [`self_energy_SBR_mixed!`](@ref): gSBR approximation
- [`c_mnFULL`](@ref): Coefficient calculation
"""
function self_energy_mak_noC!(structure, variables, times, h1, h2 , t, t′)

    # Resize self-energies when Green functions are resized
    if variables.response_type == "single"
        if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
        end
    elseif variables.response_type == "cross"
        if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
        end
        
    end

    if t′ == 1

        if variables.response_type == "single"

            variables.Σ_μ[:,t,1:t] .= 0.
            variables.Σ_R[:,t,1:t] .= 0.
            variables.Σ_B[:,t,1:t] .= 0.
            
            temp0  = zeros(Int,structure.num_species)
    
            for i in 1:structure.num_species
                temp1    = zeros(Int,structure.num_species)
                temp1[i] = 1
    
                if t == 1
                    #No division by step size at initial time because its zero
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)
                else            
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)./h1[t]
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)./h1[t]
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)./h1[t]
                end
                    
            end
        end

        if variables.response_type == "cross"
                        
            variables.Σ_μ[:,t,1:t]   .= 0.
            variables.Σ_R[:,:,t,1:t] .= 0.
            variables.Σ_B[:,:,t,1:t] .= 0.
            
            temp0  = zeros(Int,structure.num_species)
    
            for i in 1:structure.num_species
                temp1    = zeros(Int,structure.num_species)
                temp1[i] = 1

                #No deivision by step size at initial time because its zero
                if t == 1
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)
                else
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)./h1[t]

                end

                for i_2 in 1:structure.num_species
                    
                    temp2    = zeros(Int,structure.num_species)
                    temp2[i_2] = 1
    
                    if t == 1
                        variables.Σ_R[i,i_2,t,t] = c_mnFULL(structure,variables,temp1,temp2,t)
                        variables.Σ_B[i,i_2,t,t] = prod(factorial.(temp1.+temp2)).* c_mnFULL(structure,variables,temp1+temp2,temp0,t+1)
                    else
                        variables.Σ_R[i,i_2,t,t] = c_mnFULL(structure,variables,temp1,temp2,t)./h1[t]
                        variables.Σ_B[i,i_2,t,t] = prod(factorial.(temp1.+temp2)).* c_mnFULL(structure,variables,temp1+temp2,temp0,t+1)./h1[t]
                    end                
                end
            end
        end
    end
end

"""
    self_energy_alpha2!(structure, variables, times, h1, h2, t, t′)

Compute self-energy in the Mode Coupling Approximation (MCA).

This includes O(α²) perturbative corrections to MAK, capturing leading-order
fluctuation effects. Uses single-species response functions only.

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container for dynamical variables (modified in-place)
- `times::Vector{Float64}`: Time grid
- `h1::Vector{Float64}`: Integration weights
- `h2::Matrix{Float64}`: Double integration weights
- `t::Int`: Current time index
- `t′::Int`: Past time index

This captures the leading-order effect of fluctuations on the mean dynamics.

# Accuracy

- **Improves over MAK**: Includes fluctuation feedback
- **Limitations**: Perturbative, fails for large fluctuations

# Response Type

- Works with `response_type = "single"` only
- For cross-species corrections, use [`self_energy_alpha2_cross!`](@ref)

# Performance

- **Cost**: O(N²T) per time step
- **More expensive** than MAK but cheaper than SBR/gSBR

# Example

```julia
variables = ReactionVariables(structure, "single")
sol = solve_dynamics!(structure, variables, selfEnergy="MCA", tmax=10.0)

# Check that fluctuations are captured
variance = diag(variables.N[1,1,:,:])
@assert any(variance .> 0)  # Non-zero fluctuations
```

# See Also

- [`self_energy_mak_noC!`](@ref): Mean-field baseline (O(α))
- [`self_energy_alpha2_cross!`](@ref): MCA with cross-species correlations
- [`self_energy_SBR!`](@ref): Self-consistent approximation
"""
function self_energy_alpha2!(structure, variables, times, h1, h2 , t, t′)
    """
    O(α^2) corrections to the self-energy
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end

    if t′ == 1

        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.
        variables.Σ_B[:,t,:]   .= 0.

        #Temporary variables declared to hold the field values!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t]) 
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]
            Σ_B_temp[i,t] += 2*c_mnFULL(structure,variables,2*temp1,temp0,t+1)./h1[t]

            # for n ∈ structure.n_list_union
            for n ∈ structure.m_list_union
                if n ∉ [temp0,temp1]

                    Σ_R_temp[i,1:t] += c_mnFULL(structure,variables,temp1,n,t+1).*collect(
                    c_mnFULL(structure,variables,n,temp1,t′)*prod(factorial.(n).*variables.R[t,t′].^n) for t′ in 1:t)
                        
                    Σ_B_temp[i,1:t] += c_mnFULL(structure,variables,temp1,n,t+1).*collect(
                    c_mnFULL(structure,variables,n+temp1,temp0,t′)*prod(factorial.(n+temp1).*variables.R[t,t′].^n) for t′ in 1:t)
                    Σ_B_temp[i,t]   += 2*c_mnFULL(structure,variables,2*temp1,n,t+1)*sum(collect(
                    c_mnFULL(structure,variables,n,temp0,t′)*prod(factorial.(n).*variables.R[t,t′].^n) for t′ in 1:t))
                                                            
                end

                if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0)

                    Σ_μ_temp[i,1:t] += c_mnFULL(structure,variables,temp1,n,t+1).*collect(
                    c_mnFULL(structure,variables,n,temp0,t′)*prod(factorial.(n).*variables.R[t,t′].^n) for t′ in 1:t)

                end
            end
        end

        variables.Σ_R[:,t,1:t] .= Σ_R_temp[:,1:t] 
        variables.Σ_μ[:,t,1:t] .= Σ_μ_temp[:,1:t] 
        variables.Σ_B[:,t,1:t] .= Σ_B_temp[:,1:t]

    end
end
        
"""
    self_energy_SBR!(structure, variables, times, h1, h2, t, t′)

Compute self-energy in the Self-consistent Bubble Resummation (SBR) approximation.

SBR includes self-consistent treatment of single-species fluctuations but ignores
cross-reaction correlations. Each reaction's fluctuations are treated independently.

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container (modified in-place)
- `times::Vector{Float64}`: Time grid  
- `h1, h2`: Integration weights
- `t, t′`: Time indices

The sum over \$n\$ includes all reaction vectors, but cross-reaction terms are neglected.

# Key Differences from gSBR

- **SBR**: Treats each reaction independently
- **gSBR**: Couples all reactions together (more accurate)

# Restrictions

- **Only works with** `response_type = "single"`
- For cross-species, must use [`self_energy_SBR_mixed_cross_noC!`](@ref)

# Performance

- **Cost**: O(N²T²) per iteration
- **Faster** than gSBR, **slower** than MCA
- Good compromise for weakly coupled systems

# Example

```julia
variables = ReactionVariables(structure, "single")  # Required
sol = solve_dynamics!(structure, variables, selfEnergy="SBR", tmax=10.0)
```

# See Also

- [`self_energy_SBR_mixed!`](@ref): gSBR with full coupling
- [`self_energy_SBR_mixed_cross_noC!`](@ref): gSBR with cross-species
"""
function self_energy_SBR!(structure, variables, times, h1, h2 , t, t′)
    """
    SBR corrections to the self-energy. Does not mix different n , i.e. fluctuations from different reactions are not mixed here.    
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end
            
    if t′ == 1

        temp0  = zeros(Int,structure.num_species)
        id     = diagm(ones(t))
        L      = diagm(-1=>ones(t-1))
        
        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.
        variables.Σ_B[:,t,:]   .= 0.

        #Temporary variables declared to hold the field values!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t]) 
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,t,1:t])

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]
            Σ_B_temp[i,t] += 2*c_mnFULL(structure,variables,2*temp1,temp0,t+1)./h1[t]

        end

        for n in structure.m_list_union

            cNN = collect(c_mnFULL(structure,variables,n,n,tt) for tt in 1:t)
                        
            Γ   = collect(prod(factorial.(n) .* variables.R[:,tt,ttt] .^n) for tt in 1:t, ttt in 1:t)                
            cN0 = collect(c_mnFULL(structure,variables,n,temp0,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)

            #The following creates the \Chi matrix (with the shift), but also multiples the columns by the time step size
            χ   = collect(cNN[ttt].*Γ[tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*L
            
                    
            Ξ   = tril(id .- χ)      #Make the matrix ecplicitly lower triangular!
            LAPACK.trtri!('L','U', Ξ) #LAPAC functions to invert the triangular matrix here!
            ΞcN0 = Ξ*cN0

            for i in 1:structure.num_species

                temp1    = zeros(Int,structure.num_species)
                temp1[i] = 1                                

                if n ∉ [temp0,temp1] && c_mnFULL_test(structure,variables,n,temp1) != 0 && c_mnFULL_test(structure,variables,temp1,n) != 0
                    cN1  = collect(c_mnFULL(structure,variables,n,temp1,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)
                    ΞcN1 = Ξ*cN1
                    
                    Σ_R_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1)[t,1:t]
                end
                    
                if n ∉ [temp0,temp1]
                                
                    Σ_B_temp[i,t]   += (2*c_mnFULL(structure,variables,2*temp1,n,t+1).*sum(ΞcN0,dims=2))[t]
                                
                    Γ_MN  = collect(prod(factorial.(temp1+n) .* variables.R[:,tt,ttt] .^n) for tt in 1:t, ttt in 1:t) 
                    cN10  = collect(c_mnFULL(structure,variables,n+temp1,temp0,ttt).*Γ_MN[tt,ttt] for tt in 1:t, ttt in 1:t)
                    ΞcN10 = Ξ*cN10
                                
                    Σ_B_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN10)[t,1:t]
                    
                    cN1N  = collect(c_mnFULL(structure,variables,n+temp1,n,ttt).*Γ_MN[tt,ttt] for tt in 1:t, ttt in 1:t)
                    ΞcN1N = Ξ*cN1N
                                
                    Σ_B_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1N)[t,1:t].*sum(ΞcN0,dims=2)[1:t].*h1[1:t]
                            
                end

                if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0)

                    Σ_μ_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN0)[t,1:t]

                end
            end
        end
                
        variables.Σ_R[:,t,1:t] = deepcopy(Σ_R_temp[:,1:t]) 
        variables.Σ_μ[:,t,1:t] = deepcopy(Σ_μ_temp[:,1:t]) 
        variables.Σ_B[:,t,1:t] = deepcopy(Σ_B_temp[:,1:t])

    end
end

"""
    self_energy_SBR_mixed!(structure, variables, times, h1, h2, t, t′)

Compute self-energy in the Generalized Self-consistent Bubble Resummation (gSBR) approximation.

This is the more accurate approximation, including full self-consistent treatment
of all cross-reaction correlations and memory effects. Works with _only_ single-species response functions.

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container (modified in-place, `response_type="single"`)
- `times`, `h1`, `h2`: Time grid and integration weights
- `t, t′`: Current and past time indices

# Key Features

- **Self-consistency**: Self-energy depends on itself through matrix inversion
- **Full coupling**: All cross-reaction correlations included  
- **Memory**: Non-Markovian effects captured
- **Accuracy**: Validates against master equation for small systems

# Algorithm

Uses block matrix inversion via [`block_tri_lower_inverse`](@ref) to efficiently
compute the self-consistent solution.

# Computational Cost

- **Most expensive** method but also most accurate
- Uses block operations to optimize performance

# Example

```julia
variables = ReactionVariables(structure, "single")
sol = solve_dynamics!(structure, variables, selfEnergy="gSBR", 
                     tmax=10.0, atol=1e-4)

# gSBR captures strong correlations accurately
@test norm(variables.μ - master_mean) < 0.01  # Close to exact
```

# See Also

- [`self_energy_SBR!`](@ref): Simplified version without cross-reaction coupling
- [`self_energy_SBR_mixed_cross_noC!`](@ref): gSBR with cross-species responses
- [`block_tri_lower_inverse`](@ref): Matrix inversion algorithm
"""
function self_energy_SBR_mixed!(structure, variables, times, h1, h2 , t, t′)    

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end        

    if t′ == 1 
    # Only do the self-energy calcultion for teh first value of t'

        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.
        variables.Σ_B[:,t,:] .= 0.

        #Temporary variables to store the value of the self-energy!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t])
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]
            Σ_B_temp[i,t] += 2*c_mnFULL(structure,variables,2*temp1,temp0,t+1)./h1[t]

        end

        #We will do this first for \mu and then for R            
        #Creating the a list which has the non-zero entries for \Sigma_mu
        n_listNEW_μ = []

        for n in structure.m_list_union
            if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0) && c_mnFULL_test(structure,variables,n,temp0) != 0
                push!(n_listNEW_μ,n)                
            end
        end

        cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_μ, n′′ in n_listNEW_μ,  tt in 1:t)
        Γ   = collect(prod(factorial.(n′) .* variables.R[:,tt,ttt] .^n′) for n′ in n_listNEW_μ, tt in 1:t, ttt in 1:t)        
        cN0 = collect(c_mnFULL(structure,variables,n_listNEW_μ[n′],temp0,ttt).*Γ[n′,tt,ttt] for n′ in 1:length(n_listNEW_μ), tt in 1:t, ttt in 1:t)
        χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_μ), n′′ in 1:length(n_listNEW_μ), tt in 1:t, ttt in 1:t )
        Ξ   = block_tri_lower_inverse(block_identity(length(n_listNEW_μ),t).-block_lower_shift(χ))
        Ξ2  = block_mat_mix_mul(Ξ,cN0)
        Ξ_B = sum(block_mat_mix_mul(Ξ,cN0) .* reshape(h1, 1, 1, t), dims=3)[:,t]

        for i in 1:structure.num_species
            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1
            c1N      = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_μ)
            c2N      = 2*collect(c_mnFULL(structure,variables,2*temp1,n′,t+1) for n′ in n_listNEW_μ)

            Σ_μ_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]
            Σ_B_temp[i,t] += dot(c2N,Ξ_B)./h1[t]

        end

        #Now we do the calculation for \Sigma_R species wise!
        for i in 1:structure.num_species

            temp1       = zeros(Int,structure.num_species)
            temp1[i]    = 1
            n_listNEW_R = []

            for n in structure.m_list_union
                if n ∉ [temp0,temp1] && c_mnFULL_test(structure,variables,n,temp1) != 0 && c_mnFULL_test(structure,variables,temp1,n) != 0
                    push!(n_listNEW_R,n)
                end
            end

            if length(n_listNEW_R) > 0

                cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t)
                Γ   = collect(prod(factorial.(n′) .* variables.R[:,tt,ttt] .^n′) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t)        
                χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_R), n′′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)
                Ξ   = block_tri_lower_inverse(block_identity(length(n_listNEW_R),t).-block_lower_shift(χ))
                cN1 = collect(c_mnFULL(structure,variables,n_listNEW_R[n′],temp1,ttt).*Γ[n′,tt,ttt] for n′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)
                Ξ2  = block_mat_mix_mul(Ξ,cN1)
                c1N = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_R)

                Σ_R_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]
                
            end
        end

        variables.Σ_R[:,t,1:t] .= Σ_R_temp[:,1:t]
        variables.Σ_μ[:,t,1:t] .= Σ_μ_temp[:,1:t]
        variables.Σ_B[:,t,1:t] .= Σ_B_temp[:,1:t]

    end

end

"""
    self_energy_SBR_mixed_cross_noC!(structure, variables, times, h1, h2, t, t′)

Compute gSBR self-energy with **full cross-species** response functions.

This is the most complete and accurate approximation available in Achedmy, capturing:
- All cross-species correlations
- All cross-reaction correlations  
- Full memory effects
- Complete network coupling

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container (modified in-place, `response_type="cross"` required)
- `times`, `h1`, `h2`: Time grid and weights
- `t, t′`: Time indices

This includes **all** pairwise species correlations \$N_{ij}(t,t')\$.

# When to Use

- **Default choice** for most accurate results
- Required when cross-species correlations are important
- Necessary for systems with strong coupling between species

# Computational Cost

- Most expensive but also most accurate

# Example

```julia
# Enzyme kinetics with strong S-E coupling
variables = ReactionVariables(structure, "cross")  # Required
sol = solve_dynamics!(structure, variables, selfEnergy="gSBR", tmax=5.0)

# Access cross-species correlations
cross_corr_SE = variables.N[1, 2, :, :]  # S-E correlation
```

# Performance Tips

- Use `"single"` response type if cross-correlations are weak
- Reduce `tmax` or increase tolerances if too slow
- Consider SBR for quick exploratory runs

# See Also

- [`self_energy_SBR_mixed!`](@ref): gSBR with single-species responses
- [`self_energy_SBR!`](@ref): Simplified SBR
- [`block_tri_lower_inverse`](@ref): Matrix inversion used internally
"""
function self_energy_SBR_mixed_cross_noC!(structure, variables, times, h1, h2 , t, t′)

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end

    if t′ == 1 
    # Only do the self-energy calcultion for the first value of t'

        variables.Σ_R[:,:,t,1:t] .= 0.
        variables.Σ_μ[:,t,1:t]   .= 0.
        variables.Σ_B[:,:,t,:]   .= 0.

        #Temporary variables to store the value of the self-energy!
        Σ_R_temp = zero(variables.Σ_R[:,:,t,1:t])
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]

            for i_2 in 1:structure.num_species

                temp2    = zeros(Int,structure.num_species)
                temp2[i_2] = 1
                
                Σ_R_temp[i,i_2,t] += c_mnFULL(structure,variables,temp1,temp2,t+1)./h1[t]
                
                Σ_B_temp[i,i_2,t] += prod(factorial.(temp1.+temp2))*c_mnFULL(structure,variables,temp1.+temp2,temp0,t+1)./h1[t]
            end
        end

        n_listNEW_R = []
        
        for n in structure.m_list_union
            if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0) && sum(n) < 3
                push!(n_listNEW_R,n)
            end
        end

        if length(n_listNEW_R) > 0 

            cMN = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)

            Γ   = collect(==(sum(n′),sum(n′′)).*response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
            
            χ = block_mat_mul(Γ, cMN .* reshape(h1, 1, 1, 1, t))
            
            # ------- OLD ---------------

            #cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t)
            
            #Γ   = collect(prod(factorial.(n′) .* variables.R[:,tt,ttt] .^n′) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t) 
            
            #χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_R), n′′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)
            
            #------- OLD END ------------

            Ξ  = block_tri_lower_inverse(block_identity(length(n_listNEW_R),t) .- block_lower_shift(χ))

            cN0 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp0,ttt) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t)  
            cN0 = block_mat_mix_mul(Γ,cN0)

            Ξ_μ = block_mat_mix_mul(Ξ,cN0)
            Ξ_B = sum(block_mat_mix_mul(Ξ,cN0) .*reshape(h1, 1, 1, t) ,dims=3)[:,t]

            for i in 1:structure.num_species
    
                temp1       = zeros(Int,structure.num_species)
                temp1[i]    = 1
    
                c1N = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_R)
                
                Σ_μ_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ_μ)[t,1:t]


                for j in 1:structure.num_species

                    temp2       = zeros(Int,structure.num_species)
                    temp2[j]    = 1
                                                        
                    cN1 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp2,ttt) for n′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)
                    cN1 = block_mat_mix_mul(Γ,cN1)

                    Ξ2  = block_mat_mix_mul(Ξ,cN1)

                    Σ_R_temp[i,j,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]

                    c2N = prod(factorial.(temp1.+temp2)).*collect(c_mnFULL(structure, variables, temp1 .+ temp2, n′, t+1) for n′ in n_listNEW_R)
                    
                    Σ_B_temp[i,j,t] += dot(c2N,Ξ_B)./h1[t]

                end
                        
            end

        end

        variables.Σ_R[:,:,t,1:t] .= Σ_R_temp[:,:,1:t] 
        variables.Σ_μ[:,t,1:t]   .= Σ_μ_temp[:,1:t]
        variables.Σ_B[:,:,t,1:t] .= Σ_B_temp[:,:,1:t]

    end

end

"""
    self_energy_alpha2_cross!(structure, variables, times, h1, h2, t, t′)

Compute MCA (Mode Coupling Approximation) self-energy with **cross-species** response functions.

This extends the O(α²) perturbative expansion to include cross-species correlations:
- Captures pairwise species correlations
- Includes cross-reaction coupling
- More accurate than single-species MCA
- Less expensive than full gSBR

# Arguments

- `structure::ReactionStructure`: Reaction network structure
- `variables::ReactionVariables`: Container (modified in-place, `response_type="cross"` required)
- `times`, `h1`, `h2`: Time grid and weights
- `t, t′`: Time indices


# When to Use

- When cross-species correlations matter but gSBR is too expensive
- Systems with moderate coupling between species
- Good compromise between accuracy and speed

# Computational Cost

- More expensive than single-species MCA, much cheaper than gSBR

# Limitations

- Perturbative (O(α²)): breaks down for large fluctuations
- No self-consistent bubble resummation
- Less accurate than gSBR for strongly coupled systems

# Example

```julia
# Enzyme kinetics with moderate S-E coupling
variables = ReactionVariables(structure, "cross")  # Required
sol = solve_dynamics!(structure, variables, selfEnergy="MCA", tmax=10.0)

# Compare to single-species MCA
variables_single = ReactionVariables(structure, "single")
sol_single = solve_dynamics!(structure, variables_single, selfEnergy="MCA", tmax=10.0)
```

# See Also

- [`self_energy_alpha2!`](@ref): Single-species MCA
- [`self_energy_SBR_mixed_cross_noC!`](@ref): Full gSBR with cross-species
- [`response_combinations`](@ref): Helper for multi-species response products
"""
function self_energy_alpha2_cross!(structure, variables, times, h1, h2 , t, t′)    
    """
    Mode coupling approximation (MCA) i.e. O(α^2) corrections to the self-energy with different n being mixed    
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end

    if t′ == 1 
    # Only do the self-energy calcultion for the first value of t'

        variables.Σ_R[:,:,t,1:t] .= 0.
        variables.Σ_μ[:,t,1:t]   .= 0.
        variables.Σ_B[:,:,t,:]   .= 0.

        #Temporary variables to store the value of the self-energy!
        Σ_R_temp = zero(variables.Σ_R[:,:,t,1:t])
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]

            for i_2 in 1:structure.num_species

                temp2    = zeros(Int,structure.num_species)
                temp2[i_2] = 1
                
                Σ_R_temp[i,i_2,t] += c_mnFULL(structure,variables,temp1,temp2,t+1)./h1[t]
                
                Σ_B_temp[i,i_2,t] += prod(factorial.(temp1.+temp2))*c_mnFULL(structure,variables,temp1.+temp2,temp0,t+1)./h1[t]
            end
        end

        n_listNEW_R = []
        
        for n in structure.m_list_union
            if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0) && sum(n) < 3
                push!(n_listNEW_R,n)
            end
        end

        if length(n_listNEW_R) > 0

            cMN = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)

            Γ   = collect(==(sum(n′),sum(n′′)).*response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
            
            χ = block_mat_mul(Γ, cMN .* reshape(h1, 1, 1, 1, t))

            Ξ = block_identity(length(n_listNEW_R),t)

            cN0 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp0,ttt) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t)  
            cN0 = block_mat_mix_mul(Γ,cN0)

            Ξ_μ = block_mat_mix_mul(Ξ,cN0)
            Ξ_B = sum(block_mat_mix_mul(Ξ,cN0) .*reshape(h1, 1, 1, t) ,dims=3)[:,t]

            for i in 1:structure.num_species
    
                temp1       = zeros(Int,structure.num_species)
                temp1[i]    = 1
    
                c1N = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_R)
                
                Σ_μ_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ_μ)[t,1:t]


                for j in 1:structure.num_species

                    temp2       = zeros(Int,structure.num_species)
                    temp2[j]    = 1
                                                        
                    cN1 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp2,ttt) for n′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)
                    cN1 = block_mat_mix_mul(Γ,cN1)

                    Ξ2  = block_mat_mix_mul(Ξ,cN1)

                    Σ_R_temp[i,j,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]

                    c2N = prod(factorial.(temp1.+temp2)).*collect(c_mnFULL(structure, variables, temp1 .+ temp2, n′, t+1) for n′ in n_listNEW_R)
                    
                    Σ_B_temp[i,j,t] += dot(c2N,Ξ_B)./h1[t]
                end
                        
            end

        end

        variables.Σ_R[:,:,t,1:t] .= Σ_R_temp[:,:,1:t] 
        variables.Σ_μ[:,t,1:t]   .= Σ_μ_temp[:,1:t]
        variables.Σ_B[:,:,t,1:t] .= Σ_B_temp[:,:,1:t]

    end

end