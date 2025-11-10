"""
    solve_dynamics!(structure, variables; kwargs...)

Solve the memory-corrected dynamics for a chemical reaction network.

# Arguments

- `structure::ReactionStructure`: Reaction network structure with stoichiometry and rates
- `variables::ReactionVariables`: Storage container for dynamical variables (modified in-place)

## Approximation Method
- `selfEnergy::String = "gSBR"`: Self-energy approximation scheme
  - `"gSBR"`: Generalized self-consistent bubble resummation **Recommended**
  - `"SBR"`: Self-consistent bubble resummation (single-species only, ignores the joint fluctuations induced by different reactions)
  - `"MCA"`: Mode coupling approximation (perturbative to O(α²))
  - `"MAK"`: Mass action kinetics (mean-field, no memory)

## Time Parameters
- `tstart::Float64 = 0.0`: Initial time
- `tmax::Float64 = 1.0`: Final simulation time

## Solver Tolerances
- `atol::Float64 = 1e-3`: Absolute tolerance for adaptive solver
- `rtol::Float64 = 1e-2`: Relative tolerance for adaptive solver

## Advanced Solver Parameters
- `k_max::Int = 12`: Maximum interpolation order
- `dtini::Float64 = 0.0`: Initial time step (0 = auto)
- `dtmax::Float64 = Inf`: Maximum time step
- `qmax::Float64 = 5`: Maximum step size growth factor
- `qmin::Float64 = 1//5`: Minimum step size shrink factor
- `γ::Float64 = 9//10`: Safety factor for step size adaptation
- `kmax_vie::Int = k_max ÷ 2`: Maximum order for Volterra integral equations

# Returns

- `sol`: Solution object from `kbsolve!` with fields:
  - `t::Vector{Float64}`: Adaptive time grid
  - `w::Vector{Vector{Float64}}`: Integration weights at each time step
  - `retcode::Symbol`: Solution status (`:Success` if converged)

# Side Effects

The `variables` object is modified in-place with computed values:
- `variables.μ`: Mean trajectories
- `variables.R`: Response functions
- `variables.C`: Correlation functions
- `variables.N`: Number-number correlators
- `variables.Σ_R`, `variables.Σ_μ`, `variables.Σ_B`: Self-energies

# Algorithm

Solves coupled integro-differential Kadanoff-Baym equations:

```math
\\begin{aligned}
\\partial_t \\mu_i(t) &= k_{1i} - k_{2i} \\mu_i(t) + \\Sigma_\\mu^{i}(t) \\\\
(\\partial_t &+ k_{2i}) R_{ij}(t,t') &= \\delta(t-t') \\delta_{ij} + \\int_{t'}^{t} d\\tau'' \\, \\sum_k \\Sigma_R^{ik}(t,\\tau'') R_{kj}(\\tau'',t')
\\end{aligned}
```

where self-energies \$\\Sigma\$ are computed according to the chosen approximation method.

# Approximation Methods Explained

## gSBR (Generalized Self-consistent Bubble Resummation) ⭐
- **Accuracy**: Best available, validated against numerical solutions of the master equation
- **Includes**: All cross-species and cross-reaction corrections and full memory effects
- **Use when**: Accuracy is critical, small molecule numbers
- **Works with**: Both "single" and "cross" response types

## SBR (Self-consistent Bubble Resummation)
- **Accuracy**: Good for weakly coupled systems
- **Includes**: Only diagonal self-energies, no cross-species nor cross-reaction memory corrections
- **Use when**: Species are weakly coupled
- **Restriction**: Only works with "single" response type

## MCA (Mode Coupling Approximation)
- **Accuracy**: Perturbative O(α²) correction to MAK
- **Includes**: Leading-order fluctuation corrections
- **Use when**: Fluctuations are weak
- **Unstable**: May lead to numerical instabilities in most cases with large reaction rates
- **Works with**: Both "single" and "cross" response types

## MAK (Mass Action Kinetics)
- **Accuracy**: Mean-field only, no fluctuations
- **Includes**: No memory effects
- **Use when**: Quick baseline comparison needed
- **Limitation**: Fails for small molecule counts

# Example

```julia
using Achedmy, Catalyst

# Define Michaelis-Menten enzyme kinetics
enzyme = @reaction_network begin
    @species S(t)=1.0 E(t)=0.9 C(t)=0.1 P(t)=0.0
    @parameters k_f=1.0 k_b=0.1 k_cat=1.0
    (k_f, k_b), S + E <--> C
    k_cat, C --> E + P
end

# Setup
structure = ReactionStructure(enzyme)
variables = ReactionVariables(structure, "cross")

# Solve with high accuracy
sol = solve_dynamics!(
    structure, 
    variables,
    selfEnergy = "gSBR",
    tmax = 5.0,
    tstart = 0.0,
    atol = 1e-4,
    rtol = 1e-3
)

# Check convergence
@assert sol.retcode == :Success "Solver did not converge!"

# Extract results
time = sol.t
mean_product = variables.μ[4, :]  # P(t)
variance_product = diag(variables.N[4,4,:,:])  # Var[P(t)]
response_SE = variables.R[1,2,:,:]  # How E responds to perturbation in S

# Visualize
using PyPlot
figure(figsize=(10,4))
subplot(121)
plot(time, mean_product)
xlabel("Time")
ylabel("⟨P(t)⟩")

subplot(122)
plot(time, variance_product)
xlabel("Time")
ylabel("Var[P(t)]")
```

# Performance Tips

1. **Start loose, tighten if needed**: Begin with `atol=1e-3`, `rtol=1e-2`
2. **Time range**: Longer `tmax` increases cost quadratically or cubically due to memory integrals
3. **Method selection**:
   - Production runs: `gSBR` with `"cross"`
   - Large systems (>10 species): `gSBR` with `"single"`
   - Quick tests: `MAK` or `MCA`
4. **Tolerance tuning**: If variances go negative, decrease tolerances

# Common Issues

## Negative Variances
**Problem**: `diag(variables.N[i,i,:,:])` contains negative values

**Solutions**:
- Decrease `atol` and `rtol` (e.g., `1e-5`)
- Use `"cross"` response type instead of `"single"`
- Reduce `tmax` or increase `dtmax` to limit time range
- Check that initial conditions are physical (non-negative)

## Slow Convergence
**Problem**: Simulation takes too long

**Solutions**:
- Increase tolerances to `atol=1e-2`, `rtol=1e-1`
- Switch from `"cross"` to `"single"` response type
- Use `SBR` or `MCA` instead of `gSBR`
- Reduce `tmax` or limit time range

## SBR with cross response error
**Problem**: Error when using `selfEnergy="SBR"` with `response_type="cross"`

**Solution**: Use `selfEnergy="gSBR"` or change to `response_type="single"`

# See Also

- [`ReactionStructure`](@ref): Define reaction networks
- [`ReactionVariables`](@ref): Storage container
- [KadanoffBaym.jl](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/): Underlying solver
- [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/): Reaction network DSL
"""
function solve_dynamics!(structure,variables; selfEnergy="gSBR", tmax=1., tstart=0., atol=1e-3, rtol=1e-2, k_max = 12, dtini = 0.0, dtmax = Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax_vie = k_max ÷ 2)
    
    #Define which self-energy to use:
    if variables.response_type == "single"
        if selfEnergy     == "gSBR"
            sE = (x...) -> self_energy_SBR_mixed!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            sE = (x...) -> self_energy_SBR!(structure, variables, x...)
        elseif selfEnergy == "MAK"
            sE = (x...) -> self_energy_mak_noC!(structure, variables, x...)
        elseif selfEnergy == "MCA"
            sE = (x...) -> self_energy_alpha2!(structure, variables, x...)
        end
    elseif variables.response_type == "cross"
        if selfEnergy == "gSBR"
            sE = (x...) -> self_energy_SBR_mixed_cross_noC!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            throw(ErrorException("SBR self energy is only available when using single species response functions. For cross species response functions use gSBR self energy instead."))  
        #    sE = (x...) -> self_energy_SBR_cross!(structure, variables, x...)
        elseif selfEnergy == "MAK"
            sE = (x...) -> self_energy_mak_noC!(structure, variables, x...)
        elseif selfEnergy == "MCA"
            sE = (x...) -> self_energy_alpha2_cross!(structure, variables, x...)
        end
    end

    if variables.response_type == "single"
        @time sol = kbsolve!(
        (x...) -> fv!(structure, variables, x...),
        (x...) -> fd!(structure, variables, x...),
        [variables.R],
        (tstart, tmax);
        callback =  sE,
        atol = atol,
        rtol = rtol,
        stop = x -> (println("t: $(x[end])"); flush(stdout); false),
        v0 = [variables.μ],
        f1! = (x...) -> f1!(structure, variables, x...),
        kmax = k_max, dtini = dtini, dtmax = dtmax, qmax = qmax, qmin = qmin, γ = γ,kmax_vie=kmax_vie)

        #Calculate C here separately for the calculation at the last time step!
        if isdefined(variables, :C)
            #Check the type of variables.C and handle accordingly
            if isa(variables.C, GreenFunction)
                if (n = size(variables.R, 2)) > size(variables.C, 2)
                    resize!(variables.C, n)
                end
            else
                #TODO: Retain the original values of C if the initial correlation is defined!
                n = size(variables.R, 2)
                variables.C = zeros(Float64,structure.num_species,n,n)
            end
        else
            n = size(variables.R, 2)
            variables.C = zeros(Float64,structure.num_species,n,n)
        end

        #Calculate C here separately for the calculation at the last time step!
        if (n = size(variables.R, 2)) > size(variables.C, 2)
            resize!(variables.C, n)
        end
        
        #Update the correlation functions!
        t = length(sol.w)
        for j in 1:structure.num_species
            variables.C[j,:,:] .= 0.
            R1 = collect(ttt <= tt ? variables.R[j,tt,ttt] : 0 for tt in 1:t, ttt in 1:t)
            ΣB = collect(variables.Σ_B[j,tt,ttt].*sol.w[tt][tt] for tt in 1:t, ttt in 1:t)
            R2 = collect(ttt <= tt ? variables.R[j,tt,ttt].*sol.w[tt][ttt] : 0 for tt in 1:t, ttt in 1:t)
            variables.C[j,1:t,1:t] += R1*(ΣB*transpose(R2))                   
            
            #Old update -- IGNORE ---
            # R1 = collect(variables.R[j,tt,ttt] for tt in 1:t, ttt in 1:t) 
            # ΣB = collect(variables.Σ_B[j,tt,ttt] for tt in 1:t, ttt in 1:t)
            # for tt in 1:t
            #     #ΣB[tt,1:tt] .*= sol.w[tt][1:tt].^2
            #     ΣB[tt,1:tt] .*= sol.w[tt][1:tt].*sol.w[t][1:tt]
            # end
            # R2 = collect(variables.R[j,ttt,tt] for tt in 1:t, ttt in 1:t)
            # variables.C[j,1:t,1:t] += R1*(ΣB*R2)
        end

        #Calculate the number-number correlator here!
        if (n = size(variables.R, 2)) > size(variables.N, 2)
            resize!(variables.N, n)
        end

        for tt in 1:t
            for i in 1:structure.num_species
                variables.N[i,tt,1:t] = variables.C[i,tt,1:t] .+ variables.μ[i,1:t] .* variables.R[i,tt,1:t]
            end
        end
                
    elseif variables.response_type == "cross"
        @time sol = kbsolve!(
        (x...) -> fv_cross!(structure, variables, x...),
        (x...) -> fd!(structure, variables, x...),
        [variables.R],
        (tstart, tmax);
        callback =  sE,
        atol = atol,
        rtol = rtol,
        stop = x -> (println("t: $(x[end])"); flush(stdout); false),
        v0 = [variables.μ],
        f1! = (x...) -> f1!(structure, variables, x...),
        kmax = k_max, dtini = dtini, dtmax = dtmax, qmax = qmax, qmin = qmin, γ = γ,kmax_vie=kmax_vie)

        #Calculate C here separately for the calculation at the last time step!
        if isdefined(variables, :C)
            #Check the type of variables.C and handle accordingly
            if isa(variables.C, GreenFunction)
                if (n = size(variables.R, 3)) > size(variables.C, 3)
                    resize!(variables.C, n)
                end
            else
                #TODO: Retain the original values of C if the initial correlation is defined!
                n = size(variables.R, 3)
                variables.C = zeros(Float64,structure.num_species,structure.num_species,n,n)
            end
        else
            n = size(variables.R, 3)
            variables.C = zeros(Float64,structure.num_species,structure.num_species,n,n)
        end

        #Update the correlation functions!
        t = length(sol.w)
        for j in 1:structure.num_species
            for j2 in 1:structure.num_species
                variables.C[j,j2,:,:] .= 0.
                for j_sum1 in 1:structure.num_species
                    for j_sum2 in 1:structure.num_species

                        R1 = collect(ttt <= tt ? variables.R[j,j_sum1,tt,ttt] : 0 for tt in 1:t, ttt in 1:t)                            

                        ΣB = collect(variables.Σ_B[j_sum1,j_sum2,tt,ttt].*sol.w[tt][tt] for tt in 1:t, ttt in 1:t)

                        R2 = collect(ttt <= tt ? variables.R[j2,j_sum2,tt,ttt].*sol.w[tt][ttt] : 0 for tt in 1:t, ttt in 1:t)

                        variables.C[j,j2,1:t,1:t] += R1 * (ΣB * transpose(R2))

                    end
                end
            end
        end

        #Calculate the number-number correlator here!
        if (n = size(variables.R, 3)) > size(variables.N, 3)
            resize!(variables.N, n)
        end

        for tt in 1:t
            for i in 1:structure.num_species
                for j in 1:structure.num_species
                    variables.N[i,j,tt,1:t] = variables.C[i,j,tt,1:t] .+ variables.μ[j,1:t] .* variables.R[i,j,tt,1:t]
                end
            end
        end

    else
        throw(ErrorException("Unknown response type $(variables.response_type). Should be either 'single' or 'cross'"))

    end
    
    return sol
end

function integrate1(hs::Vector, t1, Σ::GreenFunction, μ::GreenFunction; tmax=t1)
    """
    To integrate the self-energy correction to the mean
    """

    retval = zero(μ[t1])

    for k in 1:tmax
        retval +=  Σ[t1,k]*hs[k]
    end

    return retval
end

function integrate2(hs::Vector, t1, t2, Σ::GreenFunction, R::GreenFunction, μ::GreenFunction; tmax=t1)
    """
    To integrate the self-energy corrections to the response
    """

    retval = zero(R[t1,t2])

    #for k in t2+1:t1
    for k in t2:t1
        retval += Σ[t1,k].*R[k,t2].*hs[k]
    end

    return retval
end

function integrate2_cross(hs::Vector, t1, t2, Σ::GreenFunction, R, μ::GreenFunction; tmax=t1)
    """
    To integrate the self-energy corrections to the response
    """

    retval = zero(R[:,:,t1,t2])
    num_species = length(μ[1])

    for j in 1:num_species
        for j2 in 1:num_species
            #for k in t2+1:t1
            for j_sum in 1:num_species
                for k in t2:t1
                    retval[j,j2] += Σ[j,j_sum,t1,k].*R[j_sum,j2,k,t2].*hs[k] #Need to add a shift here? --IMP (PAY ATTENTION!)
                end
            end
        end
    end

    return retval
    
end

function fv!(structure, variables, out, times, h1, h2, t, t′)
    """
    Vertical evolution
    """

    if t == 1
        #self_energy_mak!(structure, variables, times, h1, h1 , t, t) #-- not needed called while calling f1
        corr = variables.Σ_R[t,t]

    else
        corr = integrate2(h1, t, t′, variables.Σ_R, variables.R, variables.μ)
    end

    retval  = zero(variables.R[t,t′])

    for j in 1:structure.num_species
        retval[j] = -structure.rate_destruction[j].*variables.R[j,t,t′]
    end

    out[1]  = retval .+ corr

end

function fv_cross!(structure, variables, out, times, h1, h2, t, t′)
    """
    Vertical evolution for cross responses
    """

    if t == 1
        corr = variables.Σ_R[t,t]
    else
        corr = integrate2_cross(h1, t, t′, variables.Σ_R, variables.R, variables.μ)
    end

    retval  = zero(variables.R[t,t′])

    for j in 1:structure.num_species
        for j2 in 1:structure.num_species
            retval[j,j2] = -structure.rate_destruction[j].*variables.R[j,j2,t,t′]
        end
    end

    # The relaxation of this constraint for cross responses leads to much better results! (ref SIR model) 
    # #Checks the value of R_ii. If its negative or greater than 1, it outputs the vertical evolution to be zero!
    # for i in 1:structure.num_species
    #     if variables.R[i,i,t,t′] < 0 || variables.R[i,i,t,t′] > 1
    #         corr[i,i] = 0
    #     end
    # end

    out[1]  = retval .+ corr

end

function fd!(structure, variables, out, times, h1, h2, t, t′)
    """
    Diagonal evolution
    """

    out[1] = zero(out[1])

end


function f1!(structure, variables, out, times, h1, t)
    """
    Evolution for the mean
    """

    if t == 1

        self_energy_mak_noC!(structure, variables, times, h1, h1 , t, t)
        corr = variables.Σ_μ[t,t]

    else
        corr = integrate1(h1, t, variables.Σ_μ, variables.μ)
    end

    retval   = zero(variables.μ[t])

    for j in 1:structure.num_species
        retval[j] = structure.rate_creation[j] - structure.rate_destruction[j]*variables.μ[j,t]
    end

    #Checks the value of \mu_i. If its negative, it outputs the vertical evolution to be zero!
    for i in 1:structure.num_species
        if variables.μ[i,t] < 0
            corr[i] = 0
        end
    end

    out[1]  = retval .+ corr

end