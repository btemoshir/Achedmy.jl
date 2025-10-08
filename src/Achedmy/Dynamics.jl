include("SelfEnergy.jl")

function solve_dynamics!(structure,variables; selfEnergy="gSBR", tmax=1., tstart=0., atol=1e-3, rtol=1e-2, k_max = 12, dtini = 0.0, dtmax = Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax_vie = k_max ÷ 2)
    
    """
    Solve the dynamics using KB.jl solver. Changes the variables in place.
    
    Inputs
    ------
    - structure
    - variables
    - atol
    - rtol
    - tmax
    - tstart
    - selfEnergy: string
                What kind of self-energy to use, "gSBR", "SBR", "MAK", "MCA"
                'gSBR'    -- generalized SBR corrections to the self-energy with different n being mixed (Slow)
                'SBR'     -- SBR corrections to the self-energy. Does not mix different n (reaction) vectors, i.e the fluctuations in different reactions are independent. (Faster)
                'MCA'     -- Mode coupling approximation or the O(α^2) corrections to the self-energy, 
                'MAK'     -- Mass action kinetics (MAK) or the Mean field or O(α) corrections to the self-energy
    - for other parameters see kbsolve! documentation

    Outputs
    -------
    - sol: solution object from kbsolve! sol.t with the time steps and sol.w with the weights at each time step (step size)
    - variables: updated in place with the new values of μ, R, C, N and Σ_μ, Σ_R, Σ_B
    """
    
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
        elseif selfEnergy == "gSBRC"
            throw(ErrorException("gSBRC self energy is not implemented yet!"))
            sE = (x...) -> self_energy_SBR_mixed_cross!(structure, variables, x...)
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