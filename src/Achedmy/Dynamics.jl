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
                'SBR'     --  SBR corrections to the self-energy. Does not mix different n vectors
                'gSBR'    --  generalized SBR corrections to the self-energy with different n being mixed (Slow)
                'MCA'     -- Mode coupling approximation or the O(α^2) corrections to the self-energy, 
                #TODO --  with cross responses
                'MAK'     -- Mass action kinetics (MAK) or the Mean field or O(α) corrections to the self-energy

    TODO: Resize the self-energies to the size of the response function!
    """
    
    #Define which self-energy to use:

    if variables.response_type == "single"
        if selfEnergy     == "gSBR"
            sE = (x...) -> self_energy_SBR_mixed!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            sE = (x...) -> self_energy_SBR!(structure, variables, x...)
        elseif selfEnergy == "MAK"
            sE = (x...) -> self_energy_mak_noC!(structure, variables, x...)
            #TODO -- Implement without the C calculation self-consistently!
            #sE = (x...) -> self_energy_mak!(structure, variables, x...)
        elseif selfEnergy == "MCA"
            sE = (x...) -> self_energy_alpha2!(structure, variables, x...)
        end
    elseif variables.response_type == "cross"
        if selfEnergy == "gSBR"
            sE = (x...) -> self_energy_SBR_mixed_cross_noC!(structure, variables, x...)
            #sE = (x...) -> self_energy_SBR_mixed_cross!(structure, variables, x...)
            #TODO: Implement the direct calculation of the self-energy using the C order parameter
        elseif selfEnergy == "gSBRC"
            sE = (x...) -> self_energy_SBR_mixed_cross!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            error("SBR self energy is only available when using single species response functions. For cross species response functions use gSBR self energy instead.")  
        #    sE = (x...) -> self_energy_SBR_cross!(structure, variables, x...)
        elseif selfEnergy == "MAK"
            sE = (x...) -> self_energy_mak_noC!(structure, variables, x...)
            #TODO -- Implement without the C calculation self-consistently!
            #sE = (x...) -> self_energy_mak!(structure, variables, x...)
        elseif selfEnergy == "MCA"
            sE = (x...) -> self_energy_alpha2_cross!(structure, variables, x...)
            #TODO -- Implement this, the O(α^2) corrections to the self-energy
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
        if (n = size(variables.R, 2)) > size(variables.C, 2)
            resize!(variables.C, n)
        end
                
        #Update the correlation functions!
        t = length(sol.w)
        for j in 1:structure.num_species
            variables.C[j,:,:] .= 0.
            R1 = collect(variables.R[j,tt,ttt] for tt in 1:t, ttt in 1:t) 
            ΣB = collect(variables.Σ_B[j,tt,ttt] for tt in 1:t, ttt in 1:t)
            for tt in 1:t
                #ΣB[tt,1:tt] .*= sol.w[tt][1:tt].^2
                ΣB[tt,1:tt] .*= sol.w[tt][1:tt].*sol.w[t][1:tt]
            end
            R2 = collect(variables.R[j,ttt,tt] for tt in 1:t, ttt in 1:t)
            variables.C[j,1:t,1:t] += R1*(ΣB*R2)
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
        if (n = size(variables.R, 3)) > size(variables.C, 3)
            resize!(variables.C, n)
        end

        #Update the correlation functions!
        t = length(sol.w)
        for j in 1:structure.num_species
            for j2 in 1:structure.num_species
                variables.C[j,j2,:,:] .= 0.
                for j_sum1 in 1:structure.num_species
                    for j_sum2 in 1:structure.num_species

                        R1 = collect(variables.R[j,j_sum1,tt,ttt] for tt in 1:t, ttt in 1:t)                            
                        ΣB = collect(variables.Σ_B[j_sum1,j_sum2,tt,ttt] for tt in 1:t, ttt in 1:t)
                        for tt in 1:t
                            # ΣB[tt,1:tt] .*= sol.w[tt][1:tt].^2
                            ΣB[tt,1:tt] .*= sol.w[tt][1:tt].*sol.w[t][1:tt] #TODO: To check!
                        end
                        R2 = collect(variables.R[j2,j_sum2,ttt,tt] for tt in 1:t, ttt in 1:t)                            
                        variables.C[j,j2,1:t,1:t] += R1 * (ΣB * R2)

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
                    #TODO: Should this be the connected correlator? Is this defined here like this?
                end
            end
        end

    elseif variables.response_type == "cross_old"
        @time sol = kbsolve!(
        (x...) -> fv_cross_withC!(structure, variables, x...),
        (x...) -> fd_withC!(structure, variables, x...),
        [variables.C, variables.R],
        (tstart, tmax);
        callback =  sE,
        atol = atol,
        rtol = rtol,
        stop = x -> (println("t: $(x[end])"); flush(stdout); false),
        v0 = [variables.μ],
        f1! = (y...) -> f1!(structure, variables, y...),
        kmax = k_max,dtini = dtini, dtmax = dtmax, qmax = qmax, qmin = qmin, γ = γ,kmax_vie=kmax_vie)

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
        retval += Σ[t1,k].*R[k,t2].*hs[k] #Need to add a shift here? --IMP (PAY ATTENTION!)
        #retval += Σ[t1,k].*R[k-1,t2].*hs[k-1] #This has been shifted! PAY ATTENTION!
    end

    return retval
end

#function integrate2_cross(hs::Vector, t1, t2, Σ::GreenFunction, R::GreenFunction, μ::GreenFunction; tmax=t1)
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

function integrate2_cross_forC(hs::Vector, t1, t2, Σ::GreenFunction, R, μ::GreenFunction; tmax=t1)
"""
To integrate the self-energy corrections to the correlation function
"""

retval = zero(R[:,:,t1,t2])
num_species = length(μ[1])

for j in 1:num_species
    for j2 in 1:num_species
        #for k in t2+1:t1
        for j_sum in 1:num_species
            for k in 1:t1
                retval[j,j2] += Σ[j,j_sum,t1,k].*R[j_sum,j2,k,t2].*hs[k] #Need to add a shift here? --IMP (PAY ATTENTION!)
            end
        end
    end
end

return retval

end

function integrate2_cross_forC2(hs::Vector, t1, t2, Σ::GreenFunction, R, μ::GreenFunction; tmax=t1)
    """
    To integrate the self-energy corrections to the correlation function
    """
    
    retval = zero(R[:,:,t1,t2])
    num_species = length(μ[1])
    
    for j in 1:num_species
        for j2 in 1:num_species
            #for k in t2+1:t1
            for j_sum in 1:num_species
                for k in 1:t2
                    retval[j,j2] += Σ[j,j_sum,t1,k].*R[j2,j_sum,t2,k].*hs[k] #Need to add a shift here? --IMP (PAY ATTENTION!)
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

function fv_cross_withC!(structure, variables, out, times, h1, h2, t, t′)
    """
    Vertical evolution for cross responses
    """
    
    #println(t," ",t′)

    if t == 1
        corr = variables.Σ_R[t,t]
        corr_C = variables.Σ_B[t,t]
    else
        corr = integrate2_cross(h1, t, t′, variables.Σ_R, variables.R, variables.μ)

        #corr_C = integrate2_cross_forC(h1, t, t′, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross_forC(h1, t, t′, variables.Σ_B, permutedims(variables.R,(2,1,4,3)), variables.μ)
        corr_C = integrate2_cross_forC(h1, t, t′, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross_forC2(h2, t, t′, variables.Σ_B, variables.R, variables.μ)


        #corr_C = integrate2_cross(h1, t, 1, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross(h1, t, 1, variables.Σ_B, permutedims(variables.R,(2,1,3,4)), variables.μ)
        #corr_C = integrate2_cross(h1, t, 1, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross(h1, t, 1, variables.Σ_B, variables.R, variables.μ)
    end

    retval    = zero(variables.R[t,t′])
    retval_C  = zero(variables.C[t,t′])

    for j in 1:structure.num_species
        for j2 in 1:structure.num_species
            retval[j,j2]   = -structure.rate_destruction[j].*variables.R[j,j2,t,t′]
            retval_C[j,j2] = -structure.rate_destruction[j].*variables.C[j,j2,t,t′]
        end
    end

    #out[1]  .= retval .+ corr    #Why does this dot create such a big issue???
    #out[1]  = retval   .+ corr
    #out[2]  = retval_C .+ corr_C

    #Note that out[2] is for response and out[1] is for the correlation!
    out[2]  = retval   .+ corr
    out[1]  = retval_C .+ corr_C

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

function fd_withC!(structure, variables, out, times, h1, h2, t, t′)
    """
    Diagonal evolution when correlation functions are also calculated
    """

    #Diaginal evolution of the response!
    out[2] = zero(out[2])

    #temp = integrate2_cross(h1, t, 1, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross(h1, t, 1, variables.Σ_B, permutedims(variables.R,(2,1,3,4)), variables.μ)

    retval_C  = zero(variables.C[t,t′])
    for j in 1:structure.num_species
        for j2 in 1:structure.num_species
            retval_C[j,j2] = -structure.rate_destruction[j].*variables.C[j,j2,t,t′]
        end
    end

    temp1 = integrate2_cross_forC(h1, t, t′, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross_forC(h1, t, t′, variables.Σ_B, permutedims(variables.R,(2,1,4,3)), variables.μ)
    
    #temp1 = integrate2_cross_forC(h1, t, t′, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross_forC2(h2, t, t′, variables.Σ_B, variables.R, variables.μ)
    #temp2 = integrate2_cross_forC(h1, t′, t, variables.Σ_R, variables.C, variables.μ) .+ integrate2_cross_forC(h1, t′,t, variables.Σ_B, permutedims(variables.R,(2,1,4,3)), variables.μ)
    
    #out[1] = 0.5*(temp1 .+ temp2) .+ retval_C
    #out[1] = temp1 .+ temp2 .+ 2*retval_C

    out[1] = temp1 .+ retval_C

end


function f1!(structure, variables, out, times, h1, t)
    """
    Evolution for the mean
    """

    if t == 1

        #TODO: Handle this function properly, especially for \Sigma_B field!
        self_energy_mak!(structure, variables, times, h1, h1 , t, t) #This is unnecessary updating the \Sigma_B value here!
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
    #TODO: Replace the correction by MAK because the time evolution becomes problematic otherwise!
    #Call the self-energy function here or in fv directly! 

    out[1]  = retval .+ corr

end