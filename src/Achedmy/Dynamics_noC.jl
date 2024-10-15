include("SelfEnergy.jl")

function solve_dynamics!(structure,variables,atol,rtol,tmax,tstart=0.,selfEnergy="SBR",usingC=True)
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
                What kind of self-energy to use, "SBR-mix", "SBR", "MAK", "alpha2"
                'SBR'     --  SBR corrections to the self-energy. Does not mix different n vectors
                'SBR-mix' --  SBR corrections to the self-energy with different n being mixed (Slow)
                'alpha2'  -- O(\alpha^2) corrections to the self-energy
                'MAK'     -- Mass action kinetics (MAK) or the Mean field or O(\alpha) corrections to the self-energy
    - usingC: True or False
                Decides whether to use Correlation functions in the analysis and update them or not
    """
    
    #Define which self-energy to use:

    if variables.response_type == "single"
        if selfEnergy == "SBR-mix"
            sE = (x...) -> self_energy_SBR_mixed!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            sE = (x...) -> self_energy_SBR!(structure, variables, x...)
        elseif selfEnergy == "MAK"
            sE = (x...) -> self_energy_mak!(structure, variables, x...)
        elseif selfEnergy == "alpha2"
            sE = (x...) -> self_energy_alpha2!(structure, variables, x...)
        end
    elseif variables.response_type == "cross"
        if selfEnergy == "SBR-mix"
            sE = (x...) -> self_energy_SBR_mixed_cross!(structure, variables, x...)
        elseif selfEnergy == "SBR"
            sE = (x...) -> self_energy_SBR_cross!(structure, variables, x...)
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
        f1! = (x...) -> f1!(structure, variables, x...))
        
        #Can calculate C here separately (uses \hat B) for the calculation!
        if (n = size(variables.R, 2)) > size(variables.C, 2)
            resize!(variables.C, n)
        end
        
        for i in 1:structure.num_species
            temp = collect( variables.R[i,tt,ttt].*sol.w[length(sol.w)][ttt] for tt in 1:length(sol.w), ttt in 1:length(sol.w) )
            
            variables.C[i,:,:] = temp*variables.Σ_B[i,1:length(sol.w),1:length(sol.w)]*transpose(temp)
        end
        
    elseif variables.response_type == "cross"
        @time sol = kbsolve!(
        (x...) -> fv_cross!(structure, variables, x...),
        (x...) -> fd_cross!(structure, variables, x...),
        [variables.R],
        (tstart, tmax);
        callback =  sE,
        atol = atol,
        rtol = rtol,
        stop = x -> (println("t: $(x[end])"); flush(stdout); false),
        v0 = [variables.μ],
        f1! = (x...) -> f1!(structure, variables, x...))      

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

    #out[1]  .= retval .+ corr    #Why does this dot create such a big issue???
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

        self_energy_mak!(structure, variables, times, h1, h1 , t, t)
        corr = variables.Σ_μ[t,t]

    else
        corr = integrate1(h1, t, variables.Σ_μ, variables.μ)
    end

    retval   = zero(variables.μ[t])

    for j in 1:structure.num_species
        retval[j] = structure.rate_creation[j] - structure.rate_destruction[j]*variables.μ[j,t]
    end

    #out[1]  .= retval .+ corr #Why does this dot create such a big issue???
    out[1]  = retval .+ corr

end