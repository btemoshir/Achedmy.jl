include("BlockOp.jl")
include("Cmn.jl")

function self_energy_mak_noC!(structure, variables, times, h1, h2 , t, t′)
    """
    Mass action kinetics (MAK) or the Mean field or O(α) corrections to the self-energy
    """

    # Resize self-energies when Green functions are resized
    if variables.response_type == "single"
        if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
            #resize!(variables.C, n)
        end
    elseif variables.response_type == "cross"
        if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
        end
        
        #Resize C as we resize R, but C is one size larger than R!
        #if (m = size(variables.R, 3)+2) > (size(variables.C, 3))
        #    resize!(variables.C, m+1)
        #end
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
                    #No deivision by step size at initial time because its zero
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)
                else            
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)./h1[t]
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)./h1[t]
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)./h1[t]
                end
                
                #REMEMBER TO SYMMETRIZE THE B? IS it needed or done explicitly by the symmetry of the response?
    
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

function self_energy_mak!(structure, variables, times, h1, h2 , t, t′)
    """
    Mass action kinetics (MAK) or the Mean field or O(α) corrections to the self-energy
    """

    # Resize self-energies when Green functions are resized
    if variables.response_type == "single"
        if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
            #resize!(variables.C, n)
        end
    elseif variables.response_type == "cross"
        if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
            resize!(variables.Σ_R, n)
            resize!(variables.Σ_μ, n)
            resize!(variables.Σ_B, n)
        end
        
        #Resize C as we resize R, but C is one size larger than R!
        if (m = size(variables.R, 3)+2) > (size(variables.C, 3))
            resize!(variables.C, m+1)
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
                    #No deivision by step size at initial time because its zero
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)
                else            
                    variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)./h1[t]
                    variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)./h1[t]
                    variables.Σ_B[i,t,t] = 2*c_mnFULL(structure,variables,2*temp1,temp0,t)./h1[t]
                end
                
                #REMEMBER TO SYMMETRIZE THE B? IS it needed or done explicitly by the symmetry of the response?
    
            end
        end

        if variables.response_type == "cross"
                        
            variables.Σ_μ[:,t,1:t] .= 0.
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
                        #TODO: Review this whether to divide by h1[t] or not, but its zero at the initial time!
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

            for n ∈ structure.n_list_union
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

function self_energy_SBR_legacy!(structure, variables, times, h1, h2 , t, t′)
    """
    SBR corrections to the self-energy. Does not mix different n     
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
    end
    #print(t,t′,"\n")
            
    if t′ == 1

        temp0  = zeros(Int,structure.num_species)
        id     = diagm(ones(t))
        L      = diagm(-1=>ones(t-1))
        
        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.

        #Temporary variables declared to hold the field values!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t]) 
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]

        end

        for n in structure.n_list_union

            #This loop is executed iff cnn != 0 #and cN0 != 0
            if c_mnFULL_test(structure,variables,n,n) != 0 && c_mnFULL_test(structure,variables,n,temp0) != 0

                cNN = collect(c_mnFULL(structure,variables,n,n,tt) for tt in 1:t)
                
                # IMP -- +1 has been added to the time index of cNN compared to previous version!
                        
                #cNN = collect(c_mnFULL(structure,variables,n,n,tt+1) for tt in 1:t)
                Γ   = collect(prod(factorial.(n) .* variables.R[:,tt,ttt] .^n) for tt in 1:t, ttt in 1:t)                
                cN0 = collect(c_mnFULL(structure,variables,n,temp0,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)

                #The following creates the \Chi matrix (with the shift), but also multiples the columns by the time step size
                χ   = collect(cNN[ttt].*Γ[tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*L
                        
                #Where does L multiply?? REDO -- CHECK!! IMP
                #χ   = L*collect(cNN[ttt].*Γ[tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)
                        
                Ξ   = tril(id .- χ)      #Make the matrix ecplicitly lower triangular!
                LAPACK.trtri!('L','U',Ξ) #LAPAC functions to invert the triangular matrix here!
                ΞcN0 = Ξ*cN0

                for i in 1:structure.num_species

                    temp1    = zeros(Int,structure.num_species)
                    temp1[i] = 1

                    if n ∉ [temp0,temp1] && c_mnFULL_test(structure,variables,n,temp1) != 0 && c_mnFULL_test(structure,variables,temp1,n) != 0
                        cN1  = collect(c_mnFULL(structure,variables,n,temp1,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)
                        ΞcN1 = Ξ*cN1

                        Σ_R_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1)[t,1:t]
                        #print("R ",(c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1)[t,1:t],"\n")
                    end

                    if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0)

                        Σ_μ_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN0)[t,1:t]
                        #print("mu ",(c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN0)[t,1:t],"\n")

                    end
                end
            end
        end
        #print(Σ_R_temp[:,1:t],"\n")
        #print()
                
                
        variables.Σ_R[:,t,1:t] .= Σ_R_temp[:,1:t] 
        variables.Σ_μ[:,t,1:t] .= Σ_μ_temp[:,1:t] 

    end
end
        
function self_energy_SBR!(structure, variables, times, h1, h2 , t, t′)
    """
    SBR corrections to the self-energy. Does not mix different n     
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end
    #print(t,t′,"\n")
            
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

        for n in structure.n_list_union

            #This loop is executed iff cnn != 0 #and cN0 != 0
            #if c_mnFULL_test(structure,variables,n,n) != 0 #&& c_mnFULL_test(structure,variables,n,temp0) != 0

                cNN = collect(c_mnFULL(structure,variables,n,n,tt) for tt in 1:t)
                
                # IMP -- +1 has been added to the time index of cNN compared to previous version!                        
                #cNN = collect(c_mnFULL(structure,variables,n,n,tt+1) for tt in 1:t)
                            
                Γ   = collect(prod(factorial.(n) .* variables.R[:,tt,ttt] .^n) for tt in 1:t, ttt in 1:t)                
                cN0 = collect(c_mnFULL(structure,variables,n,temp0,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)

                #The following creates the \Chi matrix (with the shift), but also multiples the columns by the time step size
                χ   = collect(cNN[ttt].*Γ[tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*L
                
                #Where does L multiply?? REDO -- CHECK!! IMP
                #χ   = L*collect(cNN[ttt].*Γ[tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)
                        
                Ξ   = tril(id .- χ)      #Make the matrix ecplicitly lower triangular!
                LAPACK.trtri!('L','U',Ξ) #LAPAC functions to invert the triangular matrix here!
                ΞcN0 = Ξ*cN0

                for i in 1:structure.num_species

                    temp1    = zeros(Int,structure.num_species)
                    temp1[i] = 1                                

                    if n ∉ [temp0,temp1] && c_mnFULL_test(structure,variables,n,temp1) != 0 && c_mnFULL_test(structure,variables,temp1,n) != 0
                        cN1  = collect(c_mnFULL(structure,variables,n,temp1,ttt).*Γ[tt,ttt] for tt in 1:t, ttt in 1:t)
                        ΞcN1 = Ξ*cN1
                        
                        Σ_R_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1)[t,1:t]
                        #print("R ",(c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1)[t,1:t],"\n")
                    end
                        
                    if n ∉ [temp0,temp1]
                                    
                        Σ_B_temp[i,t]   += (2*c_mnFULL(structure,variables,2*temp1,n,t+1).*sum(ΞcN0,dims=2))[t]
                        print((2*c_mnFULL(structure,variables,2*temp1,n,t+1).*sum(ΞcN0,dims=2))[t])
                                    
                        Γ_MN  = collect(prod(factorial.(temp1+n) .* variables.R[:,tt,ttt] .^n) for tt in 1:t, ttt in 1:t) 
                        cN10  = collect(c_mnFULL(structure,variables,n+temp1,temp0,ttt).*Γ_MN[tt,ttt] for tt in 1:t, ttt in 1:t)
                        ΞcN10 = Ξ*cN10
                                    
                        Σ_B_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN10)[t,1:t]
                        
                        cN1N  = collect(c_mnFULL(structure,variables,n+temp1,n,ttt).*Γ_MN[tt,ttt] for tt in 1:t, ttt in 1:t)
                        ΞcN1N = Ξ*cN1N
                                    
                        #CHECK THIS!!! highest order term in the SBR series!!!
                        Σ_B_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN1N)[t,1:t].*sum(ΞcN0,dims=2)[1:t].*h1[1:t]
                                
                    end

                    if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0)

                        Σ_μ_temp[i,1:t] += (c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN0)[t,1:t]
                        #print("mu ",(c_mnFULL(structure,variables,temp1,n,t+1).*ΞcN0)[t,1:t],"\n")

                    end
                end
            #end
        end
                
        variables.Σ_R[:,t,1:t] = deepcopy(Σ_R_temp[:,1:t]) 
        variables.Σ_μ[:,t,1:t] = deepcopy(Σ_μ_temp[:,1:t]) 
        variables.Σ_B[:,t,1:t] = deepcopy(Σ_B_temp[:,1:t])

    end
end



function self_energy_SBR_mixed!(structure, variables, times, h1, h2 , t, t′)    
    """
    generalized SBR (gSBR) corrections to the self-energy with different n being mixed (Slow -- uses block inversion instead of LAPAC inversion!)  

    TODO: ADD hat(R2) or hat(B2) properly, it is currently lacking! -- TODO

    """

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

        for n in structure.n_list_union
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

        for i in 1:structure.num_species
            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1
            c1N      = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_μ)

            Σ_μ_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]
        end


        #Now we do the calculation for \Sigma_R species wise!

        for i in 1:structure.num_species

            temp1       = zeros(Int,structure.num_species)
            temp1[i]    = 1
            n_listNEW_R = []

            for n in structure.n_list_union
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

function self_energy_SBR_mixed_cross_noC!(structure, variables, times, h1, h2 , t, t′)    
    """
    generalized SBR (gSBR) corrections to the self-energy with different n being mixed (Slow -- uses block inversion instead of LAPAC inversion!)    
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
        
        for n in structure.m_list_union #TODO: This has been changed!
        #for n in structure.n_list_union
            if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0) && sum(n) < 3
                push!(n_listNEW_R,n)
            end
        end

        if length(n_listNEW_R) > 0 

            # cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t)
            # Needs to be diagonal in the tt domain!

            cMN = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)

            Γ   = collect(==(sum(n′),sum(n′′)).*response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
            
            # χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,n′′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_R), n′′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)

            χ = block_mat_mul(Γ, cMN .* reshape(h1, 1, 1, 1, t))
            
            # ------- OLD ---------------

            #cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t)
            
            #Γ   = collect(prod(factorial.(n′) .* variables.R[:,tt,ttt] .^n′) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t) 
            
            #χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_R), n′′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)
            
            #------- OLD END ------------

            Ξ  = block_tri_lower_inverse(block_identity(length(n_listNEW_R),t) .- block_lower_shift(χ))
            #Ξ   = block_tri_lower_inverse_old(block_identity(length(n_listNEW_R),t).-block_lower_shift(χ))

            # cN0 = collect(sum(c_mnFULL(structure,variables,n_listNEW_R[n′′],temp0,ttt).*Γ[n′,n′′,tt,ttt] for n′′ in 1:length(n_listNEW_R)) for n′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)
            cN0 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp0,ttt) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t)  
            cN0 = block_mat_mix_mul(Γ,cN0)

            Ξ_μ = block_mat_mix_mul(Ξ,cN0)
            Ξ_B = sum(block_mat_mix_mul(Ξ,cN0) .*reshape(h1, 1, 1, t) ,dims=3)[:,t]


            #Ξ_μ = sum(block_mat_mix_mul(Ξ,cN0),dims=3)[:,t]
            #Changed -- Older version (with error)

            for i in 1:structure.num_species
    
                temp1       = zeros(Int,structure.num_species)
                temp1[i]    = 1
    
                c1N = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_R)
                
                Σ_μ_temp[i,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ_μ)[t,1:t]

                #Σ_μ_temp[i,t] += sum(c1N[n].*Ξ_μ[n] for n in 1:length(n_listNEW_R))
                #Changed -- Older version (with error)

                for j in 1:structure.num_species

                    temp2       = zeros(Int,structure.num_species)
                    temp2[j]    = 1
                
                    # cN1 = collect(sum(c_mnFULL(structure,variables,n_listNEW_R[n′′],temp2,ttt).*Γ[n′,n′′,tt,ttt] for n′′ in 1:length(n_listNEW_R)) for n′ in 1:length(n_listNEW_R),  tt in 1:t, ttt in 1:t)
                                        
                    cN1 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp2,ttt) for n′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)
                    cN1 = block_mat_mix_mul(Γ,cN1)

                    Ξ2  = block_mat_mix_mul(Ξ,cN1)
                    
                    # for ind in 1:length(n_listNEW_R)
                    #     Σ_R_temp[i,j,1:t] .+= (c1N[ind] .* Ξ2[ind,t,1:t])
                    # end

                    Σ_R_temp[i,j,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]

                    c2N = prod(factorial.(temp1.+temp2)).*collect(c_mnFULL(structure, variables, temp1 .+ temp2, n′, t+1) for n′ in n_listNEW_R)

                    #Σ_B_temp[i,j,t] += sum(c2N[n].*Ξ_μ[n] for n in 1:length(n_listNEW_R)) #This is being evaluated to zero! This is an issue!
                    
                    Σ_B_temp[i,j,t] += dot(c2N,Ξ_B)./h1[t]

                    #Σ_B_temp[i,j,t] += sum(block_vec_mat_mul_single_sp(c2N,Ξ_μ)[t,1:t])
                    #Σ_B_temp[i,j,t] += sum(block_vec_mat_mul_single_sp(c2N,Ξ_μ)[t-1,1:t-1])
                    #Σ_B_temp[i,j,t] += sum(block_vec_mat_mul_single_sp(c2N,Ξ_μ)[1:t,t])
                end
                        
            end

        end

        variables.Σ_R[:,:,t,1:t] .= Σ_R_temp[:,:,1:t] 
        variables.Σ_μ[:,t,1:t]   .= Σ_μ_temp[:,1:t]
        variables.Σ_B[:,:,t,1:t] .= Σ_B_temp[:,:,1:t]

    end

end

function self_energy_SBR_mixed_cross!(structure, variables, times, h1, h2 , t, t′)    
    """
    generalized SBR (gSBR) corrections to the self-energy (using C) with different n being mixed (Slow -- uses block inversion instead of LAPAC inversion!)    
    
    Uses self-consistent C in the calculation!

    #UNSTABLE -- TODO

    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 3)) > size(variables.Σ_R, 3)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
        resize!(variables.Σ_B, n)
    end
    
    #Resize C as we resize R, but C is one size larger than R!
    if (m = size(variables.R, 3)+2) > (size(variables.C, 3))
        resize!(variables.C, m+1)
    end

    if t′ == 1 
    # Only do the self-energy calcultion for the first value of t'

        variables.Σ_R[:,:,t,1:t] .= 0.
        variables.Σ_μ[:,t,1:t]   .= 0.
        variables.Σ_B[:,:,t,1:t] .= 0.

        #Temporary variables to store the value of the self-energy!
        Σ_R_temp = zero(variables.Σ_R[:,:,t,1:t])
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])
        Σ_B_temp = zero(variables.Σ_B[:,:,t,1:t])

        #Update Correlation functions
        for j in 1:structure.num_species
            for j2 in 1:structure.num_species                
                variables.C[j,j2,1:t,1:t] .= 0.
                temp = zero(variables.C[j,j2,1:t,1:t])
                for j_sum1 in 1:structure.num_species
                    for j_sum2 in 1:structure.num_species
                        #variables.C[j,j2,1:t,1:t] += collect(variables.R[j,j_sum1,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*(variables.Σ_B[j_sum1,j_sum2,1:t,1:t]*transpose(collect(variables.R[j2,j_sum2,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)))
                        temp .+= collect(variables.R[j,j_sum1,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*(variables.Σ_B[j_sum1,j_sum2,1:t,1:t]*transpose(collect(variables.R[j2,j_sum2,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)))
                    end
                end
                variables.C[j,j2,1:t,1:t] .= temp
            end
        end

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species
            
            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]
            
            for j_p in 1:structure.num_species
                m_p      = zeros(Int,structure.num_species)
                m_p[j_p] = 1

                for j_q in 1:j_p
                    m_q      = zeros(Int,structure.num_species)
                    m_q[j_q] = 1
                
                    Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,m_p.+m_q,t+1).*variables.C[j_p,j_q,t,t]./h1[t]

                    #print(variables.C[j_p,j_q,t+1,t+1])
                    
                end
            end

            for i_2 in 1:structure.num_species

                temp2    = zeros(Int,structure.num_species)
                temp2[i_2] = 1
                
                Σ_R_temp[i,i_2,t] += c_mnFULL(structure,variables,temp1,temp2,t+1)./h1[t]
                
                Σ_B_temp[i,i_2,t] += prod(factorial.(temp1.+temp2)).*c_mnFULL(structure,variables, temp1.+temp2,temp0,t+1)./h1[t]
                
                for j_p in 1:structure.num_species
                    m_p      = zeros(Int,structure.num_species)
                    m_p[j_p] = 1

                    for j_q in 1:j_p
                        m_q      = zeros(Int,structure.num_species)
                        m_q[j_q] = 1
                        
                        Σ_B_temp[i,i_2,t] += prod(factorial.(temp1.+temp2)).*c_mnFULL(structure,variables,temp1.+temp2,m_p.+m_q,t+1).*variables.C[j_p,j_q,t,t]./h1[t]
                    end
                end
            end
        end

        n_listNEW_R = []

        #for n in structure.n_list_union ## M list or N list here???
        for n in structure.n_list_union
            
            if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0) &&  sum(n) < 3
                push!(n_listNEW_R,n)
            end
            
        end
        
        if length(n_listNEW_R) > 0

            cMN = collect(c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t)

            #NEW
            #Create a c(m,n,t,t') diagonal array
            #cMN = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,n′′,tt) for n′ in n_listNEW_R, n′′ in n_listNEW_R,  tt in 1:t, ttt in 1:t)

            Γ   = collect(==(sum(n′),sum(n′′))*response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)

            #Γ   = collect(response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
            
            #NEW
            #Γ   = collect(==(sum(n′),sum(n′′))*response_combinations(n′,n′′,variables.R[:,:,tt,ttt]) for n′ in n_listNEW_R, n′′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
            
            χ   = collect(cMN[n′,n′′,ttt].*Γ[n′,n′′,tt,ttt].*h1[ttt] for n′ in 1:length(n_listNEW_R), n′′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)

            #NEW
            #χ   = block_mat_mul(Γ,cMN)
            
            Ξ   = block_tri_lower_inverse(block_identity(length(n_listNEW_R),t) .- block_lower_shift(χ))
        
            for i in 1:structure.num_species
    
                temp1       = zeros(Int,structure.num_species)
                temp1[i]    = 1
    
                c1N = collect(c_mnFULL(structure,variables,temp1,n′,t+1) for n′ in n_listNEW_R)

                for j in 1:structure.num_species

                    temp2       = zeros(Int,structure.num_species)
                    temp2[j]    = 1
                
                    cN1 = collect(sum(c_mnFULL(structure,variables,n_listNEW_R[n′′],temp2,ttt).*Γ[n′,n′′,tt,ttt] for n′′ in 1:length(n_listNEW_R)) for n′ in 1:length(n_listNEW_R), tt in 1:t, ttt in 1:t)

                    #NEW
                    #cN1 = collect(==(tt,ttt)*c_mnFULL(structure,variables,n′,temp2,tt) for n′ in n_listNEW_R, tt in 1:t, ttt in 1:t)
                    Ξ2  = block_mat_mix_mul(Ξ,cN1)
                    
                    #NEW
                    #Ξ2  = block_mat_mix_mul(Ξ,block_mat_mix_mul(Γ,cN1))

                    # OR DOT EQUAL?
                    Σ_R_temp[i,j,1:t] .+= block_vec_mat_mul_single_sp(c1N,Ξ2)[t,1:t]

                end
            end
        end

        variables.Σ_R[:,:,t,1:t] .= Σ_R_temp[:,:,1:t] 
        variables.Σ_μ[:,t,1:t]   .= Σ_μ_temp[:,1:t]
        variables.Σ_B[:,:,t,1:t] .= Σ_B_temp[:,:,1:t]
        
        #Update Correlation functions using the identity:
        #for j in 1:structure.num_species
        #    for j2 in 1:structure.num_species                
        #        variables.C[j,j2,1:t+1,1:t+1] .= 0.
        #        for j_sum1 in 1:structure.num_species
        #            for j_sum2 in 1:structure.num_species
        #                variables.C[j,j2,2:t+1,2:t+1] += collect(variables.R[j,j_sum1,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)*(variables.Σ_B[j_sum1,j_sum2,1:t,1:t]*transpose(collect(variables.R[j2,j_sum2,tt,ttt].*h1[ttt] for tt in 1:t, ttt in 1:t)))
        #            end
        #        end
        #    end
        #end
        #print(variables.C[2,2],"\n")
    end
end