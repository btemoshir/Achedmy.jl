
include("BlockOp.jl")
include("Cmn.jl")

function self_energy_mak!(structure, variables, times, h1, h2 , t, t′)
    """
    Mass action kinetics (MAK) or the Mean field or O(\alpha) corrections to the self-energy
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
    end

    if t′ == 1

        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.
        temp0  = zeros(Int,structure.num_species)

        for i in 1:structure.num_species
            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            if t == 1
                #No deivision by step size at initial time because its zero
                #variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t+1)
                #variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t+1)
                variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)
                variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)
            else            
                #variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
                #variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]
                variables.Σ_R[i,t,t] = c_mnFULL(structure,variables,temp1,temp1,t)./h1[t]
                variables.Σ_μ[i,t,t] = c_mnFULL(structure,variables,temp1,temp0,t)./h1[t]
            end

        end
    end

end

function self_energy_alpha2!(structure, variables, times, h1, h2 , t, t′)
    """
    O(\alpha^2) corrections to the self-energy
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
    end

    if t′ == 1

        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.

        #Temporary variables declared to hold the field values!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t]) 
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]

            for n ∈ structure.n_list_union
                if n ∉ [temp0,temp1]

                    Σ_R_temp[i,1:t] += c_mnFULL(structure,variables,temp1,n,t+1).*collect(
                    c_mnFULL(structure,variables,n,temp1,t′)*prod(factorial.(n).*variables.R[t,t′].^n) for t′ in 1:t)

                end

                if n ∉ push!(collect.(Int.(I[1:structure.num_species,k]) for k in 1:structure.num_species),temp0)

                    Σ_μ_temp[i,1:t] += c_mnFULL(structure,variables,temp1,n,t+1).*collect(
                    c_mnFULL(structure,variables,n,temp0,t′)*prod(factorial.(n).*variables.R[t,t′].^n) for t′ in 1:t)

                end
            end
        end

        variables.Σ_R[:,t,1:t] .= Σ_R_temp[:,1:t] 
        variables.Σ_μ[:,t,1:t] .= Σ_μ_temp[:,1:t] 

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
                
        variables.Σ_R[:,t,1:t] = deepcopy(Σ_R_temp[:,1:t]) 
        variables.Σ_μ[:,t,1:t] = deepcopy(Σ_μ_temp[:,1:t]) 

    end
end



function self_energy_SBR_mixed!(structure, variables, times, h1, h2 , t, t′)    
    """
    SBR corrections to the self-energy with different n being mixed (Slow -- uses block inversion instead of LAPAC inversion!)    
    """

    # Resize self-energies when Green functions are resized    
    if (n = size(variables.R, 2)) > size(variables.Σ_R, 2)
        resize!(variables.Σ_R, n)
        resize!(variables.Σ_μ, n)
    end        

    if t′ == 1 
    # Only do the self-energy calcultion for teh first value of t'

        variables.Σ_μ[:,t,1:t] .= 0.
        variables.Σ_R[:,t,1:t] .= 0.

        #Temporary variables to store the value of the self-energy!
        Σ_R_temp = zero(variables.Σ_R[:,t,1:t])
        Σ_μ_temp = zero(variables.Σ_μ[:,t,1:t])

        temp0    = zeros(Int,structure.num_species)

        for i in 1:structure.num_species

            temp1    = zeros(Int,structure.num_species)
            temp1[i] = 1

            Σ_R_temp[i,t] += c_mnFULL(structure,variables,temp1,temp1,t+1)./h1[t]
            Σ_μ_temp[i,t] += c_mnFULL(structure,variables,temp1,temp0,t+1)./h1[t]

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

    end

end