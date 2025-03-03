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


#m_list, n_list, m_listFULL, n_listFULL, m_listNEW, n_listNEW = mnList(structure_aaa)



function c_mn_no_mu(structure,int_rxn_index,m,n)
    """
    Creates the c_mn for each individual beta reaction (without the μ factors)
    """
    
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

function c_mn(structure,variables,int_rxn_index,m,n,time)
    """
    Creates the c_mn for each individual beta reaction (including the μ factors)
    """

    if any(n.-structure.stochiometry_react[:,int_rxn_index] .> 0.)
        
        c_mnBeta = 0.

    else
        if time>1
        
        c_mnBeta = structure.rate_interaction[int_rxn_index]*(prod(
        binomial.(structure.stochiometry_prod[:,int_rxn_index],m).*(1 .^(structure.stochiometry_prod[:,int_rxn_index].-m))
        ) .- prod(binomial.(structure.stochiometry_react[:,int_rxn_index],m).*(1 .^(structure.stochiometry_react[:,int_rxn_index].-m) )
        )).*prod(binomial.(structure.stochiometry_react[:,int_rxn_index],n)).*prod(variables.μ[:,time-1] .^(structure.stochiometry_react[:,int_rxn_index].-n))
        
        #Note that this cmn is defined with the time shift t- in \mu! ## PAY ATTENTION!
                
        else
            c_mnBeta = 0.
        end
        
    end
        
    return c_mnBeta
    
end

function c_mnFULL(structure,variables,m,n,time)

    retval = 0.
    
    for k in 1:structure.num_interactions
        retval += c_mn(structure,variables,k,m,n,time)
    end
    
    return retval
    
end

function c_mnFULL_test(structure,variables,m,n)
    
    #This function tests if a partciular cmn_full (no_mu) is zero or not.
    
    retval = 0.
    
    #TODO this thing
    
    for k in 1:structure.num_interactions
        retval += c_mn_no_mu(structure,k,m,n)
    end
    
    return retval
    
end

function create_c_mn_dict(structure,variables,m_list,n_list)
    """
    Creates a dictionary for the c_mn for each individual beta reaction (without the μ factors) where the output is a dictionary and can be looked up by [int(rxn_index),tuple(m),tuple(n)]
    """

    c_mn_dict = Dict()

    for int_rxn_index in range(1,structure.num_interactions)
        
        #Note that for the m_list, n_list above, they should be the m-list, n-list 
        #corresponding to each reaction individually 
        for m in m_list[int_rxn_index]
            for n in n_list[int_rxn_index]
                
                #Can already NOT SAVE many of the c_mn which are zero at this point!
                
                c_mn_dict[(int_rxn_index,m,n)] = c_mn_no_mu(structure,int_rxn_index,m,n)
                #c_mn_dict[(int_rxn_index,tuple(m),tuple(n))] = c_mn_no_mu(structure,int_rxn_index,m,n)
            
            end
        end
    end
    
    return c_mn_dict

end             