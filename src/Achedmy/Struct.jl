
Base.@kwdef struct ReactionStructure{T1,T2,T3,T4,T5,T6}
    
    num_species::T1      = 0
    num_interactions::T1 = 0
    num_reactions::T1    = 2*num_species + num_interactions
    
    rate_creation::T2    = zeros(0)
    rate_destruction::T2 = zeros(0)
    rate_interaction::T4 = zeros(0) #Should this data type be T3?
    
    stochiometry_prod::T3  = zeros(0)
    stochiometry_react::T3 = zeros(0)
    
    initial_values::T4     = zeros(0)
    
    n_list::T5       = []
    n_list_union::T6 = []
    
end

function ReactionStructure(reaction_system::ReactionSystem,external_initialization=false)
    
    """
    Converts a reaction structure from ReactionSystem class from Catalyst.jl and creates the ReactionStructure class more useful for us.
    
    Inputs:
    --------
    - reaction_system: class(ReactionSystem)
    - external_initialization : True or False
                       If True, it is useful to provide external rates and starting values not defined in reaction_system
    
    TODO: Add support for external initialization
    """
    
    num_int        = 0
    num_species    = numspecies(reaction_system)
    initial_values = zeros(numspecies(reaction_system))
    k1             = zeros(numspecies(reaction_system))
    k2             = zeros(numspecies(reaction_system))
    k3             = zeros(0)
    s_i            = zeros(Int64,(numspecies(reaction_system),0))
    r_i            = zeros(Int64,(numspecies(reaction_system),0))
    temp_index     = 1

    for i in reactions(reaction_system)
        if length(dependents(i,reaction_system)) == 1 && length(i.products) == 0 
            #The destruction reaction
            sp     = speciesmap(reaction_system)[i.substrates[1]]
            k2[sp] = reaction_system.defaults[i.rate]

        elseif length(dependents(i,reaction_system)) == 0 
            #The spontaneous creation reaction
            sp     = speciesmap(reaction_system)[i.products[1]]
            k1[sp] = reaction_system.defaults[i.rate]

        else 
            #The interaction reactions rates
            append!(k3,reaction_system.defaults[i.rate])

            #For stochiometry
            s_i = cat(s_i,prodstoichmat(reaction_system)[:,temp_index],dims=2)
            r_i = cat(r_i,substoichmat(reaction_system)[:,temp_index],dims=2)
            num_int += 1
        end
        temp_index += 1
    end

    #Store the initial values
    for i in species(reaction_system)
        sp = speciesmap(reaction_system)[i]
        initial_values[sp] = reaction_system.defaults[i]

    end
    
    n_list = []
    
    for i in 1:num_int
        
        y = vec(collect(product([0:Int(r_i[j,i]) for j in 1:num_species]...)))
        
        #The above creates tuples(Int, Int ... , Int (len = num_species)) which needs to be converted to an array,
        # Otherwise it runs into issues later!
        y′ = []      
        
        push!(y′,[[Int(j) for j in k] for k in y])
        
        push!(n_list, y′[1])
        
    end
    
    n_list_union = union(collect.(n_list[j] for j in 1:num_int)...)
    
    return ReactionStructure(num_species = numspecies(reaction_system),
                                           num_interactions  = num_int,
                                           rate_creation     = k1,
                                           rate_destruction  = k2,
                                           rate_interaction  = k3,
                                           stochiometry_prod = s_i,
                                           stochiometry_react= r_i,
                                           initial_values    = initial_values,
                                           n_list            = n_list,
                                           n_list_union      = n_list_union)
end