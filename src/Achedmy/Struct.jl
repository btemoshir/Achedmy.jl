
Base.@kwdef struct ReactionStructure{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    
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

    m_list::T7       = []
    m_list_union::T8 = []

    initial_C::T9 = zeros(0)
    
end

function ReactionStructure(reaction_system::ReactionSystem;external_initialization=false,initial_correlations=false)
    
    """
    Converts a reaction structure from ReactionSystem class from Catalyst.jl and creates the ReactionStructure class to be used in Achedmy.
    
    Inputs:
    --------
    - reaction_system: class(ReactionSystem)
    - external_initialization : Dict or False
                       If a Dictionary is supplied, it is useful to provide external rates and starting values not defined in reaction_system
                       The Dictionary should be of type {Any,Any} and the keys (the rates and species) should be defined by using the @unpack macro, see examples.
    - initial_correlations : vector, matrix or False
                       If a vector size(num_species), the initial values of the correlations (C[0,0]) are set to the vector. To be used when using single species response functions.
                       If a matrix of size(num_species,num_species), the initial values of the correlations (including the interspecies correlations) (C[0,0]) are set to the matrix. To be used when using cross species response functions.
                       If False, the initial values of the correlations (C[0,0]) are set to zero.
                       Note that depending on the situation, the initial_correlation value may be different from the true initial number-number correlations, see conversion formulas in the paper.
                       Take care of the order of species. The order is defined in the reaction_system!
                       TODO: Take proper care of the order of species in the initial_correlations, so there is no ambiguity and it is defined according to the species names.

    #TODO: Check the initial correlation definition in Boolean context
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

    if initial_correlations
        initial_C = initial_correlations
    else
        initial_C = zeros(numspecies(reaction_system))
        # In the function Reaction Variables, we check if the dimensions of the initial_C (if initial_C was zero) are correct and resize if necessary
    end

    if external_initialization
        merge!(reaction_system.defaults,external_initialization)
    end

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
    m_list = []
    
    for i in 1:num_int

        x = vec(collect(product([0:Int(max(r_i[j,i],s_i[j,i])) for j in 1:num_species]...)))
        y = vec(collect(product([0:Int(r_i[j,i]) for j in 1:num_species]...)))
        
        #The above creates tuples(Int, Int ... , Int (len = num_species)) which needs to be converted to an array,
        # Otherwise it runs into issues later!
        y′ = []      
        x′ = []

        push!(y′,[[Int(j) for j in k] for k in y])
        push!(x′,[[Int(j) for j in k] for k in x])

        push!(n_list, y′[1])
        push!(m_list, x′[1])
        
    end
    
    n_list_union = union(collect.(n_list[j] for j in 1:num_int)...)
    m_list_union = union(collect.(m_list[j] for j in 1:num_int)...)
    
    return ReactionStructure(num_species = numspecies(reaction_system),
                                           num_interactions  = num_int,
                                           rate_creation     = k1,
                                           rate_destruction  = k2,
                                           rate_interaction  = k3,
                                           stochiometry_prod = s_i,
                                           stochiometry_react= r_i,
                                           initial_values    = initial_values,
                                           n_list            = n_list,
                                           n_list_union      = n_list_union,
                                           m_list            = m_list,
                                           m_list_union      = m_list_union,
                                           initial_C         = initial_C,
                                           )
end