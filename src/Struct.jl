"""
    ReactionStructure

Stores the stoichiometry and rate information for a chemical reaction network.

# Fields

- `num_species::Int`: Number of chemical species in the network
- `num_interactions::Int`: Number of interaction reactions (non-baseline reactions or reactions other than simple creation/destruction)
- `num_reactions::Int`: Total number of reactions (creation + destruction + interactions)
- `rate_creation::Vector{Float64}`: Rate constants for spontaneous creation reactions
- `rate_destruction::Vector{Float64}`: Rate constants for destruction reactions
- `rate_interaction::Vector{Float64}`: Rate constants for interaction reactions
- `stochiometry_prod::Matrix{Int}`: Stoichiometry matrix for products for each `interaction` reaction
- `stochiometry_react::Matrix{Int}`: Stoichiometry matrix for reactants for each `interaction` reaction
- `initial_values::Vector{Float64}`: Initial molecular counts for each species
- `n_list`: List of reaction vectors for generating combinatorial structures
- `n_list_union`: Union of all reaction vectors
- `m_list`: List of auxiliary reaction vectors
- `m_list_union`: Union of auxiliary reaction vectors
- `initial_C::Union{Vector{Float64}, Matrix{Float64}}`: Initial correlations C(0,0) - can be a vector for single-species or matrix for inter-species correlations

# Constructor

    ReactionStructure(reaction_system::ReactionSystem; 
                     external_initialization=false,
                     initial_correlations=false)

Create a `ReactionStructure` from a Catalyst `ReactionSystem`.

# Arguments

- `reaction_system::ReactionSystem`: Catalyst reaction network
- `external_initialization::Union{Dict,Bool}=false`: External rates/values not in Catalyst define `reaction_system`
- `initial_correlations::Union{Vector,Matrix,Bool}=false`: Initial correlation values C(0,0)
  - `Vector`: For single-species correlations (size: num_species)
  - `Matrix`: For cross-species correlations (size: num_species × num_species)
  - `false`: Zero initial correlations (default)

Reactions are categorized as:
1. **Creation**: ∅ → A (rate: k₁)
2. **Destruction**: A → ∅ (rate: k₂)
3. **Interactions**: A + B → C (rate: k₃)

# Example

```julia
using Catalyst, Achedmy

# Define gene regulation network
gene_network = @reaction_network begin
    @species G(t)=0.01 P(t)=10.0
    @parameters k_on=0.1 k_off=1.0 k_p=10.0 k_d=1.0
    (k_on, k_off), 0 <--> G
    k_p, G --> G + P
    k_d, P --> 0
end

# Create structure
structure = ReactionStructure(gene_network)
println("Number of species: ", structure.num_species)
println("Initial protein count: ", structure.initial_values[2])
println("Creation rates: ", structure.rate_creation)
```

# Notes

- Non-zero initial correlations are currently not supported but planned for future versions
- Species order is determined by the order in the `ReactionSystem`
- External initialization is useful for parameter sweeps

# See Also

- [`ReactionVariables`](@ref): Storage for dynamical variables
- [`solve_dynamics!`](@ref): Solve the dynamics
"""
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
        @warn "Non-zero initial correlations are currently not supported but will be added in future versions. Please open a feature request or Github issue if you need this functionality."
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

    if num_int == 1
            n_list_union = n_list[1]
            m_list_union = m_list[1]
    elseif num_int > 1
        n_list_union = union(collect.(n_list[j] for j in 1:num_int)...)
        m_list_union = union(collect.(m_list[j] for j in 1:num_int)...)
    else
        n_list_union = []
        m_list_union = []
    end
    
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