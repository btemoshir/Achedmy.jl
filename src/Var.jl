"""
    Response

Defined as 
`` G(t,t') = 0 if t' > t ``
"""
struct Response <: KadanoffBaym.AbstractSymmetry end

@inline KadanoffBaym.symmetry(::Type{Response}) = zero

"""
    Correlation

Correlation function with no imposed symmetry since cross species correlations need not be symmetric!
"""
struct Correlation <: KadanoffBaym.AbstractSymmetry end

@inline KadanoffBaym.symmetry(::Type{Correlation}) = nothing


"""
    BlockSymmetrical
Defined as invariant under interchanging first two indices with last two indices:
``G(i,j,t,t') = G(j,i,t',t)``
The cross correlation functions are of this type!
"""
struct BlockSymmetrical <: KadanoffBaym.AbstractSymmetry end

# @inline KadanoffBaym.symmetry(::Type{BlockSymmetrical}) = x -> permutedims(x, (3, 4, 1, 2))
@inline KadanoffBaym.symmetry(::Type{BlockSymmetrical}) = permutedims((3, 4, 1, 2))

Base.@kwdef mutable struct ReactionVariables

    response_type = "cross"
    R = 0 
    μ = 0
    C = 0
    N = 0
    Σ_R = 0
    Σ_μ = 0
    Σ_B = 0
    
end

function ReactionVariables(reaction_system::ReactionStructure,response_type="cross")
    
    """
    Initializes the Reaction Variables
    
    Inputs
    ------
    - reaction_system: class(ReactionStructure)
    - response_type : string
                    "single" or "cross"
    """
    
    
    if response_type == "single"
        
        R = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response)
        R[:,1,1] = ones(reaction_system.num_species)
        
        μ = GreenFunction(zeros(Float64,reaction_system.num_species,1), OnePoint)
        μ[:,1] = reaction_system.initial_values
        
        C = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Symmetrical)
        C[:,1,1] = reaction_system.initial_C #Defines the initial correlations in the system if any
        
        N = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response)
        N[:,1,1] = μ[:,1] + C[:1,1]
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
            N = N,
            Σ_R = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response),
            Σ_B = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Symmetrical),
            Σ_μ = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response))
        
    elseif response_type == "cross"
        
        R = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response)
        R[:,:,1,1] = zeros(reaction_system.num_species,reaction_system.num_species)

        for i in 1:reaction_system.num_species
            R[i,i,1,1] = 1. #Only the diagonal responses take the equal time value of one!
        end
        
        μ = GreenFunction(zeros(Float64,reaction_system.num_species,1), OnePoint)
        μ[:,1] = reaction_system.initial_values
        
        #The cross response functions should not be Symmetric only in time but also in species!
        #TODO: Define a new symmetry type for this!

        # C = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), BlockSymmetrical)
        C = zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1)
        
        #TODO: Do this properly, such that an error is raised!
        if size(reaction_system.initial_C) == size(C[:,:,1,1])
            C[:,:,1,1] = reaction_system.initial_C #Defines the initial correlations in the system if any
        elseif size(reaction_system.initial_C) == size(C[:,1,1,])
            if any(reaction_system.initial_C .!= 0)
                @warn "Initial correlations are non-zero and correlation matrix size mismatch. Using diagonal elements only."
            end
            for i in 1:reaction_system.num_species
                C[i,i,1,1] = reaction_system.initial_C[i]
            end
        else
            @warn "Initial correlation matrix size mismatch. Using all zero initial correlations."
            C[:,:,1,1] = zeros(reaction_system.num_species,reaction_system.num_species)
        end

        N = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response)
        N[:,:,1,1] = C[:,:,1,1]
        for i in range(1,reaction_system.num_species)
            N[i,i,1,1] += μ[i,1]
        end
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
            N = N,
            Σ_R = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response),
            Σ_B = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Symmetrical),
            Σ_μ = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response))
    end

end