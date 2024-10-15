"""
    Response

Defined as 
`` G(t,t') = 0 if t' > t ``
"""
struct Response <: KadanoffBaym.AbstractSymmetry end

@inline KadanoffBaym.symmetry(::Type{Response}) = zero


Base.@kwdef struct ReactionVariables

    response_type = "cross"
    R = 0 
    μ = 0
    C = 0
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
    
    Add proper support for cross responses!
    
    """
    
    
    if response_type == "single"
        
        R = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response)
        R[:,1,1] = ones(reaction_system.num_species)
        #R[:,:,1,1] = ones(reaction_system.num_species)
        μ = GreenFunction(zeros(Float64,reaction_system.num_species,1), OnePoint)
        μ[:,1] = reaction_system.initial_values
        C = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Symmetrical)
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
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
        C = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Symmetrical)
        
        return ReactionVariables(
            response_type = response_type,
            R = R,
            μ = μ,
            C = C,
            Σ_R = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Response),
            Σ_B = GreenFunction(zeros(Float64,reaction_system.num_species,reaction_system.num_species,1,1), Symmetrical),
            Σ_μ = GreenFunction(zeros(Float64,reaction_system.num_species,1,1), Response))
    end

end