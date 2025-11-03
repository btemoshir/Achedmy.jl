using Test
using Achedmy
using Catalyst
using LinearAlgebra
using Serialization

@testset "Achedmy.jl Tests" begin
    
    @testset "Module Loading" begin
        @test isdefined(Achedmy, :ReactionStructure)
        @test isdefined(Achedmy, :ReactionVariables)
        @test isdefined(Achedmy, :solve_dynamics!)
    end
    
    include("test_structure.jl")
    include("test_variables.jl")
    include("test_dynamics.jl")
    include("test_selfenergy.jl")
    include("test_examples.jl")
end