@testset "ReactionVariables Tests" begin
    
    @testset "Variable Initialization - Cross" begin
        gene_system = @reaction_network begin
            @species G(t) = 0.0 P(t) = 10.0
            @parameters k_on = 0.1 k_off = 1.0 k_p = 10.0 k_d = 1.0
            (k_on, k_off), 0 <--> G
            k_p, G --> G + P
            k_d, P --> 0
        end
        
        structure = Achedmy.ReactionStructure(gene_system)
        variables = Achedmy.ReactionVariables(structure, "cross")
        
        # Test that arrays are properly initialized
        @test size(variables.μ) == (2, 1)  # 2 species, 1 time point initially
        @test variables.μ[1, 1] == 0.0
        @test variables.μ[2, 1] == 10.0
        
        # Test that correlation arrays exist for cross
        @test isdefined(variables, :N)
        @test isdefined(variables, :R)
        @test isdefined(variables, :C)
        @test size(variables.C) == (2, 2, 1, 1)  # Cross correlations

        # Check that all arrays are of correct type
        @test eltype(variables.μ) <: Real
        @test all(isfinite.(variables.μ[1]))
    end
    
    @testset "Variable Initialization - Single" begin
        gene_system = @reaction_network begin
            @species G(t) = 0.0 P(t) = 10.0
            @parameters k_on = 0.1 k_off = 1.0 k_p = 10.0 k_d = 1.0
            (k_on, k_off), 0 <--> G
            k_p, G --> G + P
            k_d, P --> 0
        end
        
        structure = Achedmy.ReactionStructure(gene_system)
        variables = Achedmy.ReactionVariables(structure, "single")
        
        @test size(variables.μ) == (2, 1)
        
        # For single response, dimensions should be different
        @test isdefined(variables, :N)
        @test isdefined(variables, :R)
        @test isdefined(variables, :C)

        # Check that all arrays are of correct type
        @test eltype(variables.μ) <: Real
        @test all(isfinite.(variables.μ[1]))

    end
    
end