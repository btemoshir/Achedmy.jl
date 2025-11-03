@testset "ReactionStructure Tests" begin
    
    @testset "Simple Birth-Death Process" begin
        # Define simple birth-death system
        bd_system = @reaction_network begin
            @species X(t) = 10.0
            @parameters k_birth = 1.0 k_death = 0.5
            k_birth, 0 --> X
            k_death, X --> 0
        end
        
        # Create structure
        structure = Achedmy.ReactionStructure(bd_system)
        
        # Test basic properties
        @test structure.num_species == 1
        @test structure.num_reactions == 2
        @test length(structure.rates) == 2
        @test structure.rates[1] == 1.0
        @test structure.rates[2] == 0.5
        
        # Test stoichiometry
        @test size(structure.stoich_mat) == (1, 2)
        @test structure.stoich_mat[1, 1] == 1  # Birth adds 1
        @test structure.stoich_mat[1, 2] == -1  # Death removes 1
        
        # Test initial conditions
        @test structure.initial_values[1] == 10.0
    end
    
    @testset "Gene Regulation System" begin
        gene_system = @reaction_network begin
            @species G(t) = 0.0 P(t) = 10.0
            @parameters k_on = 0.1 k_off = 1.0 k_p = 10.0 k_d = 1.0
            (k_on, k_off), 0 <--> G
            k_p, G --> G + P
            k_d, P --> 0
        end
        
        structure = Achedmy.ReactionStructure(gene_system)
        
        @test structure.num_species == 2
        @test structure.num_reactions == 4
        @test length(structure.initial_values) == 2
        @test structure.initial_values[2] == 10.0
    end
    
    @testset "Enzyme Kinetics System" begin
        enzyme_system = @reaction_network begin
            @species S(t) = 1.0 E(t) = 0.9 C(t) = 0.1 X(t) = 0.1
            @parameters k_f = 1.0 k_b = 0.1 k_d = 1.0 k_2X = 1.0 k_2S = 1.0 k_1S = 1.0
            (k_f, k_b), S + E <--> C
            k_d, C --> E + X
            k_2X, X --> 0
            (k_2S, k_1S), S <--> 0
        end
        
        structure = Achedmy.ReactionStructure(enzyme_system)
        
        @test structure.num_species == 4
        @test all(structure.initial_values .>= 0)
        
        # Test that bimolecular reactions are properly handled
        @test size(structure.stoich_mat, 1) == 4
    end
end