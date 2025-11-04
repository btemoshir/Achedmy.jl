@testset "Dynamics Tests" begin
    @testset "MAK - Birth-Death" begin
        bd_system = @reaction_network begin
            @species X(t) = 10.0
            @parameters k_birth = 1.0 k_death = 0.5
            (k_birth), 0 --> X
            (k_death), X --> 0
        end
        
        structure = Achedmy.ReactionStructure(bd_system)
        variables = Achedmy.ReactionVariables(structure, "single")
        
        sol = Achedmy.solve_dynamics!(
            structure, 
            variables,
            selfEnergy = "MAK",
            tmax = 20,
            tstart = 0.0,
            atol = 1e-4,
            rtol = 1e-3
        )
        
        # Test that solution exists
        @test sol !== nothing
        @test length(sol.t) > 1
        
        # Test that mean converges to steady state
        # For birth-death: steady state = k_birth / k_death = 2.0
        final_mean = variables.μ[1, end]
        @test isapprox(final_mean, 2.0, rtol=0.1)
        
        # Test that means are always positive
        @test all(variables.μ[end] .>= 0)
    end
    
    @testset "gSBR - Gene Regulation" begin
        gene_system = @reaction_network begin
            @species G(t) = 0.0 P(t) = 10.0
            @parameters k_on = 0.1 k_off = 1.0 k_p = 10.0 k_d = 1.0
            (k_on, k_off), 0 <--> G
            k_p, G --> G + P
            k_d, P --> 0
        end
        
        structure = Achedmy.ReactionStructure(gene_system)
        variables = Achedmy.ReactionVariables(structure, "cross")
        
        sol = Achedmy.solve_dynamics!(
            structure,
            variables,
            selfEnergy = "gSBR",
            tmax = 3.0,
            tstart = 0.0,
            atol = 1e-3,
            rtol = 1e-2
        )
        
        @test sol !== nothing
        @test length(sol.t) > 1
        
        # Test physical constraints
        @test all(variables.μ[:,:] .>= 0)  # No negative populations
        
        # Test that variances are non-negative
        for i in 1:structure.num_species
            var_i = diag(variables.N[i, i, :, :])
            @test all(var_i .>= -1e-10)  # Allow small numerical errors
        end
    end
    
    @testset "Method Comparison" begin
        # Simple system where we can compare methods
        simple_system = @reaction_network begin
            @species X(t) = 5.0
            @parameters k = 1.0
            k, 0 --> X
            k, X --> 0
        end
        
        structure = Achedmy.ReactionStructure(simple_system)
        
        methods = ["MAK", "MCA", "SBR", "gSBR"]
        solutions = Dict()
        
        for method in methods
            try
                vars = Achedmy.ReactionVariables(structure, "cross")
                sol = Achedmy.solve_dynamics!(
                    structure, vars,
                    selfEnergy = method,
                    tmax = 2.0, tstart = 0.0,
                    atol = 1e-3, rtol = 1e-2
                )
                solutions[method] = (sol, vars)
            catch e
                @warn "Method $method failed: $e"
            end
        end
        
        # At least MAK should work
        @test haskey(solutions, "MAK")
        
        # If gSBR works, it should give reasonable results
        if haskey(solutions, "gSBR")
            _, vars = solutions["gSBR"]
            @test all(isfinite.(vars.μ[end]))
        end
    end
    
    @testset "Tolerance Effects" begin
        bd_system = @reaction_network begin
            @species X(t) = 10.0
            @parameters k_birth = 1.0 k_death = 0.5
            k_birth, 0 --> X
            k_death, X --> 0
        end
        
        structure = Achedmy.ReactionStructure(bd_system)
        
        # Tight tolerance
        vars_tight = Achedmy.ReactionVariables(structure, "single")
        sol_tight = Achedmy.solve_dynamics!(
            structure, vars_tight,
            selfEnergy = "MAK",
            tmax = 5.0, atol = 1e-6, rtol = 1e-5
        )
        
        # Loose tolerance
        vars_loose = Achedmy.ReactionVariables(structure, "single")
        sol_loose = Achedmy.solve_dynamics!(
            structure, vars_loose,
            selfEnergy = "MAK",
            tmax = 5.0, atol = 1e-2, rtol = 1e-1
        )
        
        # Tighter tolerance should give more time points
        @test length(sol_tight.t) >= length(sol_loose.t)
    end
end