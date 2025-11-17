# push!(LOAD_PATH,"../src/") #TODO
using Documenter
using Achedmy

makedocs(
    sitename = "Achedmy.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://btemoshir.github.io/Achedmy.jl",
        assets = String[],
    ),
    modules = [Achedmy],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "tutorial.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    repo = "https://github.com/btemoshir/Achedmy.jl/blob/{commit}{path}#{line}",
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/btemoshir/Achedmy.jl.git",
    devbranch = "main",
)
