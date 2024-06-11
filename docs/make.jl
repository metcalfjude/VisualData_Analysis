using FractionalSaccades
using Documenter

DocMeta.setdocmeta!(FractionalSaccades, :DocTestSetup, :(using FractionalSaccades); recursive=true)

makedocs(;
    modules=[FractionalSaccades],
    authors="brendanjohnharris <bhar9988@uni.sydney.edu.au> and contributors",
    sitename="FractionalSaccades.jl",
    format=Documenter.HTML(;
        canonical="https://brendanjohnharris.github.io/FractionalSaccades.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/brendanjohnharris/FractionalSaccades.jl",
    devbranch="main",
)
