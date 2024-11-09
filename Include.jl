# setup paths -
const _ROOT = pwd();
const _PATH_TO_SRC = joinpath(_ROOT, "src")

# load external packages -
using MadNLP
using JuMP
using Plots
using Colors

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"))
include(joinpath(_PATH_TO_SRC, "Factory.jl"))
include(joinpath(_PATH_TO_SRC, "Solve.jl"))
include(joinpath(_PATH_TO_SRC, "Compute.jl"))