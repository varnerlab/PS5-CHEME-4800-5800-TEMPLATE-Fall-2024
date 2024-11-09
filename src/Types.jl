abstract type AbstractUtilityFunctionType end
abstract type AbstractSimpleChoiceProblem end
abstract type AbstractWorldModel end
abstract type AbstractProcessModel end

"""
    mutable struct MySimpleCobbDouglasChoiceProblem

A model for a Cobb-Douglas choice problem. 

### Fields
- `α::Array{Float64,1}`: the vector of parameters for the Cobb-Douglas utility function (preferences)
- `c::Array{Float64,1}`: the vector of unit prices for the goods
- `I::Float64`: the income the consumer has to spend
- `bounds::Array{Float64,2}`: the bounds on the goods [0,U] where U is the upper bound
- `initial::Array{Float64,1}`: the initial guess for the solution
"""
mutable struct MySimpleCobbDouglasChoiceProblem <: AbstractSimpleChoiceProblem

    # data -
    α::Array{Float64,1}         # preferences
    c::Array{Float64,1}         # prices
    I::Float64                  # budget
    bounds::Array{Float64,2}    # bounds
    initial::Array{Float64,1}   # initial guess

    # constructor
    MySimpleCobbDouglasChoiceProblem() = new();
end


"""
    mutable struct MyRectangularGridWorldModel

A model for a rectangular grid world.

### Fields
- `number_of_rows::Int`: the number of rows in the grid
- `number_of_cols::Int`: the number of columns in the grid
- `coordinates::Dict{Int,Tuple{Int,Int}}`: the coordinates of the grid. Key is the state index, value is the coordinate tuple
- `states::Dict{Tuple{Int,Int},Int}`: the states of the grid. Key is the coordinate tuple, value is the state index
- `moves::Dict{Int,Tuple{Int,Int}}`: the moves in the grid. Key is the action index, value is the move tuple
- `rewards::Dict{Int,Float64}`: the rewards in the grid. Key is the state index, value is the reward
"""
mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

    # data -
    number_of_rows::Int
    number_of_cols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Int,Float64}

    # constructor -
    MyRectangularGridWorldModel() = new();
end

"""
    mutable struct MyMDPProblemModel

A model for a Markov Decision Process (MDP).

### Fields
- `𝒮::Array{Int64,1}`: the state space
- `𝒜::Array{Int64,1}`: the action space
- `T::Union{Function, Array{Float64,3}}`: the transition function
- `R::Union{Function, Array{Float64,2}}`: the reward function
- `γ::Float64`: the discount factor
"""
mutable struct MyMDPProblemModel <: AbstractProcessModel

    # data -
    𝒮::Array{Int64,1}
    𝒜::Array{Int64,1}
    T::Union{Function, Array{Float64,3}}
    R::Union{Function, Array{Float64,2}}
    γ::Float64
    
    # constructor -
    MyMDPProblemModel() = new()
end

struct MyValueIterationModel 
    # data -
    k_max::Int64; # max number of iterations
end

struct MyValueFunctionPolicy
    problem::MyMDPProblemModel
    U::Array{Float64,1}
end
