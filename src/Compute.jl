function Q(p::MyMDPProblemModel, U::Array{Float64,1})::Array{Float64,2}

    # grab stuff from the problem -
    R = p.R;  # reward -
    T = p.T;    
    γ = p.γ;
    𝒮 = p.𝒮;
    𝒜 = p.𝒜

    # initialize -
    Q_array = Array{Float64,2}(undef, length(𝒮), length(𝒜))

    for s ∈ 1:length(𝒮)
        for a ∈ 1:length(𝒜)
            Q_array[s,a] = R[s,a] + γ*sum([T[s,s′,a]*U[s′] for s′ in 𝒮]);
        end
    end

    return Q_array
end

function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    # get the dimension -
    (NR, _) = size(Q_array);

    # initialize some storage -
    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    # return -
    return π_array;
end

function backup(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64)
    return maximum(lookahead(problem, U, s, a) for a ∈ problem.𝒜);
end

function lookahead(p::MyMDPProblemModel, U::Vector{Float64}, s::Int64, a::Int64)

    # grab stuff from the problem -
    R = p.R;  # reward -
    T = p.T;    
    γ = p.γ;
    𝒮 = p.𝒮;
    
    # setup my state array -
    return R[s,a] + γ*sum(T[s,s′,a]*U[i] for (i,s′) in enumerate(𝒮))
end