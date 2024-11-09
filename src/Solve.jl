function solve(problem::MySimpleCobbDouglasChoiceProblem)::Dict{String,Any}

    # initialize -
    results = Dict{String,Any}()
    α = problem.α;
    c = problem.c;
    bounds = problem.bounds;
    I = problem.I;
    xₒ = problem.initial

    # how many variables do we have?
    d = length(α);

    # Setup the problem -
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO, max_iter=500))
    @variable(model, bounds[i,1] <= x[i=1:d] <= bounds[i,2], start=xₒ[i]) # we have d variables
    
    # set objective function -   
    @NLobjective(model, Max, (x[1]^α[1])*(x[2]^α[2]));
    @constraints(model, 
        begin
            # my budget constraint
            transpose(c)*x <= I
        end
    );

    # run the optimization -
    optimize!(model)

    # populate -
    x_opt = value.(x);
    results["argmax"] = x_opt
    results["budget"] = transpose(c)*x_opt; 
    results["objective_value"] = objective_value(model);

    # return -
    return results
end

function solve(model::MyValueIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy
    
    # data -
    k_max = model.k_max;

    # initialize
    U = [0.0 for _ ∈ problem.𝒮];

    # main loop -
    for _ ∈ 1:k_max
        U = [backup(problem, U, s) for s ∈ problem.𝒮];
    end

    return MyValueFunctionPolicy(problem, U);
end