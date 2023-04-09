function bs_one_k(
    X::AbstractMatrix{<:Number}, y::AbstractArray{<:Number}, k::Integer;
    opt = Convex.MOI.OptimizerWithAttributes(Gurobi.Optimizer, "TimeLimit" => 60, MOI.Silent() => true), L = eigmax(transpose(X)*X), initial = zeros(size(X,2)),  nruns = 50, polish = true, tol = 1e-4, maxiter = 1e+3, verbose = false)

    n, p = size(X)
    
    if k == 0 
        return zeros(p)
    end

    # warmstart derived by projected gradient method
    β_best = bs_proj_gd(X, y, k; L = L, initial = initial, nruns = nruns, polish = polish, tol = tol, maxiter = maxiter, verbose = verbose)
    z0 = Float64.(β_best .!= 0)

    # used as the upper bound of β̂ inf norm
    M = 2 * norm(β_best, Inf)

    if verbose
        println("(b) . Running Gurobi's mixed integer program solver...")
    end

    if n >= p
        # Optimization problem (2.5) in the paper
        β = Variable(p)
        z = Variable(p, BinVar)

        problem = minimize(0.5 * sumsquares(X * β) - dot(transpose(X) * y, β))
        problem.constraints += abs(β) ≤ M * z  # This implies SOS-1 condition and inf norm boundedness of β̂
        problem.constraints += sum(z) ≤ k

        # warm start is used
        set_value!(β, β_best)
        set_value!(z, z0)
    else
        # Optimization problem (2.6) in the paper
        β = Variable(p)
        z = Variable(p, BinVar)
        ξ = Variable(n)

        problem = minimize(0.5 * sumsquares(ξ) - dot(transpose(X)*y, β) )
        problem.constraints += ξ == X * β
        problem.constraints += abs(β) ≤ M * z
        problem.constraints += sum(z) ≤ k

        sortedX = sort(abs.(X), dims = 2, rev = true)
        ξbound = mapslices(sum, sortedX[:, 1:k], dims = 2)  # The sum of the top k abs vals for each row of X
        problem.constraints += abs(ξ) ≤ M * ξbound  # This constraint is mentioned at (2.13) in the paper

        # warm start is used
        set_value!(β , β_best)
        set_value!(z, z0)
        set_value!(ξ, X * β_best)
    end
    
    solve!(problem, opt)

    if verbose
        println(problem.status)
    end

    if β.value == nothing
        β.value = β_best
    end

    return vec(β.value)
end


function bs_proj_gd(
    X::AbstractMatrix{<:Number}, y::AbstractArray{<:Number}, k::Integer;
    L = eigmax(transpose(X) * X), initial = zeros(size(X, 2)), nruns = 50, polish = true, tol = 1e-4, maxiter = 1e+3, verbose = false)
    
    n, p = size(X)

    β_best = copy(initial)
    criterion = Inf
    β = initial
    β_current = copy(β)
    β_new = zeros(p)

    if verbose 
        println("(a) . Performing projected gradient runs...")
    end

    for r = 1:nruns
        for i = 1:maxiter
            # this update is mentioned at (3.8) of the paper
            β_new = β_current .- vec(transpose(X) * (X * β_current .- y) / L)
            
            indices = sortperm(β_new , by = abs, rev = true)
            β_new[indices[k+1:end]] .= 0  # projection 
            
            # polishing coef on the active set
            if polish
                if k == n
                    β_new[indices[1:k]] .= svd(X[:, indices[1:k]]) \ y
                else
                    β_new[indices[1:k]] .= X[:, indices[1:k]] \ y
                end
            end

            if norm(β_new - β_current) / max(norm(β_new), 1) < tol 
                break
            end

            β_current = β_new
        end

        # measure the squared error of β_new and compare it with the current champion
        crit_candidate = sum(abs2, y - X * β_new)
        if crit_candidate < criterion 
            criterion = crit_candidate
            β_best = β_new
        end

        # rerun project gradient with another initial values
        β = initial + 2 * rand(p) .* max.(abs.(initial), 1)
        β_current = copy(β)
    end

    return β_best
end


function bestsubset(
    X::AbstractMatrix{<:Number}, y::AbstractArray{<:Number}; intercept = true,
    initial = zeros(size(X, 2)), k = 0:min(size(X, 1), size(X, 2), 200), timelimit = 100, nruns = 50, polish = true, verbose = false, tol = 1e-4, maxiter = 1e+3)

    X = Matrix(X)
    y = vec(y)

    n, p = size(X)
    sd = std(X, dims = 1)

    # store predictor standard deviation before taking standardizatoin of matrix
    pred_sd = vec(std(X, dims=1))

    # standardization of X and centering of y
    if intercept
        y = y .- mean(y)
        X = X .- mean(X, dims = 1)
    end
           
    X = X ./ std(X, dims = 1)

    # L will be used for learning rate of projected gradient method
    L = eigmax(transpose(X) * X)
    nk = length(k)
    β0 = initial
    β_path = zeros(p, nk)

    opt = Convex.MOI.OptimizerWithAttributes(Gurobi.Optimizer, "TimeLimit" => timelimit, MOI.Silent() => true)

    for i = 1:nk
        if verbose
            println("Solving best subset selection with k = " * string(k[i]))
        end
        
        β = bs_one_k(X, y, k[i]; opt = opt, L = L, initial = β0,  nruns = nruns, polish = polish, tol = tol, maxiter = maxiter, verbose = verbose)
        β_path[:, i] = β
        β0 = β
    end

    # offseting the effect of standardization
    β_path ./= sd'

    return β_path, k
end


function prediction_bs(
    X_train::AbstractMatrix{<:Number}, y_train::AbstractArray{<:Number}, X_new::AbstractMatrix{<:Number};
    intercept = true, initial = zeros(size(X, 2)), k = 0:min(size(X, 1), size(X, 2), 200), timelimit = 100, nruns = 50, polish = true, verbose = false, tol = 1e-4, maxiter = 1e+3)

    β_path, k = bestsubset(X_train, y_train; intercept = intercept, initial = initial, k = k, timelimit = timelimit, nruns = nruns, polish = polish, verbose = verbose, tol = tol, maxiter = maxiter)
    
    if intercept 
        prediction = (X_new .- mean(X_train, dims=1)) * β_path .+ mean(y_train) 
    else 
        prediction = X_new * β_path
    end

    return prediction, β_path, k
end