##
using Statistics, StatsBase, Random, LinearAlgebra, InvertedIndices , SparseArrays

# soft-thresholding operator
function softthres(x::Number, thres::Number)
    sign(x) * max( abs(x)-thres , 0 )  
end

##
# Lasso with pathwise coordinate descent for active set
function lasso(
    X::AbstractMatrix{<:Number}, y::AbstractArray{<:Number}; intercept = true ,
    nlambda = 50, nrelax = 1, tol = 1e-7, numerical_zero = 1e-4, CDmaxiter = 1e+4, KKTiter = 1e+4, verbose = false)

    X = Matrix(X)
    y = vec(y)

    n = size(X, 1)
    p = size(X, 2)

    # store predictor standard deviation before taking standardizatoin of matrix
    sd = std(X, dims=1)

    # standardization
    if intercept == true
    y = y .- mean(y)
    X = X .- mean(X, dims = 1)
    end

    X = X ./ std(X, dims = 1)


    # threshold for the convergence of coordinate descent 
    # same with glmnet default setting
    thres = sum(abs2, y) * tol
    
    
    # create gamma grid
    if nrelax > 1
        g = -1 / (nrelax - 1)
        γgrid = 1:g:0
    else
        γgrid = 1
    end
    
    # create lambda grid
    # same with glmnet default setting
    if n > p
        minλdivide = 10000
    else
        minλdivide = 100
    end

    λmax = norm(X'y, Inf) / n
    λgrid = exp.(range(log(λmax), log(λmax / minλdivide), length = nlambda))
    
    # create β_path for non-relax lasso
    β_path = spzeros(p, length(λgrid))  

    # Will be used to stop if coordinate descent does not converge
    stop = true

    # Will be used to return current beta if KKT does not satisfied
    notKKT = true
    
    β_current = spzeros(p)
    β_new = spzeros(p)
    
    Add1 = Vector{Int64}()  # will be used for updating strong set S
    Add2 = Vector{Int64}()  # will be used for updating strong set S
    add1 = Vector{Int64}()  # will be used for updating active set A
    add2 = Vector{Int64}()  # will be used for updating active set A

    z = Vector{Float64}(undef, n) # will be used for in-place multiplication of the form Xβ
    w = Vector{Float64}(undef, p) # will be used for in-place multiplication of the form (y-Xβ)'X

    
    # main loop

    for j in 1:length(λgrid)

        λ = λgrid[j]

        if verbose && j % 10 == 0
            println("Solving the Lasso for " * string(j) * "-th λ") 
        end

        ## exponentially enlarge the numerical zero criterion
        if iseven(j)
            numerical_zero *= 1.11
        end

        # trivial case
        if j == 1
            β_path[1:end, j, :] .= 0 
            continue
        end

        # z becomes y - Xβ
        mul!(z, X, -β_current)
        z .+= y
        # w becomes X'(y - Xβ)
        mul!(w, transpose(X), z)

        A = findall(!iszero, β_current) # active set
        S = findall(iszero, abs.(w) .<  2 * n * (λ - λgrid[j-1]) ) # strong set

        
        # do coordinate descent and check KKT conditions
        for iter = 1:KKTiter
            # coordinate descent for active set A
            for i = 1: CDmaxiter
                CDconv = true
                for k in A
                    
                    # z becomes y - X_{-k}β_{-k}
                    mul!(z, X[:, Not(k)], -β_new[Not(k)])
                    z .+= y
                    β_new[k] = softthres(dot(X[:, k] , z), n*λ) / sum(abs2, X[:, k])
                    
                    mul!(z, X, -β_current)
                    z .+= y
                    obj_current = sum(abs2, z) / (2*n) + λ * norm(β_current, 1)
                    
                    mul!(z, X, -β_new)
                    z .+= y
                    obj_new = sum(abs2, z) / (2*n) + λ * norm(β_new, 1)
                    
                    # Coordinate descent convergence condition
                    # Each inner coordinate-descent loop continues until the maximum change in the objective after any coefficient update is less than threshold value
                    if abs(obj_current - obj_new) > thres
                        CDconv = false
                    end
                end

                # stop condition
                if CDconv
                    stop = false
                    break
                end

                β_current = copy(β_new)  # To avoid reference problem β_current vs β_new, use copy function   
            end

            # stop condition : if coordinate descent does not converge then store the current beta
            if stop
                println("Too many iterations for finding optimizer through coordinate descent")
                β_path[:, j] = β_new
                break
            end

            # Check whether each component in strong set S satisfies KKT conditions
            Sz = intersect(S, findall(iszero , β_new))
            Snz = setdiff(S, Sz)
            
            w1 = Vector{Float64}(undef, length(Snz))
            w2 = Vector{Float64}(undef, length(Sz))

            # z becomes y - Xβ
            mul!(z, X, -β_new)
            z .+= y
            # w1 becomes X[Snz]'(y - Xβ)
            mul!(w1, transpose(X[: , Snz]), z) 
            # w2 becomes X[Sz]'(y - Xβ)
            mul!(w2, transpose(X[: , Sz]), z) 
            
            Add1 = Snz[ abs.(abs.( w1 ) .- n*λ ).> numerical_zero]
            Add2 = Sz[ abs.( w2 ) .> n*λ  ]

            # If KKT is not all satisfied, then add offending coef to A and repeat iteration again.
            if !isempty(Add1) || !isempty(Add2)
                union!(A, Add1, Add2)
                if verbose && iter % 10 == 0
                    println("KKT condition " * string(j)*" -th lambda " *string(iter) * "-th iteration")
                    println(norm(abs.( w1 ) .- n*λ , Inf)) 
                    println("numerical zero value is : " * string(numerical_zero))
                    println("Length of Add2 : " * string(length(Add2)) )
                end
                continue
            end
            
            # Check KKT condition for all variables
            # KKT condition is different for members of the active set and variables not in the active set
            A = findall(!iszero, β_new)
            NotA = setdiff(1:p, A)

            v1 = Vector{Float64}(undef, length(A))
            v2 = Vector{Float64}(undef, length(NotA))

            # z becomes y - Xβ
            mul!(z, X, -β_new)
            z .+= y
            # v1 becomes X[A]'(y - Xβ)
            mul!(v1,  transpose(X[: , A]), z) 
            # v2 becomes X[NotA]'(y - Xβ)
            mul!(v2,  transpose(X[: , NotA]), z) 

            add1 = A[ abs.(abs.( v1 ) .- n*λ ) .> numerical_zero ]
            add2 = NotA[ abs.( v2 ) .> n*λ  ]
            
            # If KKT is not all satisfied, then add offending coef to A , recompute the strong set S 
            # and repeat iteration again 
            if !isempty(add1) || !isempty(add2)
                union!(A, add2)

                if verbose
                    println("Strong set update")
                    println("Length of add1 : " * string(length(add1)) )
                    println("Length of add2 : " * string(length(add1)) )
                end

                # z becomes y - Xβ
                mul!(z, X, -β_new)
                z .+= y
                # w becomes X'(y - Xβ)
                mul!(w, transpose(X), z)
                S = findall(iszero, abs.( w ) .<  2 * n * (λ - λgrid[j-1]) )
                continue

            # Else if all KKT conditions are satisfied then the store β in solution path    
            else
                notKKT = false
                β_path[:, j] = β_new 
                break
            end
        end

        # store current beta if KKT does not satisfied
        if notKKT
            println("Too many iterations to satisfy KKT conditions")
            β_path[:, j] = β_new
        end
        
        β_current = copy(β_new)  # warmstart for next loop
    end
   
    
    if nrelax == 1
        # offsetting the effect of standardization
        β_path = Matrix(β_path) ./ vec(sd)
        return β_path, λgrid
    else
        relaxβ_path = zeros(p, nrelax * length(λgrid))
        for j = 1:length(λgrid)
            Aλ = findall(!iszero, β_path[:,j])
            βls = zeros(p)
            βls[Aλ] = X[:, Aλ] \ y
            for i = 1:length(γgrid)
                γ = γgrid[i]
                relaxβ_path[:, nrelax * (j-1) + i] = γ * β_path[:,j] + (1- γ) * βls
            end
        end

        # offsetting the effect of standardization
        relaxβ_path ./= vec(sd)
        return relaxβ_path, λgrid, γgrid
    end
end



function prediction_lasso(
    X_train::AbstractMatrix{<:Number}, y_train::AbstractArray{<:Number}, X_new::AbstractMatrix{<:Number};
    intercept = true, nlambda = 50, nrelax = 1, tol = 1e-7, numerical_zero = 1e-4, CDmaxiter = 1e+4, KKTiter = 1e+4, verbose = false)

    if nrelax == 1
        β_path, λgrid = lasso(X_train, y_train ; intercept = intercept, nlambda = nlambda, nrelax = nrelax, tol = tol, numerical_zero = numerical_zero, CDmaxiter = CDmaxiter, KKTiter = KKTiter,  verbose = verbose)
        
        if intercept 
            prediction = (X_new .- mean(X_train, dims=1)) * β_path .+ mean(y_train) 
        else 
            prediction = X_new * β_path
        end
        
        return prediction, β_path, λgrid

    else
        relaxβ_path, λgrid, γgrid = lasso(X_train, y_train ; intercept = intercept, nlambda = nlambda, nrelax = nrelax, tol = tol, numerical_zero = numerical_zero, CDmaxiter = CDmaxiter, KKTiter = KKTiter,  verbose = verbose)

        if intercept 
            prediction = (X_new .- mean(X_train, dims=1)) * relaxβ_path .+ mean(y_train) 
        else 
            prediction = X_new * relaxβ_path
        end

        return prediction, relaxβ_path, λgrid, γgrid

    end

end