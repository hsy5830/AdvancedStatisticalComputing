using GLMNet, Statistics, StatsBase, Random, LinearAlgebra, InvertedIndices

function lasso_glmnet(X::AbstractMatrix{<:Number}, y::AbstractArray{<:Number}; nlambda = 100, nrelax = 1, intercept = true)

    X = Matrix(X)
    y = vec(y)

    n, p = size(X)

    fit = glmnet(X, y, nlambda = nlambda, intercept = intercept)
    β_glm = fit.betas
    β_path = zeros(p, nrelax * size(β_glm, 2))
    λgrid = fit.lambda

    # store predictor standard deviation before taking standardizatoin of matrix
    sd = std(X, dims=1)

    # standardization
    if intercept == true
        y = y .- mean(y)
        X = X .- mean(X, dims = 1)
    end

    X = X ./ std(X, dims = 1)

    if nrelax > 1
        g = -1 / (nrelax - 1)
        γgrid = 1:g:0
        
        for i in 1:size(β_glm, 2)
            β_ls = zeros(p)

            β = β_glm[:, i]
            A = findall(!iszero, β)

            if length(A) != 0
                XA = X[:, A]
                β_ls[A] = XA \ y
                # offsetting the effect of standardization
                β_ls ./= vec(sd)
                for j in 1:length(γgrid)
                    γ = γgrid[j]
                    β_path[:, nrelax * (i - 1) + j] = γ * β + (1 - γ) * β_ls
                end
            end
        end
        
        return β_path, λgrid
    else
        return β_glm, λgrid
    end
end


function prediction_glmnet(
    X_train::AbstractMatrix{<:Number}, y_train::AbstractArray{<:Number}, X_new::AbstractMatrix{<:Number};
    nlambda = 100, nrelax = 1, intercept = true)

    if nrelax == 1
        β_path, λgrid = lasso_glmnet(X_train, y_train; nlambda = nlambda, nrelax = nrelax, intercept = intercept)
        prediction = (X_new .- mean(X_train, dims=1)) * β_path .+ mean(y_train)

        return prediction, β_path, λgrid
    else
        β_path, λgrid  = lasso_glmnet(X_train, y_train; nlambda = nlambda, nrelax = nrelax, intercept = intercept)
        prediction = (X_new .- mean(X_train, dims=1)) * β_path .+ mean(y_train)

        return prediction, β_path, λgrid
    end
end

## prediction function used for plotting effective degrees of freedom
function prediction_glmnet_df(
    X_train::AbstractMatrix{<:Number}, y_train::AbstractArray{<:Number}, X_new::AbstractMatrix{<:Number};
    nlambda = 100, nrelax = 1,  intercept = true)

    if nrelax == 1
        β_path = lasso_glmnet(X_train, y_train; nlambda = 100, nrelax = nrelax , intercept = true)[1]
        p, len = size(β_path)
        β_path = [β_path  zeros(p, 100 - len)]
        prediction = (X_new .- mean(X_train, dims = 1)) * β_path .+ mean(y_train)

        return prediction, β_path
    else
        β_path = lasso_glmnet(X_train, y_train; nlambda = 100, nrelax = nrelax , intercept = true)[1]
        p, len = size(β_path)
        β_path = [β_path  zeros(p, nrelax * 100 - len)]
        prediction = (X_new .- mean(X_train, dims = 1)) * β_path .+ mean(y_train) 
        
        return prediction, β_path
    end
end