function mgs_update(Q::AbstractArray{<:Number}, v::AbstractArray{<:Number}) 
    n, k = size(Q)
    r = zeros(k+1)
    u = copy(v)
   
    for i = 1:k
        r[i] = dot(Q[:, i], u)     # Note that Q[:, i] mutates as iteration of "i = 1 : k" proceeds. 
        u -= r[i] * Q[:, i]        
    end
    r[k+1] = norm(u)
    u /= r[k+1]
    
    return u, r
end


using LinearAlgebra, Statistics , Random
function fs_new(X::AbstractMatrix{<:Number}, 
    y::AbstractArray{<:Number}; intercept=true, verbose=false, maxsteps=min((size(X,1)-intercept), size(X,2), 2000))

    X = Matrix(X)
    y = vec(y)

    n, p = size(X)

    # standardization
    if intercept == true
        y = y .- mean(y)
        X = X .- mean(X, dims = 1)
    end

    # normalizing every column of X to have Euclidean norm 1
    sx = sqrt.(sum(abs2, X, dims=1))

    X = X ./ sx

    β_path = zeros(p, maxsteps+1)
    β_path[:, 1] = zeros(p)

    # Choose predictor that have biggest absolute correlation with response y
    u = transpose(X) * y
    j_hit = argmax(abs.(u))

    if verbose
        println("The first variable is added")
    end

    # r is current number of active predictors and Act stores index of currently active predictors
    r = 1
    Act = [j_hit]
    Inact = collect(1:p)[Not(j_hit)]

    # X1 is currently active predictor matrix and X2 is currently inactive predictor matrix
    X1 = X[:, j_hit]
    X2 = X[:, Not(j_hit)]
    Q, R = qr(X1)
    Q1 = reshape(Q[:, 1], n , 1) # To make Q1 as a matrix

    # We will keep updating Q1 and R as we select new variables 
    for k = 1 : (maxsteps-1)
        # This yields regression coefficients β given current active predictors
        beta = R \ (transpose(Q1) * y)
        
        β_path[Act, k+1] = beta

        # b is r * p-r matrix so that X1 * b becomes X2 matrix projected onto column space of X1
        b = R \ (transpose(Q1) * X2)
        # orthogonalize inactive predictors with respect to active predictors
        X2_orthed = X2 .- X1 * b
        Z = X2_orthed ./ sqrt.(sum(abs2, X2_orthed, dims=1))    # unit normalizing each column of remaining predictor matrix 

        # Find another variable having largest abs correlaton with response y among remaining predictors (after orthogonalized)
        u = transpose(Z) * y
        j_hit = argmax(abs.(u))

        # updating current active predictor numbers and indices
        r += 1
        Act = vcat(Act, Inact[j_hit])
        Inact = Inact[Not(j_hit)]

        # update X1 and X2
        X1 = hcat(X1, X2[:, j_hit])
        X2 = X2[:, Not(j_hit)]

        # update QR of X1 using modified Gram Schimdt 
        qq , rr = mgs_update(Q1, X1[:,r])

        Q1 = [Q1 qq]
        R = [R ; zeros(1, size(R,2))]
        R = [R rr]

        if verbose
            println(string(r)*"-th variable is added")
        end

    end
    
    # store the last beta coefficients at maxstep 
    beta = R \ (transpose(Q1) * y)
    β_path[Act, end] = beta

    # offsetting the effect of standardization
    β_path ./= vec(sx)

    return β_path
end


function prediction_fs(
    X_train::AbstractMatrix{<:Number}, y_train::AbstractArray{<:Number}, X_new::AbstractMatrix{<:Number};
    intercept = true,  verbose = false,  maxsteps = min((size(X,1)-intercept), size(X,2), 2000) )
    β_path = fs_new(X_train, y_train ; intercept= intercept, verbose = verbose, maxsteps = maxsteps)
    if intercept == true
        prediction = (X_new .- mean(X_train, dims=1)) * β_path .+ mean(y_train) 
    else 
        prediction = X_new * β_path
    end
    
    return prediction , β_path 
end