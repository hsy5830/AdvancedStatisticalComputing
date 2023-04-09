using LinearAlgebra, InvertedIndices, Statistics

# givens, rowrot, colrot for `update1`
function givens(a, b) # return c, s
    if b == 0.0
        c = 1.0
        s = 1.0
    else
        if (abs(b) > abs(a))
            t = -a / b
            s = 1.0 / sqrt(1 + t * t)
            c = s * t
        else
            t = -b / a
            c = 1.0 / sqrt(1 + t * t)
            s = c * t
        end
    end
    
    return c, s
end

function rowrot(A, i1, i2, m, n, j1, j2, c, s)  # A : array <-- matrix
    t1, t2 = 0, 0
    for j in j1:j2
        t1 = A[i1 + j * m]
        t2 = A[i2 + j * m]
        A[i1 + j * m] = c * t1 - s * t2
        A[i2 + j * m] = s * t1 + c * t2
    end
    
    return A
end

function colrot(A, j1, j2, m, n, i1, i2, c, s)  # A : array (<-- matrix)
    t1, t2 = 0, 0
    for i in i1:i2
        t1 = A[i + j1 * m]
        t2 = A[i + j2 * m]
        A[i + j1 * m] = c * t1 - s * t2
        A[i + j2 * m] = s * t1 + c * t2
    end
    
    return A
end

# update1 for `updateQR`
function update1(Q2, w, mp, kp) # Q2, w 는 모두 vector type
    m = mp # Q2 : m x k matrix
    k = kp
    w0, Q20 = 0, 0
    
    for j in k:-1:2
        c, s = givens(w[j-1], w[j])

        w0 = rowrot(w, j-1, j, k, 1, 0, 0, c, s)
      
        Q20 = colrot(Q2, j-2, j-1, m, k, 1, m, c, s)

    end
    
    return w0, Q20 # vector type
end

# updateQR for `fs`
function updateQR(
    Q1::AbstractMatrix{<:Number}, Q2::AbstractMatrix{<:Number},
    R::AbstractMatrix{<:Number}, col::AbstractArray{<:Number})
    
    # Q1 : m x n
    # Q2 : m x k
    # R  : n x n
    m = size(Q1, 1)
    n = size(Q1, 2)
    k = size(Q2, 2)
    
    a = update1(reshape(Q2, m * k, 1)[:,1], reshape(Q2' * col, k, 1)[:,1], m, k) # tuple

    Q2 = reshape(a[2], m, k)

    w = [Q1' * col; a[1]]  # ; 추가  / 1207 17:17
    
    Q1 = [Q1 Q2[:, 1]]    # Q1 : m x (n+1)
    Q2 = Q2[:, Not(1)]    # Q2 : m x (k-1)
    R = [R; zeros(n)']
    R = [R w[1:n+1]]     # R : (n+1) x (n+1)
    
    return Q1, Q2, R
end

# standardize for `fs`
function standardize(x, y, intercept, normalize)
    n = size(x, 1)
    p = size(x, 2)
    
    if intercept
        bx = mean(x, dims=1)
        by = mean(y)
        x = x .- bx
        y = y .- mean(y)
    else
        bx = zeros(p)
        by = 0
    end
    
    if normalize
        sx = sqrt.(sum(abs2, x, dims=1))
        x = x ./ sx
    else
        sx = ones(p)
    end
    
    return x, y, bx, by, sx
end



########## fs ##########
# X : matrix
# y : vector
function fs(x::AbstractMatrix{<:Number}, 
        y::AbstractArray{<:Number}; intercept = true, normalize = true, verbose = false, maxsteps = min((size(x,1) - intercept), size(x,2), 2000))
    
    # set up data
    n = size(x, 1)
    p = size(x, 2)
    
    # save original x and y
    x0 = copy(x)
    y0 = copy(y)
    
   
    # center and scale
    obj = standardize(x0, y0, intercept, normalize)
    x0 = obj[1]     # centered or scaled x
    y0 = obj[2]     # centered y
    bx = obj[3]    # column mean
    by = obj[4]    # mean y
    sx = obj[5]    # root of column squared sum

 
    #####
    # find the first variable to enter and its sign
    z = x0 ./ sx
    u = z' * y0
    j_hit = findmax(abs.(u))[2]
    sign_hit = sign(u[j_hit])    
    
    if verbose
        println("Variable is added.")
    end
    
    ##### iterate to find the seq of FS estimates
    # things to. eep track of, and return at the end
    buf = min(maxsteps+1, 500)
    action = zeros(buf)
    df = zeros(buf)
    beta = zeros(p, buf) # p beta estimators are stored buf times
    
    # record action, df, solution
    action[1] = j_hit
    df[1] = 0
    beta[:,1] .= 0
    
    # other things to keep track of, but not return
    r = 1                                # size of active set
    A = j_hit                            # Active set (vector)
    I = collect(1:p)[Not(j_hit)]         # Inactive set
    sign_A = sign_hit                      # Active signs
    X1 = x0[:, j_hit]                      # Matrix X[,A]
    X2 = x0[:, Not(j_hit)]                 # Matrix X[,I]
    k = 2                                # Step counter
    
    # Compute a skinny QR decomposition of X1
    qr_obj = qr(X1)
    Q = qr_obj.Q
    Q1 = Q[:, 1]
    Q2 = Q[:, Not(1)]
    R = qr_obj.R

    # Throughout the algorithm, we will maintain the decomposition 
    # X1 = Q1 * R. Dimenstions
    # X1: n x r
    # Q1: n x r
    # Q2: n x (n-r)
    # R:  r x r
    
#     if p>n maxsteps=maxsteps+1 end # 조심
    while k <= maxsteps
        
        ##### check limits
#         if k > size(action, 1)
#             buf = size(action, 1)
#             action = [action; zeros(buf)]
#             df = [df; numeric(buf)]
#             beta = [beta, zeros(p, buf)]
#         end
        
        # Key quatities for the next entry

        if k==2 R = R[1, 1] end
        a = R \ (Q1' * y0)           # r x 1
        b = R \ (Q1' * X2)          # r x (p-r) so that X1 * b is size of n * (p-r) which is equal to size of X2

        X2_resid = X2 .- X1 * b     # n x (p-r)
        z = X2_resid ./ sqrt.(sum(abs2, X2_resid, dims=1))  
        u = z' * y0 # (p-r) vector

        # Ohterwise find the next hitting time
        sign_u = sign.(u)
        abs_u = sign_u .* u
        j_hit = findmax(abs_u)[2]
        sign_hit = sign_u[j_hit]
        
        # Record action, df, solution
        action[k] = I[j_hit]
        df[k] = r
        beta[A, k] = a # beta estimators of k variables (r vars?) / A : vector
        
        # Update rest of the variables
        r = r + 1
        A = [A; I[j_hit]]
        I = I[Not(j_hit)]
        sign_A = [sign_A; sign_hit]
        X1 = [X1 X2[:, j_hit]]
        X2 = X2[:, Not(j_hit)]
        
        # Update the QR decomposition
        # new variable added to X1 (removed from X2) / qr(X1)
        if k == 2
            R = [R][:, :]
            Q1 = reshape(Q1, n, 1)
        end

        updated_qr = updateQR(Q1, Q2, R, X1[:, r])
        Q1 = updated_qr[1]
        Q2 = updated_qr[2]
        R = updated_qr[3]

        if verbose
            println("Variable added" * string(k) * string(r))
        end
        
        # update counter
        k = k + 1
    end
    
    # record df and solution at last step
    df[k] = k - 1
    beta[A, k] = R \ (Q1' * y0)
    
    # Trim
    action = action[collect(1:k-1)]
    df = df[collect(1:k)]
    beta = beta[:, collect(1:k)]
    
    # If we stopped short of the complete path, then note this
    if k - 1 < min(n-intercept, p)
        completepath = false
        bls = 0
        # else we computed the complete path, so record LS solution
    else
        completepath = true
        bls = beta[:, k]
    end
    
    # Adjust for the effect of centering and scaling
    if intercept df = df .+ 1 end
    if normalize beta = beta ./ sx' end
    if normalize & completepath
        bls = bls ./ sx'
    end

    return beta
end