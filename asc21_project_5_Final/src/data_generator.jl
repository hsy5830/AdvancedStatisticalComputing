using LinearAlgebra, Statistics, Distributions, Random

# generate the design matrix X
gen_pred = function(n::Integer, p::Integer, rho::Number)
    Σ = zeros(p, p)
    
    for i in 1:p
        for j in 1:p
            Σ[i, j] = rho^abs(i - j)
        end
    end
    
    X = Matrix(rand(MvNormal(zeros(p), Σ), n)')
    return X, Σ
end

# generate the response variable
gen_resp = function(X::AbstractMatrix{T}, β::AbstractArray{T}, Σ::AbstractMatrix{T}, ν::Number) where T <: Float64
    σ2 = β' * Σ * β / ν
    
    Y = vec(rand(MvNormal(X * β, σ2 * I), 1))
    return Y
end

# generate the coefficients
gen_beta = function(typ::Integer, p::Integer, s::Integer)
    if s >= p
        error("sparsity level error!")
    end
    
    if !(typ in [1,2,3,5])
        error("Type error!")
    end
    
    β = zeros(p)
    
    if typ == 1
        r = floor(Int, p / s)
        for i in 1:s
            β[i * r] = 1
        end
        
    elseif typ == 2
        β[1:s] .= 1
        
    elseif typ == 3
        for i in 1:s
            d = -9.5 / (s - 1)
            β[i] = 10 + (i - 1) * d
        end
        
    else typ == 5
        β[1:s] .= 1
        for i in s+1:p
            β[i] = 0.5^(i - s)
        end
    end
    
    return β
end