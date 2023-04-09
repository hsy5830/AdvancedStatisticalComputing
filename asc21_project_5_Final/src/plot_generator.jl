using Plots, RCall

function gen_plot(n::Integer, p::Integer, s::Integer; iter = 10)
    νgrid = [0.05, 0.09, 0.14, 0.25, 0.42, 0.71, 1.22, 2.07, 3.52, 6.00]
    ρgrid = [0, 0.35, 0.7]
    typ_grid = [1, 2, 3, 5]

    label = ["Best subset" "Forward stepwise" "Lasso" "Relaxed lasso"]
    ylabel = ["Relative risk (to null model)",
            "Relative test error (to Bayes)",
            "Proportion of variance explained",
            "Number of nonzeros",
            "F classification of nonzeros"]
    color = [:coral1 :yellowgreen :turquoise3 :mediumorchid2]

    RR = zeros(4, iter, length(νgrid))
    RTE = zeros(4, iter, length(νgrid))
    PVE = zeros(4, iter, length(νgrid))
    NN = zeros(4, iter, length(νgrid))
    FS = zeros(4, iter, length(νgrid))

    elapsed_time = zeros(4, length(typ_grid), length(ρgrid), length(νgrid), iter)  # methods, type, ρ, ν, iterations
    fig = Array{Any}(undef, length(typ_grid), length(ρgrid), 5)  # type, ρ, metrics

    for i in 1:length(typ_grid)
        typ = typ_grid[i]

        for j in 1:length(ρgrid)
            ρ = ρgrid[j]

            println("=========================== beta-type = $typ and ρ = $ρ starts ===========================")
            
            for m in 1:length(νgrid)
                ν = νgrid[m]

                println("==================================== ν = $ν starts ====================================")
                
                R"""
                library(bestsubset)
                set.seed(0)
                """

                for l in 1:iter

                    println("================================== iteration $l starts ===================================")

                    # the authors generate data in this way
                    # we use the same data to reproduce the results
                    @rput n
                    @rput p
                    @rput s
                    @rput ρ
                    @rput ν
                    @rput typ

                    R"""
                    data = bestsubset::sim.xy(n = n, p = p, nval = n, snr = ν, rho = ρ, beta.type = typ, s = s)
                    X_train = data$x
                    y_train = data$y
                    X_test = data$xval
                    y_test = data$yval
                    β = data$beta
                    Σ = data$Sigma
                    σ = data$sigma
                    """

                    @rget X_train
                    @rget y_train
                    @rget X_test
                    @rget y_test
                    @rget β
                    @rget Σ
                    @rget σ
                    
                    
                    # it is reasonable to set "intercept = false" since population mean is zero and the authors also implemented in this way
                    #################   Best subset   #################
                    t_bs = @elapsed β_path_bs, steps = bestsubset(X_train, y_train; k = 0:10, timelimit = 100, verbose = false, intercept = false)
                    test_bs = zeros(size(β_path_bs, 2))
                    
                    for i in 1:size(β_path_bs, 2)
                        β_test = β_path_bs[:, i]
                        test_bs[i] = norm(X_test * β - X_test * β_test)^2 / norm(X_test * β)^2
                    end
                    
                    β̂_bs = β_path_bs[: ,argmin(test_bs)]
                    

                    ##############   Forward selection   ##############
                    t_fs = @elapsed β_path_fs = fs(X_train, y_train, intercept = false)
                    test_fs = zeros(size(β_path_fs, 2))
                    
                    for i in 1:size(β_path_fs, 2)           
                        β_test = β_path_fs[:, i]
                        test_fs[i] = norm(X_test * β - X_test * β_test)^2 / norm(X_test * β)^2
                    end        
                     
                    β̂_fs = β_path_fs[: ,argmin(test_fs)]   


                    ################   Lasso (GLMNet)   ################
                    t_ls = @elapsed β_path_lasso = lasso_glmnet(X_train, y_train, nlambda = 50, nrelax = 1, intercept = false)[1]  
                    test_lasso = zeros(size(β_path_lasso, 2))
                    
                    for i in 1:size(β_path_lasso, 2)           
                        β_test = β_path_lasso[:, i]
                        test_lasso[i] = norm(X_test * β - X_test * β_test)^2 / norm(X_test * β)^2
                    end
                    
                    β̂_lasso = β_path_lasso[:, argmin(test_lasso)]   
                    

                    #############   Relaxed lasso (GLMNet)   #############
                    t_rl = @elapsed β_path_relaxo = lasso_glmnet(X_train, y_train, nlambda = 50, nrelax = 10, intercept = false)[1]
                    test_relaxo = zeros(size(β_path_relaxo, 2))
                    
                    for i in 1:size(β_path_relaxo, 2)           
                        β_test = β_path_relaxo[:, i]
                        test_relaxo[i] = norm(X_test * β - X_test * β_test)^2 / norm(X_test * β)^2
                    end
                    
                    β̂_relaxo = β_path_relaxo[:, argmin(test_relaxo)]  
                    

                    # record elapsed time
                    elapsed_time[1, i, j, m, l] = t_bs
                    elapsed_time[2, i, j, m, l] = t_fs
                    elapsed_time[3, i, j, m, l] = t_ls
                    elapsed_time[4, i, j, m, l] = t_rl


                    # compute metrics
                    r = 1
                    for β̂ in [β̂_bs, β̂_fs, β̂_lasso, β̂_relaxo]
                        RR[r, l, m] = ((β̂ .- β)' * Σ * (β̂ .- β)) / (β' * Σ * β)
                        RTE[r, l, m] = ((β̂ .- β)' * Σ * (β̂ .- β) + σ^2) / σ^2
                        PVE[r, l, m] = 1 - ((β̂ .- β)' * Σ * (β̂ .- β) + σ^2) / (β' * Σ * β + σ^2)
                        NN[r, l, m] = count(!iszero, β̂)
                        
                        nzidx = β̂ .!= 0
                        nzidx0 = β .!= 0
            
                        prec = sum(nzidx[nzidx0]) / sum(nzidx)
                        recall = count(!iszero, β̂[nzidx0]) / sum(nzidx0)
                        
                        if prec == 0 || recall == 0
                            FS[r, l, m] = 0
                        else
                            FS[r, l, m] = 2 / ((1 / recall) + (1 / prec))
                        end
                        
                        r += 1
                    end
                end
            end

            for (k, obj) in zip(1:5, [RR, RTE, PVE, NN, FS])
                err = [std(obj[o, :, :], dims = 1) / sqrt(iter) for o in 1:4]
                err = [err[1]' err[2]' err[3]' err[4]']
                
                plt = zeros(length(νgrid), 4)
                for i in 1:4
                    plt[:, i] = mean(obj[i, :, :], dims = 1)
                end

                subfig = plot(νgrid, plt, label = label, legend = :none, yerror = err,
                              xscale = :log10, xticks = (νgrid, [string(t) for t in νgrid]),
                              line = (3, color), marker = (:circle, color), markerstrokewidth = 1, markerstrokecolor = color,
                              xlabel = "Signal-to-noise ratio", ylabel = ylabel[k], framestyle = :box)

                if obj == NN
                    hline!(subfig, [s], line = :dot, color = :black, label = "")
                elseif obj == PVE
                    plot!(subfig, νgrid, νgrid ./ (νgrid .+ 1), line = :dot, color = :black, label = "")
                end
                
                fig[i, j, k] = subfig
            end
        end
    end

    return fig, elapsed_time
end