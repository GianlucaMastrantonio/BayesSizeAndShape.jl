

function sample_predictive_zbr(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
        RL<:RemoveLocation,
        CT<:TypeModelCoVariance,
        CM<:MCMCTypeModelCoVariance,
        PS<:MCMCObjectOUT,
        P<:ValueP

    }
    
    designmatrix = modeloutput.meantype.designmatrix
    beta = modeloutput.posteriorsamples.identbeta
    rmat = modeloutput.posteriorsamples.identrmat
    nsim::Int64 = size(beta,1)

    n::Int64 = modeloutput.datatype.n
    k::Int64 = modeloutput.datatype.k
    p::Int64 = modeloutput.datatype.p

    
    res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
    for iobs = 1:n

        for i = 1:size(beta,1)

            res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs])[:]
            #res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:])[:]
    
        end
    end

    rename!(res, "mu_".* string.(repeat(1:n, inner = k*p)) .* ",(" .* string.(repeat(1:k, outer = p*n)) .* "," .* string.(repeat(repeat(1:p, inner = k),outer = n)) .* ")" )
    return res


    #res = [zeros(Float64, nsim, k,p) for i = 1:n];
    #for iobs = 1:n

    #    for i = 1:size(beta,1)

    #        res[iobs][i,:,:] = designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs]
    
    #    end
    #end
    #res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
    #return res

end



function sample_predictive_zb(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
        RL<:RemoveLocation,
        CT<:TypeModelCoVariance,
        CM<:MCMCTypeModelCoVariance,
        PS<:MCMCObjectOUT,
        P<:ValueP

    }
    
    designmatrix = modeloutput.meantype.designmatrix
    beta = modeloutput.posteriorsamples.identbeta
    rmat = modeloutput.posteriorsamples.identrmat
    nsim::Int64 = size(beta,1)

    n::Int64 = modeloutput.datatype.n
    k::Int64 = modeloutput.datatype.k
    p::Int64 = modeloutput.datatype.p
    app::Matrix{Float64} = zeros(k, p)

    
    res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
    for iobs = 1:n # for each observation --> the logic is very simple 
        # to make it identifiable, we translate one mean landmark to be in (0, 0) and another (the most distant but it should be common to every unit) on the (x, 0) point
        # in this way, mean configuration are perfectly identifiable

        for i = 1:size(beta,1) # for each iteration
            app .= reshape(Vector((designmatrix[:,:,iobs]*beta[i,:,:])[:]), k, p )
            app .-= app[10,:]' 
            v = app[1,:]
            r = sqrt(sum(v.^2))
            Q = [v[1]/r -v[2]/r; 
            v[2]/r  v[1]/r]
            app[:, :] .= app[:,:] * Q
    
            res[i,((iobs-1)*k*p) .+ (1:(k*p))] = reshape(app, k*p, 1)
            #res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:])[:]
            
            #res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:])[:]
    
        end
    end

    rename!(res, "mu_".* string.(repeat(1:n, inner = k*p)) .* ",(" .* string.(repeat(1:k, outer = p*n)) .* "," .* string.(repeat(repeat(1:p, inner = k),outer = n)) .* ")" )
    return res


    #res = [zeros(Float64, nsim, k,p) for i = 1:n];
    #for iobs = 1:n

    #    for i = 1:size(beta,1)

    #        res[iobs][i,:,:] = designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs]
    
    #    end
    #end
    #res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
    #return res

end




function sample_predictive_zbr_plus_epsilon(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
    RL<:RemoveLocation,
    CT<:TypeModelCoVariance,
    CM<:MCMCTypeModelCoVariance,
    PS<:MCMCObjectOUT,
    P<:ValueP

}

designmatrix = modeloutput.meantype.designmatrix
beta = modeloutput.posteriorsamples.identbeta
sigma = modeloutput.posteriorsamples.identsigma
rmat = modeloutput.posteriorsamples.identrmat
nsim::Int64 = size(beta,1)

n::Int64 = modeloutput.datatype.n
k::Int64 = modeloutput.datatype.k
p::Int64 = modeloutput.datatype.p


res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
for iobs = 1:n
    
    for i = 1:size(beta,1)
        app .= reshape(Vector((designmatrix[:,:,iobs]*beta[i,:,:])[:]), k, p )
        app .-= app[10,:]'
        v = app[1,:]
        r = sqrt(sum(v.^2))
        Q = [v[1]/r -v[2]/r; 
        v[2]/r  v[1]/r]
        app[:, :] .= app[:,:] * Q  *rmat[i,:,:,iobs]
    
        res[i,((iobs-1)*k*p) .+ (1:(k*p))] = reshape(app, k*p, 1) +  vcat([Q*  rand(MvNormal([0.0 for i = 1:k],Symmetric(sigma[i,:,:]))) for iii = 1:p]...)
        
        

        #res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs])[:] + vcat([rand(MvNormal([0.0 for i = 1:k],Symmetric(sigma[i,:,:]))) for iii = 1:p]...)
        
      
        #for ip = 1:p
            
        #    res[i,((iobs-1)*k*p + (p-1)*k)  .+ (1:k)] =  res[i,((iobs-1)*k*p + (p-1)*k)  .+ (1:k)] 

        #end
        
        #res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:])[:]

    end


    
end

rename!(res, "X_".* string.(repeat(1:n, inner = k*p)) .* ",(" .* string.(repeat(1:k, outer = p*n)) .* "," .* string.(repeat(repeat(1:p, inner = k),outer = n)) .* ")" )
return res


#res = [zeros(Float64, nsim, k,p) for i = 1:n];
#for iobs = 1:n

#    for i = 1:size(beta,1)

#        res[iobs][i,:,:] = designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs]

#    end
#end
#res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
#return res

end



