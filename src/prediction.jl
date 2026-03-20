

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

    
    res = DataFrame(zeros(Float64,nsim,k*p*n),:auto)
    for iobs = 1:n

        for i = 1:size(beta,1)

            res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:])[:]
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
        
        

        res[i,((iobs-1)*k*p) .+ (1:(k*p))] = (designmatrix[:,:,iobs]*beta[i,:,:]*rmat[i,:,:,iobs])[:] + vcat([rand(MvNormal([0.0 for i = 1:k],Symmetric(sigma[i,:,:]))) for iii = 1:p]...)
        
      
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



