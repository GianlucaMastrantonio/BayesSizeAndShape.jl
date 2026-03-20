
"""
    sizeshape_helmertproduct_reflection(dataset::Array{Float64,3})

The function computes the Size-And-Shape version of the data `dataset`, with reflection information.  The output is computed using the helmert matrix and the SVD trasformation

"""
function sizeshape_helmertproduct_reflection(dataset::Array{Float64,3}, removelocation::DoRemoveLocation)

    n = size(dataset,3)
    k = size(dataset,1)-1
    p = size(dataset,2)

    
    if p == 2
        #helmmat = RemoveLocationHelmert(k, ValueP2());
        #posthelmert = remove_location(dataset, helmmat);
        postloc = remove_location(dataset, removelocation)
        sizeandshape, _ = compute_sizeshape_fromnolocdata(postloc, BayesSizeAndShape.KeepReflection(), BayesSizeAndShape.ValueP2());
    elseif p ==3
        #helmmat = RemoveLocationHelmert(k, ValueP3());
        #posthelmert = remove_location(dataset, helmmat);
        postloc = remove_location(dataset, removelocation)
        sizeandshape, _ = compute_sizeshape_fromnolocdata(postloc, BayesSizeAndShape.KeepReflection(), BayesSizeAndShape.ValueP3());
    else
        error("p must be <= 3")
    end
    
    
    return sizeandshape

end



function sizeshape_helmertproduct_reflection_user(dataset::Array{Float64,3}, method::String)
    
    k = size(dataset, 1) - 1
    p = size(dataset, 2)
    if p == 2
        valp = ValueP2()
    elseif p == 3
        valp = ValueP3()
    else
        error("Dimension p must be 2 or 3.")
    end

    if method == "RemoveLocationCentering" || method == "centering"
        loc_remover = RemoveLocationCentering(k, valp)
    elseif method == "RemoveLocationHelmert" || method == "helmert"
        loc_remover = RemoveLocationHelmert(k, valp)
    else
        error("Method not recognized. Use 'centering' or 'helmert'.")
    end
    return sizeshape_helmertproduct_reflection(dataset, loc_remover)
end


"""
    posterior_samples_beta(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
        RL<:RemoveLocation,
        CT<:TypeModelCoVariance,
        CM<:MCMCTypeModelCoVariance,
        PS<:MCMCObjectOUT,
        P<:ValueP   
    }
The function extract the posterior sample of the regressive coefficients from an object of type `SizeAndShapeModelOutput` 
      
"""
function posterior_samples_beta(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
    RL<:RemoveLocation,
    CT<:TypeModelCoVariance,
    CM<:MCMCTypeModelCoVariance,
    PS<:MCMCObjectOUT,
    P<:ValueP

    
}

    return modeloutput.posteriorsamples.beta

end

"""
    posterior_samples_sigma(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
        RL<:RemoveLocation,
        CT<:TypeModelCoVariance,
        CM<:MCMCTypeModelCoVariance,
        PS<:MCMCObjectOUT,
        P<:ValueP
        }

The function extract the posterior sample of the covariance matrix from an object of type `SizeAndShapeModelOutput`   

"""
function posterior_samples_sigma(modeloutput::SizeAndShapeModelOutput{KeepReflection,RL,P,DoNotRemoveSize,GramSchmidtMean,<:MCMCNormalDataKeepSize,<:LinearMean,<:MCMCLinearMean,CT,CM,PS}) where {
    RL<:RemoveLocation,
    CT<:TypeModelCoVariance,
    CM<:MCMCTypeModelCoVariance,
    PS<:MCMCObjectOUT,
    P<:ValueP

    
}

    return modeloutput.posteriorsamples.sigma

end