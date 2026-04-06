#### #### #### #### #### #### 
#### Distributions
#### #### #### #### #### #### 
struct NoPriorBeta <: ContinuousUnivariateDistribution

    function NoPriorBeta()
        new()
    end

end

struct NoPriorSigma <: ContinuousMatrixDistribution

    function NoPriorSigma()
        new()
    end

end

#### #### #### #### #### #### 
#### Sample R
#### #### #### #### #### #### 
abstract type SampleRmat end
struct DoSampleRmat <: SampleRmat

    function DoSampleRmat()
        new()
    end

end

struct DoNotSampleRmat <: SampleRmat

    function DoNotSampleRmat()
        new()
    end

end


#### #### #### #### #### #### 
#### dimension
#### #### #### #### #### #### 
abstract type ValueP end
struct ValueP2 <: ValueP

    p::Int64

    function ValueP2()
        new(2)
    end

end

struct ValueP3 <: ValueP

    p::Int64

    function ValueP3()
        new(3)
    end

end



#### #### #### #### #### #### 
#### REflections
#### #### #### #### #### ####

abstract type Reflection end
struct KeepReflection <: Reflection

    function KeepReflection()
        new()
    end

end


struct DoNotKeepReflection <: Reflection

    function DoNotKeepReflection()
        new()
    end

end



#### #### #### #### #### #### 
#### Data Trans
#### #### #### #### #### ####

abstract type RemoveLocation end

abstract type DoRemoveLocation <: RemoveLocation end
struct DoNotRemoveLocation <: RemoveLocation 
    
    function DoNotRemoveLocation()
        new()
    end

end


struct RemoveLocationHelmert{VP<:ValueP} <: DoRemoveLocation

    matrix::Array{Float64}
    valp::VP

    function RemoveLocationHelmert(k::Int64, val::VP) where {VP<:ValueP}

        H::Matrix{Float64} = zeros(Float64, k + 1, k + 1)

        for i = 1:(k+1)
            H[i, i] = (i - 1) / (i * (i - 1))^0.5
        end
        for i = 2:(k+1)
            for j = 1:(i-1)
                H[i, j] = -1.0 / (i * (i - 1))^0.5
            end
        end

        H = H[2:end, :]

        new{VP}(H, val)

    end

end



struct RemoveLocationCentering{VP<:ValueP} <: DoRemoveLocation

    matrix::Array{Float64}
    valp::VP

    function RemoveLocationCentering(k::Int64, val::VP) where {VP<:ValueP}

        #H::Matrix{Float64} = zeros(Float64, k + 1, k + 1)
        H::Matrix{Float64} = fill(-1.0 / (k + 1), k + 1, k + 1) + I

        H = H[2:end, :]
        new{VP}(H, val)

    end

end

#struct DoNotRemoveLocation{VP<:ValueP} <: RemoveLocation

#    matrix::Array{Float64}
#    valp::VP

#    function DoNotRemoveLocation(k::Int64, val::VP) where {VP<:ValueP}

#        H::Matrix{Float64} = zeros(Float64, k, k)
#        for i = 1:k
#            H[i, i] = 1.0
#        end

#        new{VP}(H, val)
#    end

#end


abstract type RemoveSize end
abstract type DoRemoveSize <: RemoveSize end
struct DoNotRemoveSize <: RemoveSize 

    function DoNotRemoveSize()
        new()
    end

end
struct RemoveSizeNorm <: DoRemoveSize

    function RemoveSizeNorm()
        new()
    end

end

#struct DoNotRemoveSize <: RemoveSize

#    function DoNotRemoveSize()
#        new()
#    end

#end

##### #### #### #### #### #### 
##### Constraints
##### #### #### #### #### ####


abstract type IdentifiabilityConstraint end
struct GramSchmidtMean <: IdentifiabilityConstraint

    function GramSchmidtMean()

        new()

    end

end

##### #### #### #### #### #### 
##### Data
##### #### #### #### #### ####

abstract type GeneralDataType end
struct SSDataType{R<:Reflection,RL<:RemoveLocation,VP<:ValueP,RS<:RemoveSize,IC<:IdentifiabilityConstraint} <: GeneralDataType

    landmarks::Array{Float64,3}
    nolocdata::Array{Float64,3}
    ssdata::Array{Float64,3}
    sdata::Array{Float64,3}
    ssdata_rotmat::Array{Float64,3}
    reflection::R
    removelocation::RL
    valp::VP
    removesize::RS
    identifiability::IC

    n::Int64
    k::Int64
    p::Int64


    function SSDataType{R,RL,VP,RS,IC}(landmarks::Array{Float64}, nolocdata::Array{Float64,3}, ssdata::Array{Float64,3}, sdata::Array{Float64,3}, ssdata_rotmat::Array{Float64,3}, reflection::R, removelocation::RL, valp::VP, removesize::RS, identifiability::IC, n::Int64, k::Int64, p::Int64) where {R<:Reflection,RL<:RemoveLocation,VP<:ValueP,RS<:RemoveSize,IC<:IdentifiabilityConstraint}


        new{R,RL,VP,RS,IC}(landmarks, nolocdata, ssdata, sdata, ssdata_rotmat, reflection, removelocation, valp, removesize, identifiability, n, k, p)

    end

end
function SSDataType(landmarks::Array{Float64}, reflection::KeepReflection, removelocation::RL, valp::P, removesize::DoNotRemoveSize, identification::IC) where {IC<:IdentifiabilityConstraint, P<:ValueP, RL<:DoRemoveLocation}

    k::Int64 = size(landmarks, 1) - 1
    n::Int64 = size(landmarks, 3)
    p::Int64 = size(landmarks, 2)

    nolocdata = remove_location(landmarks, removelocation)

    ssdata, ssdata_rotmat = compute_sizeshape_fromnolocdata(nolocdata, reflection, valp)

    # FIXME: sdata should be the shape data
    sdata = deepcopy(ssdata)

    SSDataType{KeepReflection,RL,P,DoNotRemoveSize,IC}(landmarks, nolocdata, ssdata, sdata, ssdata_rotmat, reflection, removelocation, valp, removesize, identification, n, k, p)

end


abstract type MCMCData end
struct MCMCNormalDataKeepSize{R<:Reflection,RL<:RemoveLocation,VP<:ValueP,D<:SampleRmat} <: MCMCData

    rmat_mcmc::Array{Float64,3}
    rmat_prop::Array{Float64,3}


    yr_mcmc::Array{Float64,3}
    yr_prop::Array{Float64,3}

    angles_mcmc::Array{Float64,2}
    angles_prop::Array{Float64,2}

    rmat_dosample::D

    sdprop_adapt::Array{Float64,2}
    accratio_adapt::Array{Float64,2}
    molt_adapt::Array{Float64,2}
    
    iter_adapt::Array{Int64,2}
    init_adapt::Array{Int64,2}
    end_adapt::Array{Int64,2}

    a_adapt::Array{Float64,2}
    b_adapt::Array{Float64,2}

    sumalpha::Array{Float64,2}


    function MCMCNormalDataKeepSize{R,RL,VP,D}(rmat_mcmc::Array{Float64,3}, rmat_prop::Array{Float64,3}, yr_mcmc::Array{Float64,3}, yr_prop::Array{Float64,3}, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC}, angles_mcmc::Array{Float64,2}, angles_prop::Array{Float64,2},rmat_dosample::D, sdprop_adapt::Array{Float64,2}, accratio_adapt::Array{Float64,2}, molt_adapt::Array{Float64,2}, iter_adapt::Array{Int64,2}, init_adapt::Array{Int64,2}, end_adapt::Array{Int64,2}, a_adapt::Array{Float64,2}, b_adapt::Array{Float64,2}) where {R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint,D<:SampleRmat}

        sumalpha = deepcopy(sdprop_adapt)
        sumalpha .= 0.0
        new{R,RL,VP,D}(rmat_mcmc, rmat_prop, yr_mcmc, yr_prop, angles_mcmc,angles_prop,rmat_dosample, sdprop_adapt, accratio_adapt, molt_adapt, iter_adapt, init_adapt, end_adapt, a_adapt, b_adapt, sumalpha)

    end

   
end

function MCMCdata(rmat_init::Array{Float64,3},  datat::SSDataType{KeepReflection,RL,VP,DoNotRemoveSize,IC}, rmat_dosample::D; sdprop_adapt_init::Float64 = 0.1, accratio_adapt_init::Float64 = 0.234, molt_adapt_init::Float64 = 0.4, iter_adapt_init::Int64 = 50, init_adapt_init::Int64 = 100, end_adapt_init::Int64 = 1000,  a_adapt_init::Float64 = 100.0, b_adapt_init::Float64 = 200.0) where {RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint,D<:SampleRmat}


    if size(rmat_init,1)>0

        @assert size(rmat_init,1) == datat.p
        @assert size(rmat_init,2) == datat.p
        @assert size(rmat_init,3) == datat.n

        rmat_mcmc = deepcopy(rmat_init)
        rmat_prop = deepcopy(rmat_init)

        

    else

        rmat_mcmc = reshape(vcat([Matrix{Float64}(I, datat.p, datat.p)[:] for i = 1:datat.n]...), (datat.p, datat.p, datat.n))
        rmat_prop = reshape(vcat([Matrix{Float64}(I, datat.p, datat.p)[:] for i = 1:datat.n]...), (datat.p, datat.p, datat.n))
    end

    if datat.p == 2
        dimangle = 1
    elseif datat.p == 3
        dimangle = 3
    else
        println("datat.p must be <= 3")
    end

    angles_mcmc = zeros(Float64,dimangle,datat.n)
    for i = 1:datat.n
        compute_angle_from_rmat(i,angles_mcmc, rmat_mcmc,datat.valp, datat.reflection)
    end
    angles_prop = deepcopy(angles_mcmc)

    yr_mcmc = deepcopy(datat.nolocdata)
    yr_prop = deepcopy(datat.nolocdata)

    compute_yrdata(yr_mcmc, datat.ssdata, rmat_mcmc)
    compute_yrdata(yr_prop, datat.ssdata, rmat_prop)

    sdprop_adapt::Array{Float64,2} = sdprop_adapt_init .* ones(Float64,dimangle,datat.n)
    accratio_adapt::Array{Float64,2}  = accratio_adapt_init .* ones(Float64,dimangle,datat.n)
    molt_adapt::Array{Float64,2}  = molt_adapt_init .* ones(Float64,dimangle,datat.n)
    
    iter_adapt::Array{Int64,2} = iter_adapt_init .* ones(Int64,dimangle,datat.n)
    init_adapt::Array{Int64,2} = init_adapt_init .* ones(Int64,dimangle,datat.n)
    end_adapt::Array{Int64,2} = end_adapt_init .* ones(Int64,dimangle,datat.n)

    a_adapt::Array{Float64,2}  = a_adapt_init .* ones(Float64,dimangle,datat.n)
    b_adapt::Array{Float64,2}  = b_adapt_init .* ones(Float64,dimangle,datat.n)

    
    MCMCNormalDataKeepSize{KeepReflection,RL,VP,D}(rmat_mcmc, rmat_prop, yr_mcmc, yr_prop, datat, angles_mcmc, angles_prop,rmat_dosample, sdprop_adapt, accratio_adapt, molt_adapt, iter_adapt, init_adapt, end_adapt, a_adapt, b_adapt)

end

#function MCMCMean(meanmodel::LinearMean, valp::VP, beta_init::Matrix{Float64}, prior::D, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC})::MCMCLinearMean{D} where {D<:ContinuousUnivariateDistribution,R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint}

    

#    return MCMCLinearMean(beta_mcmc, beta_prop, mean_mcmc, mean_prop, prior)

#end



##### #### #### #### #### #### 
##### data
##### #### #### #### #### ####



##### #### #### #### #### #### 
##### Mean
##### #### #### #### #### ####


abstract type TypeModelMean end
struct LinearMean{IC<:IdentifiabilityConstraint} <: TypeModelMean

    designmatrix::Array{Float64,3} #
    d::Int64
    colnames_modelmatrix::Vector{String} # small
    model_matrix::Matrix{Float64}   # small
    identifiability::IC

    function LinearMean{IC}(designmatrix::Array{Float64,3}, d::Int64, colnames_modelmatrix::Vector{String}, designmatrix_v2::Matrix{Float64}, identident::IC) where {IC<:IdentifiabilityConstraint}

        new{IC}(designmatrix, d, colnames_modelmatrix, designmatrix_v2, identident)

    end

end

function LinearMean(fm::FormulaTerm, covariates::DataFrame, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC}, identident::IC) where {R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint}

    designmatrix, d, colnames_modelmatrix, designmatrix_v2 = create_designmatrix(fm, covariates, datat.k)

    LinearMean{IC}(designmatrix, d, colnames_modelmatrix, designmatrix_v2, identident)
end


abstract type MCMCTypeModelMean end
struct MCMCLinearMean{D<:ContinuousUnivariateDistribution} <: MCMCTypeModelMean

    beta_mcmc::Matrix{Float64}
    beta_prop::Matrix{Float64}

    mean_mcmc::Array{Float64,3}
    mean_prop::Array{Float64,3}

    prior_beta::D

    function MCMCLinearMean{D}(beta_mcmc::Matrix{Float64}, beta_prop::Matrix{Float64}, mean_mcmc::Array{Float64,3}, mean_prop::Array{Float64,3}, prior::D) where {D<:ContinuousUnivariateDistribution}

        new{D}(beta_mcmc, beta_prop, mean_mcmc, mean_prop, prior)
    end
end


function MCMCMean(meanmodel::LinearMean, valp::VP, beta_init::Matrix{Float64}, prior::D, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC})::MCMCLinearMean{D} where {D<:ContinuousUnivariateDistribution,R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint}

    d::Int64 = meanmodel.d
    k::Int64 = size(meanmodel.designmatrix, 1)
    p::Int64 = valp.p
    n::Int64 = size(meanmodel.model_matrix, 1)


    if typeof(prior) <: Normal
    elseif typeof(prior) <: NoPriorBeta
    else
        error("beta_prior should be Normal or NoPriorBeta")
    end
    if size(beta_init, 1) > 0
        @assert size(beta_init, 1) >= k * d "size(beta_init,1) must be at least " * string(k * d)
        @assert size(beta_init, 2) >= p "size(beta_init,2) must be at least " * string(p)
    else
        beta_init = zeros(Float64, (k * d), p)
    end


    beta_mcmc = deepcopy(beta_init[1:(k*d), 1:p])
    beta_prop = deepcopy(beta_init[1:(k*d), 1:p])


    mean_mcmc = zeros(Float64, k, p, n)

    for ip = 1:p

        for j = 1:n
            mean_mcmc[:, ip, j] = meanmodel.designmatrix[:, :, j] * beta_mcmc[:, ip]
        end
    end
    mean_prop = deepcopy(mean_mcmc)

    return MCMCLinearMean{D}(beta_mcmc, beta_prop, mean_mcmc, mean_prop, prior)

end
###### #### #### #### #### #### 
###### Covariance
###### #### #### #### #### ####


abstract type TypeModelCoVariance end
struct GeneralCoVarianceIndependentDimension{IC<:IdentifiabilityConstraint} <: TypeModelCoVariance

    identifiability::IC

    function GeneralCoVarianceIndependentDimension{IC}(ident::IC) where {IC<:IdentifiabilityConstraint}
        new{IC}(ident)
    end

end

function GeneralCoVarianceIndependentDimension(ident::IC, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC}) where {R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint}

    return GeneralCoVarianceIndependentDimension{IC}(ident)
end


abstract type MCMCTypeModelCoVariance end
struct MCMCGeneralCoVarianceIndependentDimension{D<:ContinuousMatrixDistribution} <: MCMCTypeModelCoVariance

    covariance_mcmc::Symmetric{Float64,Matrix{Float64}}
    covariance_prop::Symmetric{Float64,Matrix{Float64}}

    invcovariance_mcmc::Symmetric{Float64,Matrix{Float64}}
    invcovariance_prop::Symmetric{Float64,Matrix{Float64}}

    logdeterminant_mcmc::Vector{Float64}
    logdeterminant_prop::Vector{Float64}

    prior_covariance::D


    function MCMCGeneralCoVarianceIndependentDimension{D}(covariance_mcmc::Symmetric{Float64,Matrix{Float64}}, covariance_prop::Symmetric{Float64,Matrix{Float64}}, invcovariance_mcmc::Symmetric{Float64,Matrix{Float64}}, invcovariance_prop::Symmetric{Float64,Matrix{Float64}}, logdeterminant_mcmc::Vector{Float64}, logdeterminant_prop::Vector{Float64}, prior_covariance::D) where {D<:ContinuousMatrixDistribution}


        new{D}(covariance_mcmc, covariance_prop, invcovariance_mcmc, invcovariance_prop, logdeterminant_mcmc, logdeterminant_prop, prior_covariance)

    end
end

function MCMCCovariance(covmodel::GeneralCoVarianceIndependentDimension, sigma_init::Symmetric{Float64,Matrix{Float64}}, prior::D, datat::SSDataType{R,RL,VP,DoNotRemoveSize,IC})::MCMCGeneralCoVarianceIndependentDimension{D} where {D<:ContinuousMatrixDistribution,R<:Reflection,RL<:RemoveLocation,VP<:ValueP,IC<:IdentifiabilityConstraint}

    k::Int64 = datat.k
    if size(sigma_init.data, 1) > 0
        covariance_mcmc = Symmetric(deepcopy(sigma_init.data))
    else
        covariance_mcmc = Symmetric(Matrix{Float64}(I, k, k))
    end
    covariance_prop = deepcopy(covariance_mcmc)

    cc = cholesky(covariance_mcmc)
    invcovariance_mcmc = Symmetric(inv(cc))
    logdeterminant_mcmc = [log(det(cc))]

    invcovariance_prop = Symmetric(deepcopy(invcovariance_mcmc))
    logdeterminant_prop = deepcopy(logdeterminant_mcmc)

    return MCMCGeneralCoVarianceIndependentDimension{D}(covariance_mcmc, covariance_prop, invcovariance_mcmc, invcovariance_prop, logdeterminant_mcmc, logdeterminant_prop, prior)

end

###### #### #### #### #### #### 
###### mcmcOUT
###### #### #### #### #### ####

abstract type MCMCObjectOUT end

struct generalMCMCObjectOUT <:MCMCObjectOUT

    nonidentbeta::Array{Float64,3} 
    nonidentsigma::Array{Float64,3}
    nonidentrmat::Array{Float64,4}
    nonidentangle::Array{Float64,3} 

    identbeta::Array{Float64,3} 
    identsigma::Array{Float64,3}
    identrmat::Array{Float64,4}
    identangle::Array{Float64,3} 

    beta::DataFrame
    sigma::DataFrame
    rmat::DataFrame
    angle::DataFrame

    postsample::Int64
    
    function generalMCMCObjectOUT(nonidentbeta::Array{Float64,3} , nonidentsigma::Array{Float64,3}, nonidentrmat::Array{Float64,4}, nonidentangle::Array{Float64,3}, identbeta::Array{Float64,3},  identsigma::Array{Float64,3}, identrmat::Array{Float64,4}, identangle::Array{Float64,3}, beta::DataFrame,  sigma::DataFrame, rmat::DataFrame, angle::DataFrame, postsample::Int64 )

        new(nonidentbeta, nonidentsigma, nonidentrmat, nonidentangle, identbeta,  identsigma, identrmat, identangle, beta,  sigma, rmat, angle,postsample)

    end
end

function create_object_output(sampletosave::Int64,mean_mcmc::MCMCLinearMean, covariance_mcmc::MCMCGeneralCoVarianceIndependentDimension, data_mcmc::MCMCNormalDataKeepSize, mean_model::LinearMean)

    p::Int64 = size(mean_mcmc.beta_mcmc,2)
    n::Int64 = size(data_mcmc.rmat_mcmc,3)
    k::Int64 = size(covariance_mcmc.covariance_mcmc,1)
    kd::Int64 = size(mean_mcmc.beta_mcmc,1) # size kd means flattened betas
    d::Int64 = mean_model.d
    pangle::Int64 = size(data_mcmc.angles_mcmc,1)
    sizebetaout1::Int64 = 0
    sizebetaout2::Int64 = 0
    sizebetaout1 = size(mean_mcmc.beta_mcmc,1)
    sizebetaout2 = p
    betaOUT::Array{Float64,3} = zeros(Float64, sampletosave, sizebetaout1,sizebetaout2)
    #println(size(betaOUT))


    
    sizesigmaout1::Int64 = 0
    sizesigmaout1 = size(covariance_mcmc.covariance_mcmc,1)
    
    sigmaOUT::Array{Float64,3} = zeros(Float64, sampletosave, sizesigmaout1,sizesigmaout1)
    
    rmatOUT::Array{Float64,4} = zeros(Float64, sampletosave, p,p,n)
    angleOUT::Array{Float64,3} = zeros(Float64, sampletosave, pangle,n)
    
    betaidentOUT = deepcopy(betaOUT)
    sigmaidentOUT = deepcopy(sigmaOUT)
    rmatidentOUT = deepcopy(rmatOUT)
    angleidentOUT = deepcopy(angleOUT)


    betaDF = DataFrame(reshape(deepcopy(betaOUT), sampletosave, k*d*p ), :auto)
    rename!(betaDF,repeat(mean_model.colnames_modelmatrix,inner = k, outer= p) .* "| mark:" .* string.(repeat(1:k, outer = d*p)) .* "| dim:" .* string.(repeat(1:p, inner = d*k)))

    sigmaDF = DataFrame(reshape(deepcopy(sigmaOUT), sampletosave, k*k ), :auto)
    rename!(sigmaDF,"s_(".* string.(repeat(1:k, outer = k)) .* "," .* string.(repeat(1:k, inner = k)) .* ")" )

    rmatDF = DataFrame(reshape(deepcopy(rmatOUT), sampletosave, p*p*n ), :auto)
    rename!(rmatDF,"R_".*  string.(repeat(1:n,inner=p*p)) .* ",(".*repeat(string.(repeat(1:p, outer = p)) .* "," .* string.(repeat(1:p, inner = p)),outer = n) .* ")" )

    #println("Pangle", pangle)
    #println(size(angleOUT))
    angleDF = DataFrame(reshape(deepcopy(angleOUT), sampletosave, pangle*n ), :auto)
    rename!(angleDF ,"theta_".* string.(repeat(1:n,inner=pangle)) .* ",(" .* string.(repeat(1:pangle,outer=n)) .* ")")


    generalMCMCObjectOUT(betaOUT, sigmaOUT,rmatOUT,angleOUT,betaidentOUT,sigmaidentOUT,rmatidentOUT,angleidentOUT, betaDF,sigmaDF,rmatDF,angleDF, sampletosave )
end

function copy_parameters_out(imcmc::Int64, out::generalMCMCObjectOUT, mean_mcmc::MCMCLinearMean, datamodel::SSDataType{<:KeepReflection, <:DoRemoveLocation, <:ValueP, <:DoNotRemoveSize,<:GramSchmidtMean}, covariance_mcmc::MCMCGeneralCoVarianceIndependentDimension, data_mcmc::MCMCNormalDataKeepSize)


    out.nonidentbeta[imcmc,:,:] = mean_mcmc.beta_mcmc[:,:]
    out.nonidentsigma[imcmc,:,:] =  covariance_mcmc.covariance_mcmc[:,:]
    out.nonidentrmat[imcmc,:,:,:] = data_mcmc.rmat_mcmc[:,:,:]
    out.nonidentangle[imcmc,:,:] = data_mcmc.angles_mcmc[:,:]
    Raw_data = datamodel.nolocdata[:, :, 1]
    n_points = size(Raw_data, 1)

    max_dist_sq = -Inf
    pair = (0, 0)

    for i in 1:n_points
        for j in (i + 1):n_points
            diff = Raw_data[i, :] .- Raw_data[j, :]
            dist_sq = sum(diff .^ 2)
        
            if dist_sq > max_dist_sq
                max_dist_sq = dist_sq
                pair = (i, j)
            end
        end
    end
    M_raw = mean_mcmc.mean_mcmc[:, :, 1] # average configuration for the first iteration
    M_anchored = M_raw .- M_raw[pair[1], :]' # subtract the first row to have first landmark in (0,0)
    r = sqrt(sum(M_anchored[pair[2],:].^2)) # compute the distance from the origin
    x = M_anchored[pair[2],1]
    y = M_anchored[pair[2],2]
    gammamat = [ x/r  -y/r ; 
             y/r   x/r ]

    #gammamat = standardize_reg_computegamma(mean_mcmc.beta_mcmc, datamodel.valp,datamodel.identifiability)
    #gammamat = standardize_reg_computegamma_gimmi(mean_mcmc.beta_mcmc, datamodel.valp,datamodel.identifiability)
    #out.identbeta[imcmc,:,:] = (mean_mcmc.beta_mcmc .-mean_mcmc.beta_mcmc[1,:]') *gammamat
    out.identbeta[imcmc,:,:] = mean_mcmc.beta_mcmc *gammamat
    out.identsigma[imcmc,:,:] = covariance_mcmc.covariance_mcmc[:,:]
    app_rot = deepcopy(data_mcmc.rmat_mcmc)
    app_angle = deepcopy(data_mcmc.angles_mcmc)
    for i = 1:size(data_mcmc.rmat_mcmc,3)
        app_rot[:,:,i] .= transpose(gammamat)*app_rot[:,:,i]
        #app_rot[:,:,i] .= app_rot[:,:,i]*transpose(gammamat)
        compute_angle_from_rmat(i,app_angle, app_rot, datamodel.valp, datamodel.reflection)
        #@assert isapprox(det(app_rot[:,:,i]),1.0)  "ss" * string(det(app_rot[:,:,i]))
    end

    #out.identbeta[imcmc,:,:] = mean_mcmc.beta_mcmc
    #out.identsigma[imcmc,:,:] = covariance_mcmc.covariance_mcmc[:,:]
    #app_rot = deepcopy(data_mcmc.rmat_mcmc)
    #app_angle = deepcopy(data_mcmc.angles_mcmc)
    #for i = 1:size(data_mcmc.rmat_mcmc,3)
    #    app_rot[:,:,i] = app_rot[:,:,i]
    #    compute_angle_from_rmat(i,app_angle, app_rot, datamodel.valp, datamodel.reflection)
    #    #@assert isapprox(det(app_rot[:,:,i]),1.0)  "ss" * string(det(app_rot[:,:,i]))
    #end


    out.identrmat[imcmc,:,:,:] = app_rot[:,:,:]
    out.identangle[imcmc,:,:] = app_angle[:,:]

    out.beta[imcmc,:] = out.identbeta[imcmc,:,:][:]
    out.sigma[imcmc,:] = out.identsigma[imcmc,:,:][:]
    out.rmat[imcmc,:] = out.identrmat[imcmc,:,:,:][:]
    out.angle[imcmc,:] = out.identangle[imcmc,:,:][:]

end

