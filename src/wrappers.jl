
"""
    SizeAndShapeWithReflectionMCMC(
        landmarks::Array{Float64,3}, 
        fm::FormulaTerm,
        covariates::DataFrame, 
        iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}},
        betaprior::ContinuousUnivariateDistribution,
        sigmaprior::ContinuousMatrixDistribution
    )

Posterior samples from the size-and-shape model - in this version, only two-dimensional data with reflection information are allowed. \\
The functions returns an object of type  `SizeAndShapeModelOutput`.
    
# Arguments

Let 
* `n` be the number of objects;
* `k+1` be the number of recorded landmark for each object
* `p` be the dimension of each landmark (only p=2 or p=3)

The arguments of the functions are 
* `landmarks`: a three-dimensional  `Array` of dimension ``(k+1)\\times p \\times n``  with the data; 
* `fm`:  a `formula`, where on the left-hand side there must be 1 and on the right-hand side there is the actual regressive formula - an intercept is needed;
* `covariates`: a `DataFrame` of covariates. The formula `fm` search for the covariates in the `DataFrame` column names;
* `iterations`: a `NamedTuple` with `iter`, `burnin`, and `thin` values of the MCMC algorithm
* `betaprior`:  a `Normal` distribution that is used as prior for all regressive coefficients
* `sigmaprior`: an `InverseWishart` distribution that is used as prior for the covariance matrix.
"""
function SizeAndShapeWithReflectionMCMC(
    landmarks::Array{Float64,3}, 
    fm::FormulaTerm,
    covariates::DataFrame, #
    iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}},
    betaprior::ContinuousUnivariateDistribution,
    sigmaprior::ContinuousMatrixDistribution
)

    
    return generalSizeAndShapeMCMC(;
    landmarks = landmarks,
    fm = fm,
    covariates = covariates,
    iterations = iterations,
    betaprior = betaprior,
    sigmaprior = sigmaprior,
    #beta_init::Matrix{Float64} = zeros(Float64,0,0),
    #sigma_init::Symmetric{Float64,Matrix{Float64}} = Symmetric(zeros(Float64,0,0)),
    #rmat_init::Array{Float64,3} = zeros(Float64,0,0,0),
    #dormat::Bool,
    identification = "gramschmidt",
    meanmodel = "linear",
    covariancemodel = "general_nocrosscorrelation",
    keepreflection = "yes",
    removelocation= "centering",
    removesize = "no",
    rmatdosample = true,
    verbose = false
)
end


#"""
#    SizeAndShapeMCMC(;
#        dataset::Array{Float64,3}, 
#        fm::FormulaTerm = @formula(1~ 1),
#        covariates::DataFrame,
#        iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}} =(
#            iter=1000,
#            burnin=200,
#            thin=2
#        ),
#        betaprior::ContinuousUnivariateDistribution = Normal(0.0, 10000.0),
#        sigmaprior::ContinuousMatrixDistribution,
#        beta_init::Matrix{Float64},
#        sigma_init::Symmetric{Float64,Matrix{Float64}},
#        rmat_init::Array{Float64,3},
#        reflection::Bool = true
#        )

#Posterior samples from the size-and-shape model - in this version, only two-dimensional data with reflection information are allowed. 
#The functions returns 2 Dataframes, with the posterior samples of the regressive coefficients and elements of the covariance matrix.


## Arguments
#Let 
#* n be the number of shapes;
#* k be the number of landmarks (on the pre-form matrix);
#* p be the dimension of each landmark (only p=2 is implemented)
#* d be the number of covariates to use + 1 (or number of regressive coefficients for each landmark-dimension, intercept included);


#The arguments are

#- `dataset::Array{Float64,3}`: Array with the data - dimension (k,p,n) of size-and-shape data. Use the function `compute_ss_from_pre` to obtain the size-and-shape data from pre-forms
#- `fm::FormulaTerm = @formula(1~ 1)`: a formula that specifies the model - the left-and size should be 1
#- `covariates::DataFrame`: a DataFrame containing the covariates - dimension (n,d). The names used in `fm` must be column names of  `covariates`. Only Real And Factor covariates are allowed. The numeric column are standardized internally.
#- `iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}}`: values of the iterations (iter), thin and burnin of the MCMC algorithm
#- `betaprior::ContinuousUnivariateDistribution`: The prior on the regressive coefficients - only a Normal distribution is allowed
#- `sigmaprior::ContinuousMatrixDistribution`: The prior on the covariance matrix - only an Inverse Wishart is allowed
#- `beta_init::Matrix{Float64}`: initial values for the regressive coefficients - dimension (k*d,p) 
#- `sigma_init::Symmetric{Float64,Matrix{Float64}}`: initial values for the covariance matrix - dimension (k,k) (it must be a valid covariance matrix) 
#- `rmat_init::Array{Float64,3}`: initial values for the rotation matrices - dimension (p,p,n) (each [1:p, 1:p, i], i = 1,2,...,n, must be a valid rotation matrix) 
#- `reflection::Bool`: true for a model with reflection information.
#"""
#function SizeAndShapeMCMC(;
#    dataset_init::Array{Float64,3}, 
#    fm::FormulaTerm = @formula(1~ 1),
#    covariates::DataFrame, #
#    iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}} =(
#        iter=1000,
#        burnin=200,
#        thin=2
#    ),
#    betaprior::ContinuousUnivariateDistribution = Normal(0.0, 10000.0),
#    sigmaprior::ContinuousMatrixDistribution,
#    beta_init::Matrix{Float64},
#    sigma_init::Symmetric{Float64,Matrix{Float64}},
#    rmat_init::Array{Float64,3},
#    reflection::Bool = true,
#    is_data_sizeandshape::Bool = true
#)

#    p::Int64 = size(dataset, 2)
#    if p != 2
#        error("the current implementation allows only `size(dataset, 2) = 2 (2-dimensional data)`")
#    end
#    if reflection == false
#        error("the current implementation allows only `reflection = true`")
#    else
#         mcmcout =generalSizeAndShapeMCMC(; 
#            dataset = dataset,
#            fm = fm,
#            covariates = covariates,
#            iterations = iterations,
#            betaprior = betaprior,
#            sigmaprior = sigmaprior,
#            beta_init = beta_init,
#            sigma_init = sigma_init,
#            rmat_init = rmat_init,
#            dormat = true,
#            reflection = KeepReflection(),
#            sigmatype = GeneralSigma()
#        )
        
#        return mcmcout.mcmcoutputArrays.beta, mcmcout.mcmcoutputArrays.sigma
#    end
    

#end