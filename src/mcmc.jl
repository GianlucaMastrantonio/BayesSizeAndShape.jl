
function generalSizeAndShapeMCMC(;
    landmarks::Array{Float64,3}, 
    fm::FormulaTerm = @formula(1~ 1),
    covariates::DataFrame, #
    iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}} =(
        iter=1000,
        burnin=200,
        thin=2
    ),
    
    betaprior::ContinuousUnivariateDistribution = Normal(0.0, 10000.0),
    sigmaprior::ContinuousMatrixDistribution,
    beta_init::Matrix{Float64} = zeros(Float64,0,0),
    sigma_init::Symmetric{Float64,Matrix{Float64}} = Symmetric(zeros(Float64,0,0)),
    rmat_init::Array{Float64,3} = zeros(Float64,0,0,0),
    #dormat::Bool,
    identification::String = ["gramschmidt"][1],
    meanmodel::String = ["linear"][1],
    covariancemodel::String = ["general_nocrosscorrelation"][1],
    keepreflection::String = ["no", "yes"][2],
    removelocation::String = ["no", "helmert", "centering"][3],
    removesize::String = ["no", "norm"][2],
    rmatdosample::Bool = true,
    verbose::Bool = true
)

    ##### dimensions
    k::Int64 = size(landmarks, 1)-1
    kland::Int64 = size(landmarks, 1)
    p::Int64 = size(landmarks, 2)
    n::Int64 = size(landmarks, 3)
    pangle::Int64 = ifelse(p==2,1,3)

    ##### Types
    reflectioninformation::Reflection = KeepReflection();
    if keepreflection == "yes"  reflectioninformation = KeepReflection()
    elseif keepreflection == "no" reflectioninformation = DoNotKeepReflection(); error("keepreflection == \"no\" not implemented")
    else error("keepreflection should be in [\"no\", \"yes\"]")
    end

    valp::ValueP = ValueP2();
    if p == 2 valp = ValueP2();
    elseif p == 3 valp = ValueP3(); #error("Models with dimension p>2 are not implemented")
    else error("Models with dimension p>2 are not implemented")
    end

    locationinformation::RemoveLocation = RemoveLocationHelmert(k, valp);
    if removelocation == "helmert" locationinformation = RemoveLocationHelmert(k, valp);
    elseif removelocation == "centering" locationinformation = RemoveLocationCentering(k, valp); #add also the centering possibility
    elseif removelocation == "no" locationinformation = DoNotRemoveLocation(kland, valp); error("removelocation == \"no\" not implemented")
    else error("removelocation should be in [\"no\", \"helmert\"]")
    end
 
    sizeinformation::RemoveSize = DoNotRemoveSize();
    if removesize == "no" sizeinformation = DoNotRemoveSize();
    elseif removesize == "norm" sizeinformation = RemoveSizeNorm(); error("removesize \"norm\" not implemented")
    else error("removesize should be in [\"no\", \"norm\"]")
    end

    
    ##### identifiability #####
    identifiability_constraint::IdentifiabilityConstraint = GramSchmidtMean()
    if identification == "gramschmidt" identifiability_constraint = GramSchmidtMean()
    else error("identification should be in [\"gramschmidt\"]")
    end

    if (identification == "gramschmidt") & (meanmodel != "linear") 
        error("identification \"gramschmidt\" can only be used with meanmodel \"linear\"")
    end

    ##### DATASET #####
    
    if rmatdosample == true
        rmatsample = DoSampleRmat()
    else
        rmatsample = DoNotSampleRmat()
    end
    
    datamodel = SSDataType(landmarks,  reflectioninformation,locationinformation,valp,sizeinformation,identifiability_constraint);
    data_mcmc = MCMCdata(rmat_init,  datamodel,rmatsample; sdprop_adapt_init = 0.1, accratio_adapt_init = 0.234, molt_adapt_init = 0.4, iter_adapt_init = 50, init_adapt_init= 100, end_adapt_init =   Int64(iterations.burnin*0.9),  a_adapt_init = 100.0, b_adapt_init = 200.0)

    

    ###### mean #####
    mean_model::TypeModelMean = LinearMean(fm, covariates, datamodel, identifiability_constraint)
    if meanmodel == "linear" mean_model = LinearMean(fm, covariates, datamodel, identifiability_constraint)
    else error("meanmodel should be in [\"linear\"]")
    end
    

    mean_mcmc::MCMCTypeModelMean = MCMCMean(mean_model, valp, beta_init,betaprior,datamodel) 
 
    ###### covariance #####
    covariance_model::TypeModelCoVariance = GeneralCoVarianceIndependentDimension(identifiability_constraint,datamodel);
    if covariancemodel == "general_nocrosscorrelation" covariance_model = GeneralCoVarianceIndependentDimension(identifiability_constraint,datamodel);
    else error("covariancemodel should be in [\"general_nocrosscorrelation\"]")
    end

    covariance_mcmc = MCMCCovariance(covariance_model, sigma_init, sigmaprior, datamodel)
    
    ##### asserts
    @assert size(datamodel.nolocdata, 3) == size(covariates,1) # n
    @assert size(datamodel.nolocdata, 3) == size(mean_model.model_matrix,1) # n
    @assert size(mean_model.designmatrix, 1) == size(datamodel.nolocdata, 1) # k
    @assert size(mean_model.designmatrix, 3) == size(datamodel.nolocdata, 3) # n
    @assert size(mean_model.designmatrix, 2) == size(datamodel.nolocdata, 1) * size(mean_model.model_matrix,2) #k d


    ##### print messages #####
    if verbose
        if (removelocation != "no") & (removesize == "no")
            print("\n\n")
            print("Size And Shape Model")
        end
        if keepreflection == "yes"
            print(" with reflection information \n")
        end

        println("\nThe data has ", 
            kland," ",  
            p,"-dimensional landmarks in ", 
            n,  " shapes",
        )
    
        if true == true
            println("The mean is modelled with a linear function and it has ", mean_model.d, " regressive coefficients for each dimension*(landmark-1), with a total of ", size(mean_model.designmatrix,2)*datamodel.p, " regressors")
        end
        if true == true
            println("The covariance is unstructured and shared between dimensions, with no cross-correlation")
        end

        if typeof(identifiability_constraint) <:GramSchmidtMean
            println("\nFor identifiability, the regressive coefficients are trasformed using a Gram-Schmidt transformation\n")
        end
        
        
    end
    
    ###### MCMC object #####
    iter = Int64(iterations.iter)
    burnin = Int64(iterations.burnin)
    thin = Int64(iterations.thin)
    sampletosave = trunc(Int64, round((iter - burnin) / thin))

    ##### #### #### #### #### 
    ##### MCMC out
    ##### #### #### #### #### 

    posteriorparamters = create_object_output(sampletosave, mean_mcmc, covariance_mcmc, data_mcmc, mean_model)

    ##### #### #### #### #### 
    ##### algrothm
    ##### #### #### #### #### 
    iterMCMC = Int64(0)
    thinburnin = burnin
    p1 = Progress(burnin, desc = "burnin ", offset = 0, showspeed = true)
    p2 = Progress(
        burnin + (sampletosave - 1) * thin,
        desc = "iterations ",
        offset = 0,
        showspeed = true,
    )
    isburn = true
    
    println("MCMC settings ")
    println("Iterations: ", iter)
    println("Burnin: ", burnin)
    println("Thin: ", thin)
    println("Number of posterior samples: ", sampletosave)
    println("Number of threads: ", Threads.nthreads())
    for iMCMC = 1:sampletosave

        for jMCMC = 1:thinburnin

            iterMCMC += 1
            

            sampler_mean(datamodel, data_mcmc,mean_model,mean_mcmc,covariance_model,covariance_mcmc)
            sampler_covariance(datamodel, data_mcmc,mean_model,mean_mcmc,covariance_model,covariance_mcmc)
            sampler_latentobservations(iterMCMC, datamodel, data_mcmc,mean_model,mean_mcmc,covariance_model,covariance_mcmc)
            
            
            ProgressMeter.next!(p2; showvalues = [(:iterations, iterMCMC)])

        end

        thinburnin = thin
        isburn = false

        copy_parameters_out(iMCMC, posteriorparamters, mean_mcmc, datamodel, covariance_mcmc, data_mcmc) 

        



    end

    
    return SizeAndShapeModelOutput(
            datamodel, 
            data_mcmc,
            mean_model,
            mean_mcmc,
            covariance_model,
            covariance_mcmc,
            posteriorparamters,
            (iter = iter, burnin = burnin, thin = thin, savedsamples =sampletosave)
            )
    ##### #### #### #### 
    ##### OUTPUT
    ##### #### #### #### 
    #nsim = size(betaOUT,1)

    



    #mcmcoutputSAVE = (beta = bbb, sigma = sss, rmat = rrr, angle = ttt)
    #mcmcoutputArraysSAVE = (beta = betaOUT, sigma = sigmaOUT, rmat = rmatOUT, angle = angleOUT)
    #covariatesSAVE = (colnames_modelmatrix = colnames_modelmatrix, fm = deepcopy(fm), covariates = deepcopy(covariates), 
    #                    designmatrix_step1 = designmatrix_v2, designmatrix_step2 = designmatrix)
    #datasetSAVE = deepcopy(dataset)
    #modeltypesSAVE = (dormat=dormat, reflection = reflection, sigmatype=sigmatype, betaprior = betaprior, sigmaprior = sigmaprior, valp = valp);
    #iterationsSAVE = (iter = iter, burnin = burnin, thin = thin, savedsamples =sampletosave);
    #indicesSAVE = (k=k, p=p, n=n, d=d);

    #return SizeAndShapeModelOutput(mcmcoutputSAVE , mcmcoutputArraysSAVE , covariatesSAVE , datasetSAVE , modeltypesSAVE , iterationsSAVE , indicesSAVE )
    

end
