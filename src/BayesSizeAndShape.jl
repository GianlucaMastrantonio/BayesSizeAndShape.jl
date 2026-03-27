module BayesSizeAndShape
 
##### Packages
using Distributions, Random
using LinearAlgebra, PDMats
using ProgressMeter, Revise
using Kronecker
using DataFrames
using CategoricalArrays
using StatsModels
using ToggleableAsserts
using CodecBzip2
using Reexport, RData

##### Include
include(joinpath("types.jl"))
include(joinpath("general_functions.jl"))
include(joinpath("mcmc.jl"))
include(joinpath("sampler_mean.jl"))
include(joinpath("sampler_covariance.jl"))
include(joinpath("sampler_latentobservations.jl"))
include(joinpath("modeloutput.jl"))
#include(joinpath("sampler.jl"))
include(joinpath("wrappers.jl"))
#include(joinpath("deprecated.jl"))
include(joinpath("dataset.jl"))
include(joinpath("prediction.jl"))

include(joinpath("external_function.jl"))


#include(joinpath("old files/sim.jl"))


##### Dataset
#rats = load(joinpath(@__DIR__,"data/rats.jld"))["data"]
#rats_cov = load(joinpath(@__DIR__,"data/rats.jld"))["cov"]
#rats_cov = load(joinpath(@__DIR__,"data/rats_cov.jld"))["data"]

##### Functions
 export
    ### Dataset
    dataset,
    dataset_desciption,
    dataset_names,
    ### Models
    generalSizeAndShapeMCMC,
    SizeAndShapeWithReflectionMCMC,
    sample_predictive_zbr,
    sample_predictive_zbr_plus_epsilon,
    ### TIPES
    KeepReflection,
    RemoveLocationHelmert,
    RemoveLocationCentering,
    ValueP2,
    ValueP3,
    DoNotRemoveSize,
    GramSchmidtMean,
    MCMCNormalDataKeepSize,
    MCMCNormalDataKeepSize,
    LinearMean,
    MCMCLinearMean,
    GeneralCoVarianceIndependentDimension,
    generalMCMCObjectOUT,
    MCMCGeneralCoVarianceIndependentDimension,
    SizeAndShapeModelOutput,
    ### EXTERNAL
    sizeshape_helmertproduct_reflection,
    posterior_samples_beta,
    posterior_samples_sigma,
    sizeshape_helmertproduct_reflection_user,
    sample_predictive_zb,
    standardize_reg_gimmi,
    standardize_reg_computegamma_gimmi

    
    #NoPriorBeta,
    #NoPriorSigma,
    #compute_ss_from_pre,
    #compute_ss!,
    #compute_ss,
    #KeepReflection,
    #DoKeepReflection,
    #GeneralSigma,
    #standardize_reg,
    #SizeAndShapeMCMC,
    #Valuep2,
    #compute_designmatrix,
    #compute_dessignmatrix,
    #compute_helmertized_configuration,
    #SizeAndShapeModelOutput,
    #predictmean,
    
    #dataset_names,
    #create_designmatrix,
    #generalSizeAndShapeMCMC
    

end 
