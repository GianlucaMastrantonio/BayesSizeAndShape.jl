function standardize_reg(reg::Matrix{Float64}, valp::ValueP2, indentifiability::GramSchmidtMean)

    reg[:,:] = reg * standardize_reg_computegamma(reg, valp,indentifiability)

end

function standardize_reg_computegamma(reg::Matrix{Float64}, valp::ValueP2, indentifiability::GramSchmidtMean)

    a1 = reg[1, :]
    a2 = reg[2, :]
    g1 = a1 / norm(a1, 2)
    g2 = (a2 - (transpose(a2) * g1) * g1) / norm(a2 - (transpose(a2) * g1) * g1, 2)
    gammamat = reshape([g1; g2], (2, 2))

    if det(gammamat) < 0
        gammamat[:, 2] = - gammamat[:, 2]
    end

    return gammamat
end

function standardize_reg(reg::Matrix{Float64}, valp::ValueP3, indentifiability::GramSchmidtMean)

    reg[:,:] = reg * standardize_reg_computegamma(reg, valp,indentifiability)

end

function standardize_reg_computegamma(reg::Matrix{Float64}, valp::ValueP3, indentifiability::GramSchmidtMean)

    a1 = reg[1, :]
    a2 = reg[2, :]
    a3 = reg[3, :]
    g1 = a1 / norm(a1, 2)
    g2 = (a2 - (transpose(a2) * g1) * g1) / norm(a2 - (transpose(a2) * g1) * g1, 2)
    g3 = (a3 - (transpose(a3) * g1) * g1  -  (transpose(a3) * g2) * g2) / norm(a3 - (transpose(a3) * g1) * g1  -  (transpose(a3) * g2) * g2, 2)

    gammamat = reshape([g1; g2; g3], (3, 3))

    if det(gammamat) < 0
        gammamat[:, 2] = - gammamat[:, 2]
    end

    return gammamat

end

#function standardize_reg(reg::Matrix{Float64}, valp::ValueP2)

#    reg[:,:] = reg * standardize_reg_computegamma(reg, valp)

#end

#function standardize_reg_computegamma(reg::Matrix{Float64}, valp::ValueP2)

#    a1 = reg[1, :]
#    a2 = reg[2, :]
#    g1 = a1 / norm(a1, 2)
#    g2 = (a2 - (transpose(a2) * g1) * g1) / norm(a2 - (transpose(a2) * g1) * g1, 2)
#    gammamat = reshape([g1; g2], (2, 2))

#    if det(gammamat) < 0
#        gammamat[:, 2] = - gammamat[:, 2]
#    end

#    return gammamat
#end


function compute_designmatrix(zmat::Matrix{Float64}, k::Int64)::Array{Float64,3}

    n, d = size(zmat)
    res::Array{Float64,3} = zeros(Float64, k, k * d, n)
    for i = 1:n
        res[:, :, i] = kronecker(transpose(zmat[i, :]), Matrix{Float64}(I, k, k))
    end

    return res

end

function compute_designmatrix(zmat::DataFrame, k::Int64)::Array{Float64,3}

    n, d = size(zmat)
    res::Array{Float64,3} = zeros(Float64, k, k * d, n)
    for i = 1:n
        res[:, :, i] = kronecker(transpose([values(zmat[i, :])...]), Matrix{Float64}(I, k, k))
    end

    return res

end

function remove_location(landmarks::Array{Float64,3}, removelocation::DoRemoveLocation)::Array{Float64,3} 
    
    
    H = removelocation.matrix
    ret = deepcopy(landmarks)
    ret = ret[2:end,:,:]
    for i = 1:size(landmarks,3)
        ret[:,:,i] = H*landmarks[:,:,i]
    end

    return ret

end

#function compute_sizeshape_fromnolocdata(landmarks::Array{Float64,3}, keepreflection::KeepReflection, valp::ValueP2)::Tuple{Array{Float64, 3}, Array{Float64, 3}}

#    dataret = deepcopy(landmarks)
#    rotret = Array{Float64}(undef,2,2,size(landmarks,3))
#    for i = axes(landmarks, 3)
    
#        app = svd(landmarks[:, :, i])
#        U = app.U
#        V = transpose(app.Vt)
#        if sign(V[1, 2]) != sign(V[2, 1])

#            dataret[:, :, i] = U * Diagonal(app.S)
#            rotret[:, :, i] = V


#        else

            
            
#            U[:,2] = -1.0*U[:,2]
#            V[:, 2] = -1.0 * V[:, 2]


#            dataret[:, :, i] = U * Diagonal(app.S)
#            rotret[:, :, i] = V
            

#        end

#    end
#    #println(size(landmarks))
#    #println(size(dataret))
#    return dataret, rotret

#end


function compute_sizeshape_fromnolocdata(landmarks::Array{Float64,3}, keepreflection::KeepReflection, valp::P)::Tuple{Array{Float64, 3}, Array{Float64, 3}} where {P<:ValueP}

    
    p::Int64 = size(landmarks,2)

    dataret = deepcopy(landmarks)
    rotret = Array{Float64}(undef,p,p,size(landmarks,3))
    for i = axes(landmarks, 3)
    
        app = svd(landmarks[:, :, i])
        U = app.U
        V = transpose(app.Vt)
        if det(V) > 0.0

            dataret[:, :, i] = U * Diagonal(app.S)
            rotret[:, :, i] = V


        else

            
            
            U[:,2] = -1.0*U[:,2]
            V[:, 2] = -1.0 * V[:, 2]


            dataret[:, :, i] = U * Diagonal(app.S)
            rotret[:, :, i] = V
            

        end

    end

    return dataret, rotret

end


function create_designmatrix(fm::FormulaTerm, covariates::DataFrame, k::Int64 )::Tuple{Array{Float64, 3}, Int64, Vector{String}, Matrix{Float64}}
    

  

    #println(covariates[1:3,:])
    #error("")
    covariates_copy = deepcopy(covariates)
    for i = 1:size(covariates_copy,2)
        if  isa(covariates_copy[:,i], CategoricalArray)

        elseif isa(covariates_copy[1,i], Real)
            #covariates_copy[:,i] = (covariates_copy[:,i] .- mean(covariates_copy[:,i])) ./ std(covariates_copy[:,i])
        else
        
            error("Only factors or Real variables are allowed in covariates")
        end
    end

    designmatrix_v2_app = ModelFrame(fm, covariates_copy);
    designmatrix_v2 = ModelMatrix(designmatrix_v2_app).m
    #println(designmatrix_v2[1:3,:])
    #error("")


    designmatrix = compute_designmatrix(designmatrix_v2, k) # dimensions  k, k * d, n

    @assert sum(designmatrix_v2[:, 1] .== 1) == size(designmatrix_v2,1) "intercept needed"
    
    return designmatrix, size(designmatrix_v2,2), coefnames(designmatrix_v2_app), designmatrix_v2 

end


function compute_yrdata(yrdata::Array{Float64,3}, ssdata::Array{Float64,3}, rmat::Array{Float64,3})

    for i = axes(ssdata,3)
        yrdata[:, :, i] = ssdata[:, :, i] * transpose(rmat[:, :, i])
    end

    return nothing

end

function compute_angle_from_rmat(i::Int64,angle::Matrix{Float64},  rmat::Array{Float64,3}, valp::ValueP2, reflection::KeepReflection)

    angle[1, i] = atan(rmat[2, 1, i], rmat[1, 1, i])

    return nothing
end

function compute_angle_from_rmat(i::Int64,angle::Matrix{Float64},  rmat::Array{Float64,3}, valp::ValueP3, reflection::KeepReflection)

    #### http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf ###
    angle[2,i] = -asin(rmat[3,1,i])
    angle[3,i] = atan( rmat[3,2,i]/cos(angle[2,i]), rmat[3,3,i]/cos(angle[2,i]))
    angle[1,i] = atan( rmat[2,1,i]/cos(angle[2,i]), rmat[1,1,i]/cos(angle[2,i]))

    return nothing
end


#function compute_angle_from_rmat(i::Int64, angle::Matrix{Float64}, rmat::Array{Float64,3}, valp::Valuep3, reflection::KeepReflection)
#    #x convention
#    angle[1, i] = atan(rmat[3, 1, i], rmat[3, 2, i])
#    angle[2, i] = acos(rmat[3, 3, i])
#    angle[3, i] = -atan(rmat[1, 3, i], rmat[2, 3, i])

#    return nothing
#end
