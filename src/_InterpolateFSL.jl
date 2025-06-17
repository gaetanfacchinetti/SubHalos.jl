##################################################################################
# This file is part of SubHalos.jl
#
# Copyright (c) 2024, Gaétan Facchinetti
#
# SubHalos.jl is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. SubHalos.jl is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################

export TidalScaleNNInterpolator, make_nn_data, read_nn_data, forward, train!

struct MetaDataInterpolator
    hash::String
    θ_min::Vector{Float32}
    θ_max::Vector{Float32}
end

struct DataInterpolator{T<:AbstractFloat}
    
    θ_train::Matrix{T}
    x_train::Matrix{T}
    y_train::Vector{T}

    θ_test::Matrix{T}
    x_test::Matrix{T}
    y_test::Vector{T}

    θ_valid::Matrix{T}
    x_valid::Matrix{T}
    y_valid::Vector{T}
    
end

abstract type NNInterpolator{C<:Flux.Chain} end

struct TidalScaleNNInterpolator{C<:Flux.Chain} <: NNInterpolator{C}
    nn::C
    n_hidden_layers::Int
    n_hidden_features::Int
    train_loss::Vector{Float32}
    valid_loss::Vector{Float32}
    metadata::MetaDataInterpolator
end

function TidalScaleNNInterpolator(
    metadata::MetaDataInterpolator, 
    n_hidden_layers::Int = 2, 
    n_hidden_features::Int = 64, 
    train_loss::Vector{Float32} = Float32[], 
    valid_loss::Vector{Float32} = Float32[]) 
    
    layers = (
        [Flux.Dense(5, n_hidden_features, Flux.relu)],
        [Flux.Dense(n_hidden_features, n_hidden_features, Flux.relu) for _ in 1:n_hidden_layers],
        [Flux.Dense(n_hidden_features, 1)]
    )

    chain = Flux.Chain(vcat(layers...)...)
    return TidalScaleNNInterpolator{typeof(chain)}(chain, n_hidden_layers, n_hidden_features,  train_loss, valid_loss, metadata)

end

# Normalize a batch of inputs (shape: 5×N)
normalize_input(θ::Union{Matrix{Float32}, Vector{Float32}}, θ_min::Vector{Float32}, θ_max::Vector{Float32}) = (θ .- θ_min) ./ (θ_max .- θ_min)


# predictions of the model / forward pass
function (f::TidalScaleNNInterpolator)(r_host::T, c::T, m::T, θ::T, q::T) where {T<:AbstractFloat}
    x = Float32[log10(r_host), log10(c), log10(m), θ, log10(q)]
    return T(exp10(max.(f.nn(normalize_input(x, f.metadata.θ_min, f.metadata.θ_max))[1], -15f0)))
end

forward(x::Matrix{Float32}, nn::Flux.Chain, ::Type{T}) where {T<:TidalScaleNNInterpolator} = max.(nn(x), -15f0)
forward(x::Matrix{Float32}, model::T) where {T<:NNInterpolator} = forward(x, model.nn, T)

# loss function
loss(y_pred::Vector{Float32}, y_true::Vector{Float32}, ::Type{T}) where {T<:TidalScaleNNInterpolator} = Statistics.mean(abs.( y_pred .- y_true ) )


function read_nn_data(
    host::HostModel{T, U}, 
    shp::HaloProfile{S}, 
    cosmo::Cosmology{T, BKG};
    z::T = T(0.0),
    Δ::T = T(200.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}

    θ_train = x_train = y_train = nothing
    θ_test = x_test = y_test = nothing
    θ_valid = x_valid = y_valid = nothing
    θ_min = θ_max = nothing

    hash_str = get_hash_nn(host, shp, cosmo, z, Δ)

    JLD2.jldopen(".cache/database_tidal_striping_" * hash_str * ".jld2" ) do data
        
        θ_train = data["θ_train"]
        x_train = data["x_train"]
        y_train = data["y_train"]

        θ_test = data["θ_test"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        θ_valid = data["θ_valid"]
        x_valid = data["x_valid"]
        y_valid = data["y_valid"]

        θ_min = data["θ_min"]
        θ_max = data["θ_max"]
        
    end

    return DataInterpolator(θ_train, x_train, y_train, θ_test, x_test, y_test, θ_valid, x_valid, y_valid), MetaDataInterpolator(hash_str, θ_min, θ_max)

end 

function train!(
    model::TidalScaleNNInterpolator,
    data::DataInterpolator, 
    optimiser::Any; 
    epochs::Int=100,
    verbose::Bool=true)

    train_loader = Flux.DataLoader((data.x_train, data.y_train); batchsize=64, shuffle=true)
    T = typeof(model)

    for epoch in 1:epochs

        ## Traning part

        for (x_batch, y_batch) in train_loader
                        
            grads = Flux.gradient(model.nn) do m
            
                y_pred = forward(x_batch, m, T)[1, :]
                _loss   = loss(y_pred, y_batch, T)
                                
                return _loss
                
            end

            Optimisers.update!(optimiser, model.nn, grads[1])
            
        end

        ## Losses part for validation of the model
             
        y_pred_train = forward(data.x_train, model)[1, :]
        push!(model.train_loss, loss(y_pred_train, data.y_train, T))

        y_pred_valid = forward(data.x_valid, model)[1, :]
        push!(model.valid_loss, loss(y_pred_valid, data.y_valid, T))

        verbose && println("epoch: $epoch, train loss : $(model.train_loss[end]), valid loss : $(model.valid_loss[end])")

    end

    y_pred_test = forward(data.x_test, model)[1, :]
    return y_pred_test .- data.y_test # returns the absolute error for points in the test dataset

end

function save(model::TidalScaleNNInterpolator)
    
    model_state = Flux.state(model.nn)

    JLD2.jldsave("./.cache/tidal_scale_interpolator_" * model.metadata.hash * ".jld2"; 
        model_state = model_state, 
        θ_min = model.metadata.θ_min, 
        θ_max = model.metadata.θ_max,
        n_hidden_features = model.n_hidden_features,
        n_hidden_layers = model.n_hidden_layers,
        train_loss = model.train_loss,
        valid_loss = model.valid_loss
    )
end

function load_tidal_scale_interpolator(
    host::HostModel{T, U}, 
    shp::HaloProfile{S}, 
    cosmo::Cosmology{T, BKG}, 
    z::T = T(0.0),
    Δ::T = T(200.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}

    hash_str = get_hash_nn(host, shp, cosmo, z, Δ)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = "tidal_scale_interpolator_" * hash_str * ".jld2"


    if !(file in filenames) 
        @warn "no available cached interpolator for this model"
        return nothing
    end

    model = nothing

    JLD2.jldopen(cache_location * file, "r") do data
        
        model_state = data["model_state"] 
        
        θ_min = data["θ_min"]
        θ_max = data["θ_max"]
        n_hidden_features = data["n_hidden_features"]
        n_hidden_layers = data["n_hidden_layers"]
        train_loss = data["train_loss"]
        valid_loss = data["valid_loss"]

        metadata = MetaDataInterpolator(hash_str, θ_min, θ_max)
        model    = TidalScaleNNInterpolator(metadata, n_hidden_layers, n_hidden_features, train_loss, valid_loss)

        Flux.loadmodel!(model.nn, model_state)
    end


    return model
    
end

function get_hash_nn(
    host::HostModel{T, U}, 
    shp::HaloProfile{S}, 
    cosmo::Cosmology{T, BKG}, 
    z::T = T(0.0),
    Δ::T = T(200.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}
    
    hash_tuple = (host.name, cosmo.name, shp.name, z, Δ)

    return string(hash(hash_tuple), base=16)
end 


""" efficient evaluation of the tidal scale """
function tidal_scale(
    host::HostModel{T, U}, 
    pp::ProfileProperties{T, S}, 
    cosmo::Cosmology{T, BKG}, 
    x::Vector{Float32};
    z::T = T(0.0),
    Δ::T = T(200.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}
    
    # order of the input parameters
    # log10_r::T, log10_c::T, log10_m::T, θ::T, log10_q::T
    
    y = T.(x)
    
    subhalo = halo_from_mΔ_and_cΔ(nfwProfile, exp10(y[3]), exp10(y[2]), Δ=Δ, ρ_ref = cosmo.bkg.ρ_c0)
    res = tidal_scale(exp10(y[1]), subhalo, host, z, Δ, cosmo; pp = pp, q = exp10(y[5]), θ=y[4], disk=true, stars=true, use_tables=true)
    
    return res == 0 ? -15f0 : Float32.(log10(res))
end


function evaluate(
    x::Matrix{Float32}, 
    host::HostModel{T, U}, 
    pp::ProfileProperties{T, S}, 
    cosmo::Cosmology{T, BKG};
    z::T = T(0.0),
    Δ::T = T(200.0)
    )  where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}
        
    N = size(x, 2)
    res = Vector{Float32}(undef, N)
    has_done = [0 for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:N
        thread_id = Threads.threadid()

        res[i] = tidal_scale(host, pp, cosmo, x[:, i], z = z, Δ = Δ)

        has_done[thread_id] += 1
        if has_done[thread_id] % 100 == 0
            println("Thread $thread_id has completed $(has_done[thread_id]) evaluations")
        end
    end

    return res

end 


# Generate `n` random inputs of shape (5, n)
random_inputs(n::Int, θ_min::Vector{Float32}, θ_max::Vector{Float32}) = θ_min .+ rand(Float32, 5, n) .* (θ_max .- θ_min)



function make_nn_data(
    host::HostModel{T, U}, 
    shp::HaloProfile{S}, 
    cosmo::Cosmology{T, BKG};
    z::T = T(0.0),
    Δ::T = T(200.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, BKG<:BkgCosmology{T}}

    pp = ProfileProperties(shp, T)

    load!(host)
    load!(pp)

    θ_min::Vector{Float32} = [log10(1f-2 * Float32(host.halo.rs)), 0.0f0, -15.0f0, Float32(π/8.0), -2.0f0]
    θ_max::Vector{Float32} = [log10(Float32(host.rt)), Float32(log10(500.0)), +15.0f0, Float32(3*π/8.0), log10(0.2f0)]

    # Generating the data
    θ_train = random_inputs(60000, θ_min, θ_max)
    x_train = normalize_input(θ_train, θ_min, θ_max)
    y_train = evaluate(θ_train, host, pp, cosmo, z = z, Δ = Δ)

    θ_test = random_inputs(6000, θ_min, θ_max)
    x_test = normalize_input(θ_test, θ_min, θ_max)
    y_test = evaluate(θ_test, host, pp, cosmo, z = z, Δ = Δ)

    θ_valid = random_inputs(6000, θ_min, θ_max)
    x_valid = normalize_input(θ_valid, θ_min, θ_max)
    y_valid = evaluate(θ_valid, host, pp, cosmo, z = z, Δ = Δ)


    JLD2.jldsave(".cache/database_tidal_striping_" * get_hash_nn(host, shp, cosmo, z, Δ) * ".jld2" ; 
        θ_train = θ_train, x_train = x_train, y_train = y_train,
        θ_test = θ_test, x_test = x_test, y_test = y_test,
        θ_valid = θ_valid, x_valid = x_valid, y_valid = y_valid,
        θ_min = θ_min, θ_max = θ_max
    )
end


