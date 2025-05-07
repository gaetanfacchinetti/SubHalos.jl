##################################################################################
# This file is part of SubHalos.jl
#
# Copyright (c) 2024, Gaétan Facchinetti
#
# SubHalos.jl is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. CosmoTools.jl is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################

export get_hash, clean_cache!, make_cache!, load!, reset!

const _NPTS_R = 50
const _NPTS_M = 50
const _NPTS_C = 50


# redefinition of getproperty to instantiate tables
function Base.getproperty(obj::FSLModel, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load!(obj, s))
    end

    return getfield(obj, s)
end


##################################################################################
# Caching tidal stripping results

function _save_tidal_scale(model::FSLModel)
    
    rs = model.context.host.halo.rs
    rt = model.context.host.rt
    
    @info "| Saving tidal_scale in cache" 
    r = 10.0.^range(log10(1e-2 * rs), log10(rt), _NPTS_R)
    c = 10.0.^range(0, log10(model.options.c_max), _NPTS_C)
    m = nothing

    # Loading all the tabulated values for the host in order  
    # to avoid clashes between different threads in the following
    load!(model.context.host)
    load!(model.context.pp)

    if !(model.options.stars)

        # here we do not have any dependance on the mass
        # skip saving the table in terms of the mass for efficiency
        # moreover use parallelisation as every point is independant

        y = Array{Float64}(undef, _NPTS_R, _NPTS_C)
        Threads.@threads for ic in 1:_NPTS_C 
            for ir in 1:_NPTS_R 
                y[ir, ic] = tidal_scale(r[ir], c[ic], 1.0, model)
            end
        end
    
    else

        m = 10.0.^range(-15, 15, _NPTS_M)
        y = Array{Float64}(undef, _NPTS_R, _NPTS_C, _NPTS_M)

        Threads.@threads for im in 1:_NPTS_M
            for ic in 1:_NPTS_C
                for ir in 1:_NPTS_R
                    y[ir, ic, im] = tidal_scale(r[ir], c[ic], m[im], model)
                end
            end
        end
    end

    JLD2.jldsave(cache_location * "tidal_scale_" * get_hash(model, :tidal_scale) * ".jld2" ; r = r, m = m, c = c, y = y)
    return true
end


## Possibility to interpolate the model
function _load_tidal_scale(model::FSLModel, filepath::String)

    data    = JLD2.jldopen(filepath)

    _r = data["r"]
    _m = data["m"]
    _c = data["c"]
    _y = data["y"]

    if !(model.options.stars)
        log10_y = Interpolations.interpolate((log10.(_r), log10.(_c)), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))
        
        return (r::Real, c::Real, m::Real = 1.0) -> 
        begin
            try
                res = 10.0^log10_y(log10(r), log10(c))
                (res === NaN) && return -Inf
                return res 
            catch e
                isa(e, BoundsError) && (return tidal_scale(r, c, m, model))
                rethrow()
            end 
        end

    else
        log10_y = Interpolations.interpolate((log10.(_r), log10.(_c), log10.(_m)), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))
        
        return (r::Real, c::Real, m::Real) -> 
        begin
            try
                res = 10.0^log10_y(log10(r), log10(c), log10(m))
                (res === NaN) && return -Inf
                return res 
            catch e
                isa(e, BoundsError) && (return tidal_scale(r, c, m, model))
                rethrow()
            end
        end
    end

end


function clean_cache!(model::FSLModel, clean_tidal_scale::Bool = false)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)

    ps = [:min_concentration, :min_concentration_calibration, :min_concentration_mt]
    clean_tidal_scale && append!(ps, [:tidal_scale])


    for p in ps 
        file = string(p) * "_" * get_hash(model, p) * ".jld2" 
        !(file in filenames) && (@info "No saved function " * string(p) * " for model" * string(model))
        (file in filenames) && rm(cache_location * file)
        (file in filenames) && setproperty!(model, p, nothing) # reinitialise to nothing
    end
    
end


##################################################################################
# Caching intermediate results

function make_cache!(model::FSLModel, s::Symbol)

    (s === :tidal_scale) && (return _save_tidal_scale(model))
    
    rs = model.context.host.halo.rs
    rt = model.context.host.rt
    
    @info "| Saving " * string(s) * " in cache" 
    r = 10.0.^range(log10(1e-2 * rs), log10(rt), _NPTS_R)
    m = nothing

    if (!(model.options.stars) || (s === :min_concentration_calibration)) && (s !== :min_concentration_mt)
        # here we do not have any dependance on the mass
        # skip saving the table in terms of the mass for efficiency
        y = @eval $s.($(Ref(r))[], 1.0, $(Ref(model))[])
    else
        m = 10.0.^range(-15, 15, _NPTS_M)
        y = @eval $s.($(Ref(r))[], $(Ref(m))[]', $(Ref(model))[])
    end

    JLD2.jldsave(cache_location * string(s)  * "_" *  get_hash(model, s) * ".jld2" ; r = r, m = m, y = y)

    return true

end

""" create a unique hash combination in terms of the parameters the cached function (passed as symbol s) depends on """
function get_hash(model::FSLModel, s::Symbol)

    if (s === :tidal_scale)
        hash_tuple = (model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z, model.options.disk, model.options.stars)
    elseif (s === :min_concentration_calibration)
        hash_tuple = (model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z)
    else
        hash_tuple = (model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.params.ϵ_t, model.options.c_max, model.params.z, model.options.disk, model.options.stars)
    end

    (model.options.stars) && (hash_tuple = (hash_tuple..., (model.context.q, model.context.θ)...))

    return string(hash(hash_tuple), base=16)
end 


## Possibility to interpolate the model
function _load!(model::FSLModel, s::Symbol)
    
    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * "_" * get_hash(model, s) * ".jld2" 

    !(file in filenames) && make_cache!(model, s)
    (s === :tidal_scale) && (return _load_tidal_scale(model, cache_location * file))

    data    = JLD2.jldopen(cache_location * file)

    r = data["r"]
    m = data["m"]
    y = data["y"]

    if (!(model.options.stars) || (s === :min_concentration_calibration)) && (s !== :min_concentration_mt)
        log10_y = Interpolations.interpolate((log10.(r),), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
        return (r::Real, m::Real = 1.0) -> 10.0^log10_y(log10(r))
    else
        log10_y = Interpolations.interpolate((log10.(r), log10.(m)), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
        return (r::Real, m::Real) -> 10.0^log10_y(log10(r), log10(m))
    end

end


function load!(model::FSLModel)
    
    for field in fieldnames(FSLModel) 
        (getfield(model, field) === nothing) && setproperty!(model, field, _load!(model, field))
    end

    return nothing
end


""" make the cache functions / compare to load, even if cache already exist, remake it """
function make_cache!(model::FSLModel)

    ft = [_ft for _ft in (fieldtype.(FSLModel, fieldnames(FSLModel)))]
    make_cache!.(model,  [_f for _f in fieldnames(FSLModel)[findall(==(true), (ft .===Union{Nothing, Function}))]])

    return nothing
end



""" reset the FSLModel object entirely """
reset!(model::FSLModel) = begin reset!.(model, [field for field in fieldnames(FSLModel)]); return nothing; end
reset!(model::FSLModel, s::Symbol) = begin ((fieldtype(FSLModel, s) === Union{Nothing, Function}) && setproperty!(model, s, nothing)); return nothing; end

