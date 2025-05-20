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

export clean_cache!, make_cache!, load!, reset!

cache_location::String = ".cache/"

function _save_pseudo_mass(pp::ProfileProperties)
    
    # Check if the file already exists
    filename, exist = get_filename(pp, :pseudo_mass_I)
    (exist) && return true

    @info "| Saving pseudo_mass_I in cache" 
    β  = 10.0.^range(-7, 5, 200)
    xt = 10.0.^range(-7, 5, 200)

    # use parallelisation as every point is independant
    y = Array{Float64}(undef, 200, 200)
    Threads.@threads for iβ in 1:200 
        for ix in 1:200 
            y[iβ, ix] = pseudo_mass_I(β[iβ], xt[ix], pp.hp)
        end
    end

    JLD2.jldsave(filename ; β = β, xt = xt, y = y)

    return true

end


function _load_pseudo_mass!(pp::ProfileProperties)
    
    filename, exist = get_filename(pp, :pseudo_mass_I)
    !(exist) && _save_pseudo_mass(pp)
    
    data    = JLD2.jldopen(filename)

    _β  = data["β"]
    _xt = data["xt"]
    _y  = data["y"]

    log10_y = Interpolations.interpolate((log10.(_β), log10.(_xt),), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))

    function return_func_pm(β::Real, xt::Real)

        (β > xt) && (return 1.0) 
        
        if (β < _β[1]) || (β >= _β[end]) || (xt < _xt[1]) || (xt >= _xt[end])
            return pseudo_mass_I(β, xt, pp.hp)
        end

        return 10.0^log10_y(log10(β), log10(xt))
    end

    return return_func_pm
end



function _save_velocity_dispersion(pp::ProfileProperties)

    filename, exist = get_filename(pp, :velocity_dispersion)
    exist && _save_velocity_dispersion(pp)
    
    @info "| Saving velocity_dispersion in cache" 
    x  = 10.0.^range(-15, 5, 500)
    xt = 10.0.^range(-15, 5, 500)

    # use parallelisation as every point is independant
    y = Array{Float64}(undef, 500, 500)
    Threads.@threads for ixt in 1:500 
        for ix in 1:500 
            y[ix, ixt] = velocity_dispersion(x[ix], xt[ixt], pp.hp)
        end
    end

    JLD2.jldsave(filename ; x = x, xt = xt, y = y)

end

function _load_velocity_dispersion!(pp::ProfileProperties)
    
    filename, exist = get_filename(pp, :velocity_dispersion)
    !(exist) && _save_velocity_dispersion(pp)

    data    = JLD2.jldopen(filename)

    _x  = data["x"]
    _xt = data["xt"]
    _y  = data["y"]

    log10_y = Interpolations.interpolate((log10.(_x), log10.(_xt),), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))

    function return_func_vd(x::Real, xt::Real)

        (x >= xt) && (return 0.0) 
        
        if (x < _x[1]) || (x >= _x[end]) || (xt < _xt[1]) || (xt >= _xt[end])
            return velocity_dispersion(x, xt, pp.hp)
        end

        return 10.0^log10_y(log10(x), log10(xt))
    end

    return return_func_vd
end


###########################
# Functions for loading

get_hash(pp::ProfileProperties) = string(hash(pp.hp.name), base=16)
get_hash(hp::HaloProfile) = string(hash(hp.name), base=16)

function get_filename(pp::ProfileProperties, s::Symbol, str::String = "")
    
    (str != "") && (str = "_" * str )

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * str *  "_" * get_hash(pp) * ".jld2" 

    return cache_location * file, (file in filenames)

end

function _load!(pp::ProfileProperties, s::Symbol)
    (s === :pseudo_mass_I) && (return _load_pseudo_mass!(pp))
    (s === :velocity_dispersion) && (return _load_velocity_dispersion!(pp))
end


# redefinition of getproperty to instantiate tables
# automatic loading if the function is called
function Base.getproperty(obj::ProfileProperties, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load!(obj, s))
    end

    return getfield(obj, s)
end


# preload a number of functions
function load!(pp::ProfileProperties)

    for field in fieldnames(ProfileProperties)
        (getfield(pp, field) === nothing) && setproperty!(pp, field, _load!(pp, field))
    end
    
    return nothing
end

""" reset the ProfileProperties object entirely """
reset!(pp::ProfileProperties) = begin reset!.(pp, [field for field in fieldnames(ProfileProperties)]); return nothing; end
reset!(pp::ProfileProperties, s::Symbol) = begin ((fieldtype(ProfileProperties, s) === Union{Nothing, Function}) && setproperty!(pp, s, nothing)); return nothing; end

    
function clean_cache!(pp::ProfileProperties, s::Symbol) 
    file, exist = get_filename(pp, s)
    !(exist) && (@info "No saved function " * string(s) * " for ProfileProperty object" * string(pp))
    exist && rm(file)
    exist && setproperty!(pp, s, nothing) # reinitialise to nothing

    return nothing
end

""" clean the cache data and reset the ProfileProperty object"""
function clean_cache!(pp::ProfileProperties)

    # find all fields in the struct that are of type Union{Nothing, Function} and that can be deleted as part of the cache
    ft = [_ft for _ft in (fieldtype.(ProfileProperties, fieldnames(ProfileProperties)))]
    clean_cache!.(pp,  [_f for _f in fieldnames(ProfileProperties)[findall(==(true), (ft .===Union{Nothing, Function}))]])
    
    return nothing
end