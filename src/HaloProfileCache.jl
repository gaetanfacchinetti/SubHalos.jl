
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

export pseudo_mass_I, PseudoMass, PseudoMassType, VelocityDispersion, VelocityDispersionType, HaloProfileInterpolation, HaloProfileInterpolationType

_pseudo_mass_I(β::T, xt::T, shp::HaloProfile = nfwProfile) where {T<:AbstractFloat} = 1 - QuadGK.quadgk(lnx-> sqrt(exp(lnx)^2 - β^2)  * ρ_halo(exp(lnx), shp) * exp(lnx)^2 , log(β), log(xt), rtol=T(1e-12))[1] / μ_halo(xt, shp)

@doc raw"""

    pseudo_mass_I(β, xt [, shp])

integral expression (denoted I in arXiv:2201.09788):

`` I(\beta, x_{\rm t}) = \int_0^{\infty} {\rm d} x \, \frac{1}{(1+x^2)^(3/2)} * \frac{\mu(\beta \sqrt{1+x^2})}{\mu(x_{\rm t})} ``

with `` \beta = b / r_{\rm s}`` the ratio of the impact parameter over the scale radius and ``\mu`` the dimensionless mass

`` \mu(x) = \int_0^{\infty} \rho(x) x^2 {\rm d} x ``

such that ``\mu(x > x_{\rm t}) = \mu(x_{\rm t}) ``.
"""
function pseudo_mass_I(β::T, xt::T, shp::HaloProfile = nfwProfile) where {T<:AbstractFloat}

    (xt <= β)  && return 1
    (β == 0)   && return 0

    if (β < T(1e-5)) 
        try
            return _pseudo_mass_I(β, xt, shp)
        catch
            println("Probelem for β = $β and xt = $xt")
            rethrow()
        end
    end

    ((typeof(shp) <: αβγProfile) && (shp == plummerProfile)) && return (1 - (1 / (1 + β^2)) * (1 - β^2 / (xt^2))^(1.5))

    if ((typeof(shp) <: αβγProfile) && (shp == nfwProfile))

        (β > 1)  && return 1 + (sqrt(xt * xt - β * β) / (1 + xt) - acosh(xt / β) + (2 / sqrt(β * β - 1)) * atan(sqrt((β - 1) / (β + 1)) * tanh(acosh(xt / β) / 2 ))) / μ_halo(xt, shp)
        (β == 1) && return 1 - (-2 * sqrt((xt - 1) / (xt + 1)) + 2 * asinh(sqrt((xt - 1) / 2))) / μ_halo(xt, shp)
        (β < 1)  && return 1 + (sqrt(xt * xt - β * β) / (1 + xt) - acosh(xt / β) + (2 / sqrt(1 - β * β)) * atanh(sqrt((1 - β) / (β + 1)) * tanh(acosh(xt / β) / 2 ))) / μ_halo(xt, shp)
    end 

    # For whatever different profile
    return _pseudo_mass_I(β, xt, shp)

end

pseudo_mass_I(b::Real, rt::Real, sh::Halo) = pseudo_mass_I(b/sh.rs, rt/sh.rs, sh.hp)


function _save_pseudo_mass(hp::HaloProfile)
    
    # Check if the file already exists
    filename, exist = get_filename(hp, :pseudo_mass_I)
    (exist) && return true

    @info "| Saving pseudo_mass_I in cache" 
    β  = 10.0.^range(-7, 5, 200)
    xt = 10.0.^range(-7, 5, 200)

    # use parallelisation as every point is independant
    y = Array{Float64}(undef, 200, 200)
    @inbounds for iβ in 1:200, ix in 1:200 
        y[iβ, ix] = pseudo_mass_I(β[iβ], xt[ix], hp)
    end

    JLD2.jldsave(filename ; β = β, xt = xt, y = y)

    return true

end


function _save_velocity_dispersion(hp::HaloProfile)
    
    # Check if the file already exists
    filename, exist = get_filename(hp, :velocity_dispersion)
    (exist) && return true

    @info "| Saving velocity_dispersion in cache" 
    x  = 10.0.^range(-7, 5, 10000)
    xt = 10.0.^range(-7, 5, 10000)

    # use parallelisation as every point is independant
    y = Array{Float64}(undef, 10000, 10000)
    @inbounds for ix in 1:10000, ixt in 1:10000 
        y[ix, ixt] = velocity_dispersion(x[ix], xt[ixt], hp)
    end

    JLD2.jldsave(filename ; x = x, xt = xt, y = y)

    return true

end

#get_hash(pp::ProfileProperties) = string(hash(pp.hp.name), base=16)
get_hash(hp::HaloProfile) = string(hash(hp.name), base=16)

const GridInterpolator2D{T} = Interpolations.GriddedInterpolation{T, 2, Matrix{T}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{T}, Vector{T}}}

struct PseudoMass{T<:AbstractFloat, P<:HaloProfile{<:Real}}
    interp::GridInterpolator2D{T}
    log10_βmin::T
    log10_βmax::T
    log10_xtmin::T
    log10_xtmax::T
    hp::P
end

struct VelocityDispersion{T<:AbstractFloat, P<:HaloProfile{<:Real}}
    interp::GridInterpolator2D{T}
    log10_xmin::T
    log10_xmax::T
    log10_xtmin::T
    log10_xtmax::T
    hp::P
end

const PseudoMassType{T} = PseudoMass{T, <:HaloProfile{<:Real}}
const VelocityDispersionType{T} = VelocityDispersion{T, <:HaloProfile{<:Real}}

struct HaloProfileInterpolation{T<:AbstractFloat, P<:HaloProfile{<:Real}, PM<:PseudoMass{T, P}, VD<:VelocityDispersion{T, P}}
    pseudo_mass::PM
    velocity_dispersion::VD
    hp::P
end

const HaloProfileInterpolationType{T, P} = HaloProfileInterpolation{T, P, <:PseudoMass{T, P}, <:VelocityDispersion{T, P}}

function get_filename(hp::HaloProfile{T}, s::Symbol, str::String = "") where {T<:Real}
    
    (str != "") && (str = "_" * str )

    !(isdir(CACHE_LOCATION)) && mkdir(CACHE_LOCATION)
    filenames  = readdir(CACHE_LOCATION)
    file       = string(s) * str *  "_" * get_hash(hp) * ".jld2" 

    return CACHE_LOCATION * file, (file in filenames)

end

# Constructor for MyPseudoMass
function PseudoMass(hp::P, ::Type{T}=Float64) where {T<:AbstractFloat, P<:HaloProfile}
    
    filename, exist = get_filename(hp, :pseudo_mass_I)
    !exist && _save_pseudo_mass(hp)

    log10_y, log10_β, log10_xt = let
        JLD2.jldopen(filename, "r") do file
            T.(log10.(file["y"])), 
            T.(log10.(file["β"])), 
            T.(log10.(file["xt"]))
        end
    end

    interp = Interpolations.interpolate((log10_β, log10_xt), log10_y, Interpolations.Gridded(Interpolations.Linear()))
    
    log10_βmin, log10_βmax = extrema(log10_β)
    log10_xtmin, log10_xtmax = extrema(log10_xt)

    return PseudoMass{T, P}(interp, log10_βmin, log10_βmax, log10_xtmin, log10_xtmax, hp)
end


# Constructor for VelocityDispersion
function VelocityDispersion(hp::P, ::Type{T}=Float64) where {T<:AbstractFloat, P<:HaloProfile}
    
    filename, exist = get_filename(hp, :velocity_dispersion)
    !exist && _save_velocity_dispersion(hp)

    log10_y, log10_x, log10_xt = let
        JLD2.jldopen(filename, "r") do file
            T.(log10.(file["y"])), 
            T.(log10.(file["x"])), 
            T.(log10.(file["xt"]))
        end
    end

    interp = Interpolations.interpolate((log10_x, log10_xt), log10_y, Interpolations.Gridded(Interpolations.Linear()))
    
    log10_xmin, log10_xmax = extrema(log10_x)
    log10_xtmin, log10_xtmax = extrema(log10_xt)

    return VelocityDispersion{T, P}(interp, log10_xmin, log10_xmax, log10_xtmin, log10_xtmax, hp)
end


# Main functor method
# ATTENTION WE ACTUALLY RETURN I/β
function (f::PseudoMassType{T})(log10_β::T, log10_xt::T)::T where {T<:AbstractFloat}

    if log10_β === T(-Inf)
        xt = exp10(log10_xt)
        return ρ_halo(xt, f.hp) * xt^2 * exp10(log10_β) / μ_halo(xt, f.hp)
    end

    if log10_β >= log10_xt
        return one(T)/exp10(log10_β)
    elseif log10_β < f.log10_βmin || log10_β > f.log10_βmax || log10_xt < f.log10_xtmin || log10_xt > f.log10_xtmax
        return convert(T, pseudo_mass_I(exp10(log10_β), exp10(log10_xt), f.hp)) / exp10(log10_β)
    else
        return exp10(f.interp(log10_β, log10_xt) - log10_β)
    end

end


function (f::VelocityDispersionType{T})(log10_x::T, log10_xt::T)::T where {T<:AbstractFloat}

    res = -one(T)

    if log10_x >= log10_xt
        res = zero(T)
    else log10_x >= f.log10_xmin && log10_x <= f.log10_xmax && log10_xt >= f.log10_xtmin && log10_xt <= f.log10_xtmax
        res = convert(T, velocity_dispersion(exp10(log10_x), exp10(log10_xt), f.hp))
    end

    if res >= 0
        return res
    end

    return exp10(f.interp(log10_x, log10_xt))

end


function (f::VelocityDispersionType{T})(log10_x::Vector{T}, log10_xt::T)::Vector{T} where {T<:AbstractFloat}
    
    res = Vector{T}(undef, length(log10_x))

    res .= -1
    res[log10_x .>= log10_xt] .= 0

    mask = (log10_x .>= f.log10_xmin) .&& (log10_x .<= f.log10_xmax) .&& (log10_xt .>= f.log10_xtmin) .&& (log10_xt .<= f.log10_xtmax)
    res[mask] .= convert.(T, velocity_dispersion.(exp10.(log10_x[mask]), exp10(log10_xt), f.hp)) 
    
    res[res .< 0] = exp10.(f.interp.(log10_x[res .< 0], log10_xt))

    return res
end



function HaloProfileInterpolation(hp::P, ::Type{T}=Float64) where {T<:AbstractFloat, P<:HaloProfile}
    pm = PseudoMass(hp, T)
    vd = VelocityDispersion(hp, T)
    return HaloProfileInterpolation{T, P, PseudoMass{T, P}, VelocityDispersion{T, P}}(pm, vd, hp)
end

