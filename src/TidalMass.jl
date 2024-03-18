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

export tidal_mass, min_concentration_mt, mΔ_from_mt, xt_from_mt, density_mt_knowing_r_FSL

""" tidal mass from the other parameters"""
tidal_mass(mΔ::Real, cΔ::Real, xt::Real, hp::HaloProfile) = (xt === -Inf) ? NaN : mΔ * μ_halo(xt, hp) / μ_halo(cΔ, hp)

function tidal_mass(mΔ::Real, cΔ::Real, r_host::Real, model::FSLModel)
    xt = model.options.use_tables ? model.tidal_scale(r_host, cΔ, mΔ) : tidal_scale(r_host, halo_from_mΔ_and_cΔ(model.context.subhalo_profile, mΔ, cΔ))
    return tidal_mass(mΔ, cΔ, xt, model.context.subhalo_profile)
end

""" inverse function giving mΔ in terms of the tidal mass"""
function mΔ_from_mt(mt::Real, cΔ::Real, r_host::Real, model::FSLModel) 

    m_min = 1e-16
    m_max = model.context.m200_host

    res = -1.0

    _to_bisect(m::Real) = tidal_mass(m, cΔ, r_host, model)/mt - 1.0

    try
        res = exp(Roots.find_zero(lnm-> _to_bisect(exp(lnm)), (log(m_min), log(m_max)), Roots.Bisection(), xrtol=1e-3))
    catch e

        if isa(e, ArgumentError)
            # We set the mass to -1 as the tidal radius for this value of mt is too small to associate a mass mΔ (the tidal mass can be 0 or very small)
            (_to_bisect(m_max) <= 0) && (return -1.0)
            (_to_bisect(m_max) === NaN) && (return -1.0) 
        end

        msg = "Impossible to compute mΔ from mt at r_host = "  * string(r_host) * " Mpc, mt (cosmo) = " * string(mt) * " Msun, cΔ = " * string(cΔ) * " | [min, mean, max] = " * string(_to_bisect.([m_min, sqrt(m_min * m_max), m_max])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end

    return res
end


""" tidal scale from tidal mass"""
function xt_from_mt(mt::Real, mΔ::Real, cΔ::Real, hp::HaloProfile)
    
    if mt > mΔ
        throw(ArgumentError("The tidal mass must be less than the virial mass : mt = " * string(mt) * " Msun, mΔ = " * string(mΔ) * " Msun"))
    end

    xt_min = 1e-15
    xt_max = cΔ

    res = -1.0

    _to_bisect(xt::Real) = tidal_mass(mΔ, cΔ, xt, hp)/mt - 1.0

    try
        res = exp(Roots.find_zero(lnx-> _to_bisect(exp(lnx)), (log(xt_min), log(xt_max)), Roots.Bisection(), xrtol=1e-2))
    catch e

        if isa(e, ArgumentError)
            # We set the mass to -1 as the tidal radius for this value of mt is too small to associate a mass mΔ (the tidal mass can be 0 or very small)
            (_to_bisect(xt_max) >= 0) && (return 0.0)
        end

        msg = "Impossible to compute xt from mΔ = "  * string(mΔ) * " Mpc, mt (cosmo) = " * string(mt) * " Msun, cΔ = " * string(cΔ) * " | [min, mean, max] = " * string(_to_bisect.([xt_min, sqrt(xt_min * xt_max), xt_max])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end

    return res
end

"""
function der_lnxt_lnm(mΔ::Real, r_host::Real, cΔ::Real, model::FSLModel)
    
    subhalo_1 = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, mΔ, cΔ)
    subhalo_2 = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, mΔ, cΔ)

    xt_1 = tidal_scale(r_host, subhalo_1)
    xt_2 = tidal_scale(r_host, subhalo_2)

end
"""

""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) """
function min_concentration_mt(r_host::Real, mt::Real, model::FSLModel{<:Real} = dflt_FSLModel)

    max_c = model.options.c_max
    res = -1.0

    #table_mΔ_from_mt = _load_mΔ_from_mt(r_host, model)

    # here we compare xt = rt/rs to the parameter ϵ_t
    function _to_bisect(c200::Real) 
        m200 = mΔ_from_mt(mt, c200, r_host, model)
        (m200 == -1) && (return -1.0) # means the tidal radius from this value of mt was too small for tables (we take it to 0)
        return xt_from_mt(mt, m200, c200, model.context.subhalo_profile) / model.params.ϵ_t - 1.0
    end

    try
        res = Roots.find_zero(c200 -> _to_bisect(c200), (1.0, max_c), Roots.Bisection(), xrtol=1e-3)
    catch e
        if isa(e, ArgumentError)
            # if the problem is that at large c we still have xt < ϵ_t then the min concentration is set to the max value of c
            (_to_bisect(max_c) <= 0.0) && (return max_c)
            (_to_bisect(1.0) >= 0.0) && (return 1.0)
        end

        msg = "Impossible to compute min_concentration for rhost = "  * string(r_host) * " Mpc, mt (cosmo) = " * string(mt) * " Msun | [min, mean, max] = " * string(_to_bisect.([1.0, (max_c + 1.0)/2.0, max_c])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end
end


function density_mt_knowing_r_FSL(mt::Real, r_host::Real, model::FSLModel)

    function _to_integrate(c::Real)
        m200 = mΔ_from_mt(mt, c, r_host, model)
        (m200 == -1) && (return 0.0)
        (m200 <= model.params.m_min) && (return 0.0)
        xt = xt_from_mt(mt, m200, c, model.context.subhalo_profile)
        return pdf_virial_mass(m200, model) * pdf_concentration(c, m200, model) * μ_halo(c, model.context.subhalo_profile)  / μ_halo(xt, model.context.subhalo_profile) 
        ## MISSING A TERM ABOVE IF xt DEPENDS ON m
    end

    c_min = model.options.use_tables ? model.min_concentration_mt(r_host, mt) : min_concentration_mt(r_host, mt, model)

    return QuadGK.quadgk(lnc -> _to_integrate(exp(lnc)) * exp(lnc), log(c_min), log(model.options.c_max), rtol=1e-3)[1] * unevolved_number_subhalos(model)
end


#############################################################

function _save_mΔ_from_mt(r_host::Real, model::FSLModel)
    
    @info "| Saving mΔ_from_mt in cache" 
    mt = 10.0.^range(-16, +16, _NPTS_M)
    cΔ = 10.0.^range(0, log10(model.options.c_max), _NPTS_C)

    y = mΔ_from_mt.(mt, cΔ', r_host, model)

    # List here all the values that if changed would change the result
    hash_value = hash((r_host, model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.params.z))
  
    JLD2.jldsave(cache_location * "mΔ_from_mt_" * string(hash_value, base=16) * ".jld2" ; mt = mt,  cΔ = cΔ, y = y)

    return true
end


## Possibility to interpolate the model
function _load_mΔ_from_mt(r_host::Real, model::FSLModel)

    hash_value = hash((r_host, model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.params.z))

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = "mΔ_from_mt" * "_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save_mΔ_from_mt(r_host, model)

    data    = JLD2.jldopen(cache_location * file)

    mt = data["mt"]
    cΔ = data["cΔ"]
    y  = data["y"]

    log10_y = Interpolations.interpolate((log10.(mt), log10.(cΔ),), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
    return (mt::Real, cΔ::Real) -> 10.0^log10_y(log10(mt), log10(cΔ))

end


#############################################################