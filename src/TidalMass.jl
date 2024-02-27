export tidal_mass, min_concentration_mt, mΔ_from_mt, xt_from_mt

""" tidal mass from the other parameters"""
tidal_mass(mΔ::Real, cΔ::Real, xt::Real, hp::HaloProfile) = mΔ * μ_halo(xt, hp) / μ_halo(cΔ, hp)
tidal_mass(mΔ::Real, cΔ::Real, r_host::Real, model::FSLModel) = tidal_mass(mΔ, cΔ, tidal_scale(r_host, halo_from_mΔ_and_cΔ(model.context.subhalo_profile, mΔ, cΔ)), model.context.subhalo_profile)


""" inverse function giving mΔ in terms of the tidal mass"""
function mΔ_from_mt(mt::Real, cΔ::Real, xt::Real, hp::HaloProfile) 

    m_min = 1e-16
    m_max = 1e+16

    res = -1.0

    _to_bisect(m::Real) = tidal_mass(m, cΔ, xt, hp)/mt - 1.0

    try
        res = exp(Roots.find_zero(lnm-> _to_bisect(exp(lnm)), (log(m_min), log(m_max)), Roots.Bisection(), xrtol=1e-2))
    catch e

        if isa(e, ArgumentError)
            # We set the mass to -1 as the tidal radius for this value of mt is too small to associate a mass mΔ (the tidal mass can be 0 or very small)
            (_to_bisect(m_max) <= 0) && (return -1.0)
        end

        msg = "Impossible to compute mΔ from mt at xt = "  * string(xt) * " Mpc, mt (cosmo) = " * string(mt) * " Msun, cΔ = " * string(cΔ) * " | [min, mean, max] = " * string(_to_bisect.([m_min, sqrt(m_min * m_max), m_max])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end

    return res
end


""" tidal scale from tidal mass"""
function xt_from_mt(mt::Real, mΔ::Real, cΔ::Real, hp::HaloProfile)
    
    if mt > mΔ
        throw(ArgumentError("The tidal mass must be less than the virial mass."))
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

    max_c = model.params.max_concentration
    res = -1.0

    println(r_host)

    # here we compare xt = rt/rs to the parameter ϵ_t
    function _to_bisect(c200::Real) 
        m200 = mΔ_from_mt(mt, c200, tidal_scale(r_host, halo_from_mΔ_and_cΔ(model.context.subhalo_profile, mΔ, c200)), model.context.subhalo_profile)
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
        m = mΔ_from_mt(mt, r_host, c, model)
        #pdf_concentration(c, m) * 
    end

end

#############################################################

cache_location::String = ".cache/"

const _NPTS_C = 200
const _NPTS_X = 200

export _save_mΔ_from_mt

function _save_mΔ_from_mt(hp::HaloProfile)
    
    @info "| Saving mΔ_from_mt in cache" 
    mt = 10.0.^range(-16, +16, _NPTS_M)
    cΔ = 10.0.^range(0, log10(500), _NPTS_C)
    xt = 10.0.^range(-15, log10(500), _NPTS_X)

    y = [mΔ_from_mt(m_val, c_val, x_val, hp) for m_val in mt, c_val in cΔ, x_val in xt]
  
    JLD2.jldsave(cache_location * string(s)  * "_" * string(hash(hp.name), base=16) * ".jld2" ; mt = mt,  cΔ = cΔ, xt = xt, y = y)

    return true

end


## Possibility to interpolate the model
function _load(hp::HaloProfile)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * "_" * string(hash(hp.name), base=16) * ".jld2" 

    !(file in filenames) && _save(hp)

    data    = JLD2.jldopen(cache_location * file)

    mt = data["mt"]
    cΔ = data["cΔ"]
    xt = data["xt"]
    
    y  = data["y"]

    log10_y = Interpolations.interpolate((log10.(mt), log10.(cΔ), log10.(xt),), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
    return (xt::Real, cΔ::Real, mt::Real) -> 10.0^log10_y(log10(mt), log10(cΔ), log10(xt))

end


#############################################################