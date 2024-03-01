export pseudo_mass_I, PseudoMassTable, solve_xt_new, average_δw2_stars, average_δv2_stars, average_δu2_stars, average_energy_kick_stars, b_min, β_min
export moments_relative_velocity_kms, average_inverage_relative_velocity_sqr_kms, average_relative_velocity_kms, average_inverse_relative_velocity_kms

###################################
_pseudo_mass_I(β::Real, xt::Real, shp::HaloProfile = nfwProfile) = 1.0 - QuadGK.quadgk(lnx-> sqrt(exp(lnx)^2 - β^2)  * ρ_halo(exp(lnx), shp) * exp(lnx)^2 , log(β), log(xt), rtol=1e-12)[1] / μ_halo(xt, shp)

@doc raw"""

    pseudo_mass_I(β, xt [, shp])

integral expression (denoted I in arXiv:2201.09788):

`` I(\beta, x_{\rm t}) = \int_0^{\infty} {\rm d} x \, \frac{1}{(1+x^2)^(3/2)} * \frac{\mu(\beta \sqrt{1+x^2})}{\mu(x_{\rm t})} ``

with `` \beta = b / r_{\rm s}`` the ratio of the impact parameter over the scale radius and ``\mu`` the dimensionless mass

`` \mu(x) = \int_0^{\infty} \rho(x) x^2 {\rm d} x ``

such that ``\mu(x > x_{\rm t}) = \mu(x_{\rm t}) ``.
"""
function pseudo_mass_I(β::Real, xt::Real, shp::HaloProfile = nfwProfile)

    (xt <= β)  && return 1.0
    (β < 1e-5) && return _pseudo_mass_I(β, xt, shp)

    ((typeof(shp) <: αβγProfile) && (shp == plummerProfile)) && return (1.0 - (1.0 / (1.0 + β^2)) * (1 - β^2 / (xt^2))^(1.5))

    if ((typeof(shp) <: αβγProfile) && (shp == nfwProfile))

        (β > 1)  && return 1.0 + (sqrt(xt * xt - β * β) / (1 + xt) - acosh(xt / β) + (2. / sqrt(β * β - 1)) * atan(sqrt((β - 1) / (β + 1)) * tanh(0.5 * acosh(xt / β)))) / μ_halo(xt, shp)
        (β == 1) && return 1.0 - (-2 * sqrt((xt - 1) / (xt + 1)) + 2 * asinh(sqrt((xt - 1) / 2))) / μ_halo(xt, shp)
        (β < 1)  && return 1.0 + (sqrt(xt * xt - β * β) / (1 + xt) - acosh(xt / β) + (2. / sqrt(1 - β * β)) * atanh(sqrt((1 - β) / (β + 1)) * tanh(0.5 * acosh(xt / β)))) / μ_halo(xt, shp)
    end 

    # For whatever different profile
    return _pseudo_mass_I(β, xt, shp)

end

pseudo_mass_I(b::Real, rt::Real, sh::Halo) = pseudo_mass_I(b/sh.rs, rt/sh.rs, sh.hp)


mutable struct PseudoMassTable
    const hp::HaloProfile{<:Real}
    pseudo_mass_I::Union{Nothing, Function}
end

PseudoMassTable(hp::HaloProfile) = PseudoMassTable(hp, nothing)

function _save_pseudo_mass(pmtab::PseudoMassTable)
    
    @info "| Saving pseudo_mass_I in cache" 
    β  = 10.0.^range(-7, 5, 200)
    xt = 10.0.^range(-7, 5, 200)

    # use parallelisation as every point is independant
    y = Array{Float64}(undef, 200, 200)
    Threads.@threads for iβ in 1:200 
        for ix in 1:200 
            y[iβ, ix] = pseudo_mass_I(β[iβ], xt[ix], pmtab.hp)
        end
    end

    hash_value = hash(pmtab.hp.name)
    JLD2.jldsave(cache_location * "pseudo_mass_I_" * string(hash_value, base=16) * ".jld2" ; β = β, xt = xt, y = y)

end

function _load_pseudo_mass(pmtab::PseudoMassTable)
    
    hash_value = hash(pmtab.hp.name)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = "pseudo_mass_I_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save_pseudo_mass(pmtab)

    data    = JLD2.jldopen(cache_location * file)

    _β  = data["β"]
    _xt = data["xt"]
    _y  = data["y"]

    log10_y = Interpolations.interpolate((log10.(_β), log10.(_xt),), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))

    function return_func_pm(β::Real, xt::Real)

        (β > xt) && (return 1.0) 
        
        if (β < _β[1]) || (β >= _β[end]) || (xt < _xt[1]) || (xt >= _xt[end])
            return pseudo_mass_I(β, xt, pmtab.hp)
        end

        return 10.0^log10_y(log10(β), log10(xt))
    end

    return return_func_pm

end

# redefinition of getproperty to instantiate tables
function Base.getproperty(obj::PseudoMassTable, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load_pseudo_mass(obj))
    end

    return getfield(obj, s)
end


preload!(pmtab::PseudoMassTable) = _load_pseudo_mass(pmtab)


##################################################
# Relative velocity between stars and halos in a disk

pdf_relative_velocity(v::Real, σ::Real, v_star::Real) = (v_star^2 + v^2)/(2.0*σ^2) > 1e+2 ? 0.0 : sqrt(2.0/π) * v / (σ * v_star) * sinh(v * v_star / σ^2) *exp(-(v_star^2 + v^2)/(2.0*σ^2))

""" average relative velocity in units of σ and v_star """
function average_relative_velocity(σ::Real, v_star::Real)
    X = v_star / (sqrt(2.0) * σ)
    return σ * sqrt(2.0 / π) * (exp(-X^2) + sqrt(π)/2.0*(1+2*X^2)*SpecialFunctions.erf(X)/X)
end

""" average of one over the relative velocity"""
average_inverse_relative_velocity(σ::Real, v_star::Real) = SpecialFunctions.erf(v_star/(sqrt(2.0) * σ))/v_star

""" average of (one over the relative velocity squared) """
average_inverse_relative_velocity_sqr(σ::Real, v_star::Real) = sqrt(2.0)* SpecialFunctions.dawson(v_star/(sqrt(2.0) * σ)) / (σ * v_star)

function moments_relative_velocity(σ::Real, v_star::Real, n::Int)
    (n == 1)  && (return average_relative_velocity(σ, v_star))
    (n == -1) && (return average_inverse_relative_velocity(σ, v_star))
    (n == -2) && (return average_inverse_relative_velocity_sqr(σ, v_star))
    throw(ArgumentError("Moment of order " * string(n) * " not defined for the relative velocity"))
end

""" probability distribution function for the relative velocity between stars and subhalos in 1/(km/s) """
function pdf_relative_velocity_kms(v::Real, r_host::Real, host::HostModel{<:Real}, use_tables::Bool = true)
    σ = use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)
    v_star = use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)
    return pdf_relative_velocity(v, σ, v_star)
end

""" moments of the relative velocity"""
function moments_relative_velocity_kms(r_host::Real, host::HostModel{<:Real}, n::Int, use_tables::Bool = true)
    σ = use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)
    v_star = use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)
    return moments_relative_velocity(σ, v_star, n)
end

# define corresponding function in terms of the host
for f in [:average_relative_velocity, :average_inverse_relative_velocity, :average_inverage_relative_velocity_sqr]
    @eval begin 
        function ($(Symbol(string(f) * "_kms")))(r_host::Real, host::HostModel{<:Real}, use_tables::Bool = true)
            σ = use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)
            v_star = use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)
            return ($f)(σ, v_star)
        end
    end
end


####################################
# Energy gain per encounters and per disk crossing

function δw2_stars(x::Real, xt::Real, β::Real)
    pmI = pseudo_mass_I(β, xt, shp)
    return (pmI^2 + 3 * (1 - 2*pmI) / (3 + 2 * (x/β)^2)) / β^2
end

""" average value of the velocity kick squared per encounter, result in (Mpc / s)^2 for `rs` in Mpc """
function average_δu2_stars(r_host::Real, rs::Real, host::HostModel{<:Real}, use_tables::Bool= true)
    v_m2 = sqrt(moments_relative_velocity_kms(r_host, host, -2, use_tables)) / KM_TO_MPC
    m_p2 = sqrt(host.stars.mass_model.average_mstar2)
    return (2 * G_NEWTON * m_p2 / rs * v_m2)^2 
end

average_δu2_stars(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, use_tables::Bool = true) = average_δu2_stars(r_host, subhalo.rs, host, use_tables)
average_δu2_stars_kms(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, use_tables::Bool = true) = average_δu2_stars(r_host, subhalo, host, use_tables) * MPC_TO_KM


@doc raw""" 
    
    average_δw2_stars(x, xt, β_min, β_max, shp)

Average value of δw2. In order to avoid numerical issues ``\beta_{\rm min} > 0`` as the integral is performed in log-space.

`` \overline{(\delta w)^2} = \int_{\beta_{\rm min}}^{\beta_{\rm max}} (\delta w)^2 p_\beta(\beta) {\rm d} \beta``

with the probability distribution of `` \beta = b / r_{\rm s} `` given by

`` p_\beta(\beta) = \frac{2 \beta}{\beta_{\rm max}^2 - \beta_{\rm min}^2} ``.

The last argument, `shp` stands for subhalo profile and must be of type `HaloProfile{<:Real}``
"""
function average_δw2_stars(x::Real, xt::Real, β_min::Real, β_max::Real, pmtab::PseudoMassTable)
    
    function _to_integrate(lnβ::Real) 
        pmI = pmtab.pseudo_mass_I(exp(lnβ), xt)
        return (pmI^2 + 3 * (1 - 2*pmI) / (3 + 2 * (x/exp(lnβ))^2))
    end

    res = 0.0

    (β_max >= xt) && (res  += 0.5 * log(β_max^2 / (3*β_max^2 + 2*x^2) * (3 + 2*x^2 / max(xt, β_min)^2 )))
    (β_min < xt) && (res += QuadGK.quadgk(_to_integrate, log(β_min), log(min(xt, β_max)), rtol= 1e-3)[1])

    return 2 / (β_max^2 - β_min^2) * res
end

""" result in (Mpc / s)^2 """
function average_δv2_stars(x::Real, xt::Real, β_min::Real, r_host::Real, rs::Real, host::HostModel{<:Real}, pmtab::PseudoMassTable, use_tables::Bool = true)
    β_max = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)) / rs
    return average_δu2_stars(r_host, rs, host, use_tables) * average_δw2_stars(x, xt, β_min, β_max, pmtab)
end

""" result in (Mpc / s)^2 """
function average_energy_kick_stars(x::Real, xt::Real, β_min::Real, r_host::Real, rs::Real, host::HostModel{<:Real}, pmtab::PseudoMassTable, θ::Real = π/3, use_tables::Bool = true)
    n_stars = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    (n_stars == 0) && (return 0) # means that we are in a region with no stars
    return 0.5 * n_stars * average_δv2_stars(x, xt, β_min, r_host, rs, host, pmtab, use_tables)
end

####################################
# Evaluation of β_min

b_min_over_b_max(q::Real, n_stars::Real) = (n_stars > 0) ? sqrt(1-(1-q)^(1.0/floor(Int, n_stars))) : 1.0

function b_min(q::Real, r_host::Real, host::HostModel{<:Real}, θ::Real = π/3, use_tables::Bool = true) 
    b_max   = use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)
    n_stars = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    return b_min_over_b_max(q, n_stars) * b_max 
end

β_min(q::Real, r_host::Real, rs::Real, host::HostModel, θ::Real = π/3, use_tables::Bool = true) = b_min(q, r_host, host, θ, use_tables) / rs


