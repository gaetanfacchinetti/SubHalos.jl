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


export pseudo_mass_I, ProfileProperties, solve_xt_new, average_δw2_stars, average_δv2_stars, average_δu2_stars, average_energy_kick_stars, b_min, β_min
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


const Interpolator{T} = Interpolations.GriddedInterpolation{T, 2, Matrix{T}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{T}, Vector{T}}}

struct PseudoMass{T<:AbstractFloat, S<:Real}
    interp::Interpolator{T}
    log10_βmin::T
    log10_βmax::T
    log10_xtmin::T
    log10_xtmax::T
    hp::HaloProfile{S}
end

function get_filename(hp::HaloProfile{T}, s::Symbol, str::String = "") where {T<:Real}
    
    (str != "") && (str = "_" * str )

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * str *  "_" * get_hash(hp) * ".jld2" 

    return cache_location * file, (file in filenames)

end

# Constructor for MyPseudoMass
function PseudoMass(hp::HaloProfile{S}, ::Type{T}=Float64) where {T<:AbstractFloat, S<:Real}
    
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

    return PseudoMass{T, S}(interp, log10_βmin, log10_βmax, log10_xtmin, log10_xtmax, hp)
end


mutable struct ProfileProperties{T<:AbstractFloat, S<:Real}
    hp::HaloProfile{S}
    pm::PseudoMass{T}
    velocity_dispersion::Union{Nothing, Function}
end

function ProfileProperties(hp::HaloProfile{S}, ::Type{T}=Float64) where {T<:AbstractFloat, S<:Real} 
    pm = PseudoMass(hp, T)
    return ProfileProperties(hp, pm, nothing)
end

# definition of length and iterator on our struct
# allows to use f.(x, y) where y is of type HaloProfile
Base.length(::ProfileProperties) = 1
Base.iterate(iter::ProfileProperties) = (iter, nothing)
Base.iterate(::ProfileProperties, state::Nothing) = nothing



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


# Fast 10^x using cached log(10)
const LOG10_64::Float64 = log(Float64(10.0))
const LOG10_32::Float32 = log(Float32(10.0))
const LOG10_16::Float16 = log(Float16(10.0))

@inline exp10_fast(x::Float64) = exp(LOG10_64 * x)
@inline exp10_fast(x::Float32) = exp(LOG10_32 * x)
@inline exp10_fast(x::Float16) = exp(LOG10_16 * x)

# Main functor method
@inline function (f::PseudoMass{T})(log10_β::T, log10_xt::T)::T where {T<:AbstractFloat}
    if log10_β > log10_xt
        return one(T)
    elseif log10_β < f.log10_βmin || log10_β > f.log10_βmax || log10_xt < f.log10_xtmin || log10_xt > f.log10_xtmax
        return convert(T, pseudo_mass_I(exp10_fast(log10_β), exp10_fast(log10_xt), f.hp))
    else
        return exp10_fast(f.interp(log10_β, log10_xt))
    end
end

# Integrand functor (avoid abusive memory usage)
struct Averageδw2Integrand{T<:AbstractFloat}
    x::T
    log10_xt::T
    psm::PseudoMass{T}
end

@inline function (f::Averageδw2Integrand{T})(log10_β::T)::T where {T<:AbstractFloat}
    βinv2 = inv(exp10_fast(2 * log10_β))
    pmI = f.psm(log10_β, f.log10_xt)
    return pmI*pmI + T(3) * (T(1) - T(2) * pmI) / (T(3) + T(2) * f.x * f.x * βinv2) * log(T(10))
end


# Compute average δw² for stars
function average_δw2_stars(x::T, xt::T, β_min::T, β_max::T, pp::ProfileProperties{T, S})::T where {T<:AbstractFloat, S<:Real}
        
    res = zero(T)
    f = Averageδw2Integrand{T}(x, log10(xt), pp.pm)

    if β_max >= xt
        res += T(0.5) * log(β_max * β_max / (T(3) * β_max * β_max + T(2) * x*x) * (T(3) + T(2) * x*x / max(xt, β_min)^2))
    end
    
    if β_min < xt
        res += QuadGK.quadgk(f, log10(β_min), log10(min(xt, β_max)); rtol=1e-3)[1]
    end

    return T(2) / (β_max^2 - β_min^2) * res
end




@doc raw""" 
    
    average_δw2_stars(x, xt, β_min, β_max, shp)

Average value of δw2. In order to avoid numerical issues ``\beta_{\rm min} > 0`` as the integral is performed in log-space.

`` \overline{(\delta w)^2} = \int_{\beta_{\rm min}}^{\beta_{\rm max}} (\delta w)^2 p_\beta(\beta) {\rm d} \beta``

with the probability distribution of `` \beta = b / r_{\rm s} `` given by

`` p_\beta(\beta) = \frac{2 \beta}{\beta_{\rm max}^2 - \beta_{\rm min}^2} ``.

The last argument, `shp` stands for subhalo profile and must be of type `HaloProfile{<:Real}``
"""
function my_average_δw2_stars(x::Real, xt::Real, β_min::Real, β_max::Real, pp::ProfileProperties)
    
    function _to_integrate(lnβ::Real) 
        pmI = pp.pseudo_mass_I(exp(lnβ), xt)
        return (pmI^2 + 3 * (1 - 2*pmI) / (3 + 2 * (x/exp(lnβ))^2))
    end

    res = 0.0

    (β_max >= xt) && (res  += 0.5 * log(β_max^2 / (3*β_max^2 + 2*x^2) * (3 + 2*x^2 / max(xt, β_min)^2 )))
    (β_min < xt) && (res += QuadGK.quadgk(_to_integrate, log(β_min), log(min(xt, β_max)), rtol= 1e-3)[1])

    return 2 / (β_max^2 - β_min^2) * res
end

""" result in (Mpc / s)^2 """
function average_δv2_stars(x::Real, xt::Real, β_min::Real, r_host::Real, rs::Real, host::HostModel{<:Real}, pp::ProfileProperties, use_tables::Bool = true)
    β_max = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)) / rs
    (β_max <= β_min) && return 0.0 # can happen using interpolation tables if the precision is not high enough
    return average_δu2_stars(r_host, rs, host, use_tables) * average_δw2_stars(x, xt, β_min, β_max, pp)
end

""" result in (Mpc / s)^2 """
function average_energy_kick_stars(x::Real, xt::Real, β_min::Real, r_host::Real, rs::Real, host::HostModel{<:Real}, pp::ProfileProperties, θ::Real = π/3, use_tables::Bool = true)
    n_stars = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    (n_stars == 0) && (return 0) # means that we are in a region with no stars
    return 0.5 * n_stars * average_δv2_stars(x, xt, β_min, r_host, rs, host, pp, use_tables)
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


