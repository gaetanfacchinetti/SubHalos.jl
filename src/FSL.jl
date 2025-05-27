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

export subhalo_mass_function_template
export mass_function_merger_tree
export FSLParams, FSLParamsPL, FSLParamsMT, FSLContext, FSLOptions, FSLModel
export dflt_FSLParamsPL, dflt_FSLParamsMT, dflt_FSLContext, dflt_FSLOptions, dflt_FSLModel
export min_concentration, min_concentration_calibration, ccdf_concentration, ccdf_concentration_calibration
export mass_fraction, normalisation_factor, number_subhalos, unevolved_number_subhalos
export pdf_virial_mass, pdf_position, pdf_rmc_FSL, pdf_rm_FSL, pdf_r_FSL, pdf_m_FSL, density_rmc_FSL, density_rm_FSL, density_r_FSL, density_m_FSL
export test_FSL_1, test_FSL_2, test_FSL_3, get_hash_tidal_scale, set_tidal_scale_interpolator!, load_tidal_scale_interpolator!

abstract type FSLParams{T<:AbstractFloat} end

struct FSLParamsPL{T<:AbstractFloat} <: FSLParams{T}
    
    # basic parameters of the model
    α_m::T
    ϵ_t::T
    m_min::T

    # normalisation values
    mass_frac::T # mass fraction in the form of halo with simulation-kind parameters
    x1_frac::T # m1/mhost for the integral of the normalisation
    x2_frac::T # m2/mhost for the integral of the normalisation

    # to be properly implemented in the future
    z::T
end

struct FSLParamsMT{T<:AbstractFloat} <: FSLParams{T}
    
    # basic parameters of the model
    γ_1::T
    α_1::T
    γ_2::T
    α_2::T
    β::T
    ζ::T
    ϵ_t::T
    m_min::T

    # to be properly implemented in the future
    z::T
end

# definition of length and iterator on our struct
# allows to use f.(x, y) where y is of type FSLModel
Base.length(::FSLParams) = 1
Base.iterate(iter::FSLParams) = (iter, nothing)
Base.iterate(::FSLParams, state::Nothing) = nothing

FSLParamsPL(α_m::T, ϵ_t::T, m_min::T) where {T<:AbstractFloat} = FSLParamsPL(α_m, ϵ_t, m_min, T(0.11), T(2.2e-6), T(8.8e-4), T(0.0))
FSLParamsMT(γ_1::T, α_1::T, γ_2::T, α_2::T, β::T, ζ::T, ϵ_t::T, m_min::T) where {T<:AbstractFloat} = FSLParamsMT(γ_1, α_1, γ_2, α_2, β, ζ, ϵ_t, m_min, T(0.0))


struct FSLContext{
    T<:AbstractFloat, 
    C<:Cosmology, 
    HM<:HostModel, 
    HP<:HaloProfile,
    PP<:ProfileProperties
    }
    
    cosmo::C
    host::HM
    subhalo_profile::HP
    
    θ::T # angle between the subhalo's trajectory and the disk
    q::T # fraction of crossing with an impact parameter b > b_0(q)
    
    # Precomputed values
    m200_host::T  # defined from the HostHalo
    pp::PP        # defined from the HaloProfile
end


function FSLContext(
    cosmo::Cosmology = planck18, 
    host::HostModel = milky_way_MM17_g1, 
    sp::HaloProfile = nfwProfile; 
    θ::AbstractFloat = π/3,
    q::AbstractFloat = 0.2)

    T, BKG = get_cosmology_type(cosmo)
    THM, U = get_host_halo_type(host)
    S      = get_halo_profile_type(sp)
    
    @assert THM === T

    pp = ProfileProperties(sp, T)
    return FSLContext{T, Cosmology{T, BKG}, HostModel{T, U}, typeof(sp), ProfileProperties{T, S}}(cosmo, host, sp, θ, q, convert(T, mΔ(host.halo, 200, cosmo)), pp)
end

get_context_type(::FSLContext{T, C, HM, HP, PP}) where {T<:AbstractFloat, C<:Cosmology, HM<:HostModel, HP<:HaloProfile, PP<:ProfileProperties} = T, C, HM, HP, PP


# options to run the code
struct FSLOptions{T<:AbstractFloat}
    disk::Bool
    stars::Bool
    mc_model::Symbol
    use_tables::Bool
    use_nn::Bool
    c_max::T
end

@inline function FSLOptions(
    disk::Bool = true, 
    stars::Bool = false, 
    mc_model::Symbol = :SC12, 
    use_tables::Bool = true, 
    use_nn::Bool = true, 
    ::Type{T} = Float64
    ) where {T<:AbstractFloat} 
    
    return  FSLOptions(disk, stars, mc_model, use_tables, use_nn, T(500.0))
end


mutable struct FSLModel{ 
    P<:FSLParams, 
    Q<:FSLContext, 
    O<:FSLOptions}

    const params::P
    const context::Q
    const options::O

    tidal_scale::Union{Nothing, Function, TidalScaleNNInterpolator}
    min_concentration::Union{Nothing, Function}
    min_concentration_mt::Union{Nothing, Function}
    min_concentration_calibration::Union{Nothing, Function}

end


##################
## In construction

const GridInterpolator3D{T} = Interpolations.GriddedInterpolation{T, 3, Array{T, 3}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{T}, Vector{T}, Vector{T}}}

# wrapper structure of the tidal scale type
# in order to evaluate function without overhead 
# and ensuring type stability
struct TidalScaleInterpolator{
    T<:AbstractFloat, 
    S<:Real, 
    U<:Real, 
    C<:Cosmology{T, <:BkgCosmology{T}}, 
    FSLP<:FSLParams{T}, 
    HM<:HostModel{T, U}, 
    HP<:HaloProfile{S}, 
    PP<:ProfileProperties{T, S}
    }

    model::FSLModel{FSLP, FSLContext{T, C, HM, HP, PP}, FSLOptions{T}}
    interp::Union{GridInterpolator3D{T}, TidalScaleNNInterpolator}
end 

function (f::TidalScaleInterpolator)(r_host::T, c200::T, m200::T) where {T<:AbstractFloat}
    return f.interp(r_host, c200, m200)
end


function TidalScaleInterpolator(model::FSLModel)

    # load the table
    if model.use_nn === false

        !(isdir(cache_location)) && mkdir(cache_location)
        filenames  = readdir(cache_location)
        file       = string(s) * "_" * get_hash(model, :tidal_scale) * ".jld2" 

        !(file in filenames) && make_cache!(model, :tidal_scale)
    end


end

###################


function set_tidal_scale_interpolator!(model::FSLModel, tsi::TidalScaleNNInterpolator)
   
    if model.options.use_nn === false
        @warn "Not necessary to load an interpolator for this model if use_nn is set to false"
    end

    if model.options.stars === false
        @warn "Not necessary to load an interpolator for this model if encounters with stars switched off"
    end

    hash_str = get_hash_nn(model.context.host, model.context.subhalo_profile, model.context.cosmo, model.params.z, 200.0)
    
    if hash_str == tsi.metadata.hash
        model.tidal_scale = tsi
    else
        @warn "FSL model and interpolator metadata are incompatible"
    end
end

""" if an interpolator for the tidal scale already exists we can load it directly """
@inline function load_tidal_scale_interpolator!(model::FSLModel)
    model.tidal_scale = load_tidal_scale_interpolator(model.context.host, model.context.subhalo_profile, model.context.cosmo, model.params.z, 200.0)
end


@inline function FSLModel(params::P, context::Q, options::O) where {P<:FSLParams,  Q<:FSLContext, O<:FSLOptions} 
    return FSLModel{P, Q, O}(params, context, options, nothing, nothing, nothing, nothing)
end 

# definition of length and iterator on our struct
# allows to use f.(x, y) where y is of type FSLModel
Base.length(::FSLModel) = 1
Base.iterate(iter::FSLModel) = (iter, nothing)
Base.iterate(::FSLModel, state::Nothing) = nothing


# definition of a constant model
const dflt_FSLOptions::FSLOptions   = FSLOptions(true, false, :SCP12, true, true, 500.0)
const dflt_FSLContext::FSLContext   = FSLContext(planck18, milky_way_MM17_g1, nfwProfile)
const dflt_FSLParamsPL::FSLParamsPL = FSLParamsPL(1.95, 1e-2, 1e-6)
const dflt_FSLParamsMT::FSLParamsMT = FSLParamsMT(0.019, 1.94, 0.464, 1.58, 24.0, 3.4, 1e-2, 1e-6)

dflt_FSLModel::FSLModel = FSLModel(dflt_FSLParamsPL, dflt_FSLContext, dflt_FSLOptions)


""" tidal scale in terms of the parameters of the model, r_host, c200 and m200 """
function tidal_scale(r_host::T, c200::T, m200::T, model::FSLModel) where {T<:AbstractFloat}
    return tidal_scale(r_host, halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=T(200.0), ρ_ref = model.context.cosmo.bkg.ρ_c0), 
        model.context.host, model.params.z, T(200.0), model.context.cosmo,  pp = model.context.pp,
        q = model.context.q, θ = model.context.θ, disk = model.options.disk, stars = model.options.stars, use_tables = model.options.use_tables) 
end 


struct BissectionConcentration{
    T<:AbstractFloat, 
    S<:Real, 
    U<:Real, 
    C<:Cosmology{T, <:BkgCosmology{T}}, 
    FSLP<:FSLParams{T}, 
    HM<:HostModel{T, U}, 
    HP<:HaloProfile{S}, 
    PP<:ProfileProperties{T, S}
    }

    r_host::T
    m200::T
    model::FSLModel{FSLP, FSLContext{T, C, HM, HP, PP}, FSLOptions{T}}

end


function (f::BissectionConcentration{T, S, U, C, FSLP, HM, HP, PP})(
    c200::T
    ) where {T<:AbstractFloat, 
    S<:Real, 
    U<:Real, 
    C<:Cosmology{T, <:BkgCosmology{T}}, 
    FSLP<:FSLParams{T}, 
    HM<:HostModel{T, U}, 
    HP<:HaloProfile{S}, 
    PP<:ProfileProperties{T, S}}
   
    # if we use interpolation tables then we use the interpolation tables   
    (f.model.options.use_nn) && (return f.model.tidal_scale(f.r_host, c200, f.m200, f.model.context.θ, f.model.context.q) / f.model.params.ϵ_t - T(1))
    #(f.model.options.use_tables) && (return f.model.tidal_scale(f.r_host, c200, f.m200) / f.model.params.ϵ_t - T(1))

    subhalo = halo_from_mΔ_and_cΔ(f.model.context.subhalo_profile, f.m200, c200, Δ=T(200), ρ_ref = f.model.context.cosmo.bkg.ρ_c0)
    return tidal_scale(f.r_host, subhalo, f.model.context.host, f.model.params.z, T(200), f.model.context.cosmo, pp=f.model.context.pp, q=f.model.context.q, θ=f.model.context.θ, disk=f.model.options.disk, stars=f.model.options.stars, use_tables=f.model.options.use_tables) / f.model.params.ϵ_t - T(1)
end


""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) """
function min_concentration(r_host::T, m200::T, model::FSLModel = dflt_FSLModel) where {T<:AbstractFloat}

    max_c = model.options.c_max
    res = T(-1)

    f = BissectionConcentration(r_host, m200, model)

    try
        res = Roots.find_zero(f, (T(1), max_c), Roots.Bisection(), xrtol=T(1e-3))
    catch e
        
        if isa(e, ArgumentError)
            # if the problem is that at large c we still have xt < ϵ_t then the min concentration is set to the max value of c
            (f(max_c) <= 0) && (return max_c)
            (f(1.0) >= 0) && (return T(1))
        end

        msg::String = "Impossible to compute min_concentration for rhost = "  * string(r_host) * " Mpc, m200 (cosmo) = " * string(m200) * " Msun | [min, mean, max] = " * string(f.([1.0, (max_c + 1.0)/2.0, max_c])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end

    return res

end



""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) not including baryons """
function min_concentration_calibration(r_host::T, m200::T, model::FSLModel = dflt_FSLModel) where {T<:AbstractFloat}

    max_c = model.options.c_max
    res = -1.0

    # here we compare xt = rt/rs to the parameter ϵ_t
    # for calibration on DM-only simulation only the Jacobi radius matters and ϵ_t = 1
    function _to_bisect(c200::AbstractFloat) 
        subhalo = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=T(200), ρ_ref = model.context.cosmo.bkg.ρ_c0)
        return min(jacobi_scale_DM_only(r_host, subhalo, model.context.host), c200)  - 1
    end

    try
        res = Roots.find_zero(c200 -> _to_bisect(c200), (1, max_c), Roots.Bisection(), xrtol=T(1e-3))
    catch e
        if isa(e, ArgumentError)
            # if the problem is that at large c we still have xt < ϵ_t then the min concentration is set to the max value of c
            (_to_bisect(max_c) <= 0.0) && (return max_c)
            (_to_bisect(1.0) >= 0.0) && (return 1.0)
        end

        msg = "Impossible to compute min_concentration_calibration for rhost = "  * string(r_host) * " Mpc, m200 (cosmo) = " * string(m200) * " Msun | [min, mean, max] = " * string(_to_bisect.([1.0, (max_c + 1.0)/2.0, max_c])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end
end

""" complementary cumulative distribution function of the concentration at min_concentration """
function ccdf_concentration(r_host::T, m200::T, model::FSLModel = dflt_FSLModel) where {T<:AbstractFloat}
    
    c_min = model.options.use_tables ? convert(T, model.min_concentration(r_host, m200)) : min_concentration(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

end

""" complementary cumulative distribution function of the concentration at min_concentration_DM_only """
function ccdf_concentration_calibration(r_host::T, m200::T, model::FSLModel = dflt_FSLModel) where {T<:AbstractFloat}
    
    c_min = model.options.use_tables ? model.min_concentration_calibration(r_host, m200) : min_concentration_calibration(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

end

""" fraction of mass in the form of subhalos per number of subhalos and without normalisation"""
function _mass_fraction(m1::T, m2::T, model::FSLModel; calibration::Bool = false) where {T<:AbstractFloat}
    
    _to_integrate_on_m(r_host::Real, m::Real) = m * pdf_virial_mass(m, model.context.m200_host, model.params) * (calibration ? ccdf_concentration_calibration(r_host, m, model) : ccdf_concentration(r_host, m, model)) 
    _to_integrate_on_r(r_host::Real) = QuadGK.quadgk(lnm -> _to_integrate_on_m(r_host, exp(lnm))*exp(lnm), log(m1), log(m2), rtol = 1e-3)[1] * pdf_position(r_host, model)
    
    r_min = 1e-2 * model.context.host.halo.rs
    r_max = model.context.host.rt

    return QuadGK.quadgk(lnr -> _to_integrate_on_r(exp(lnr))*exp(lnr), log(r_min), log(r_max), rtol = 1e-3)[1] / model.context.m200_host

end

mass_fraction( model::FSLModel, m1::Real = 2.2e-6 * model.context.m200_host, m2::Real = 8.8e-4 * model.context.m200_host) = _mass_fraction(m1, m2, model) * unevolved_number_subhalos(model)

""" normalisation factor for the probabitlity density """
function normalisation_factor(model::FSLModel)

    m_200_min  = model.params.m_min

    _to_integrate_on_m(r_host::Real, m::Real) = pdf_virial_mass(m, model.context.m200_host, model.params) * ccdf_concentration(r_host, m, model)
    _to_integrate_on_r(r_host::Real) = QuadGK.quadgk(lnm -> _to_integrate_on_m(r_host, exp(lnm))*exp(lnm), log(m_200_min), log(model.context.m200_host), rtol= 1e-3)[1] * pdf_position(r_host, model) 
    
    r_min = 1e-2 * model.context.host.halo.rs
    r_max = model.context.host.rt

    return QuadGK.quadgk(lnr -> _to_integrate_on_r(exp(lnr))*exp(lnr), log(r_min), log(r_max), rtol = 1e-3)[1]
end


@doc raw"""
    unevolved_number_subhalos(model; [mass_frac, [x_range]])

Number of subhalos in the host defined in model.context.host before accounting tidal effects
"""
function unevolved_number_subhalos(model::FSLModel)
    
    if isa(model.params, FSLParamsMT)
        return unevolved_number_subhalos(model.context.m200_host, model.params)
    else
        mass_frac_num = _mass_fraction(model.params.x1_frac * model.context.m200_host, model.params.x2_frac * model.context.m200_host, model, calibration = true)
        return model.params.mass_frac / mass_frac_num
    end

end

""" number of subhalos surviving tidal disruptions """
number_subhalos(model::FSLModel) = unevolved_number_subhalos(model) * normalisation_factor(model)

_pdf_rmc_FSL(r_host::Real, m200::Real, c200::Real, model::FSLModel) = pdf_position(r_host, model) * pdf_virial_mass(m200, model) * pdf_concentration(c200, m200, model) 
_pdf_rm_FSL(r_host::Real, m200::Real, model::FSLModel) = pdf_position(r_host, model) * pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_m_knowing_r_FSL(m200::Real, r_host::Real, model::FSLModel) = pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_r_knowing_m_FSL(r_host::Real, m200::Real, model::FSLModel) =  pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_r_FSL(r_host::T, model::FSLModel) where {T<:AbstractFloat} = convert(T, QuadGK.quadgk(lnm -> pdf_virial_mass(exp(lnm), model) * ccdf_concentration(r_host, exp(lnm), model) * exp(lnm), log(model.params.m_min), log(model.context.m200_host), rtol = 1e-3)[1] * pdf_position(r_host, model))
_pdf_m_FSL(m200::Real, model::FSLModel) = QuadGK.quadgk(lnr -> pdf_position(exp(lnr), model) * ccdf_concentration(exp(lnr), m200, model) * exp(lnr), log(1e-2 * model.context.host.halo.rs), log(model.context.host.rt), rtol = 1e-3)[1] * pdf_virial_mass(m200, model)

pdf_rmc_FSL(r_host::Real, m200::Real, c200::Real, model::FSLModel) = _pdf_rmc_FSL(r_host, m200, c200, model) / normalisation_factor(model)
density_rmc_FSL(r_host::Real, m200::Real, c200::Real, model::FSLModel) = _pdf_rmc_FSL(r_host, m200, c200, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)

pdf_rm_FSL(r_host::Real, m200::Real, model::FSLModel) = _pdf_rm_FSL(r_host, m200, model) / normalisation_factor(model)
pdf_m_knowing_r_FSL(m200::Real, r_host::Real, model::FSLModel) = _pdf_m_knowing_r_FSL(m200, r_host, model) / normalisation_factor(model)
pdf_r_knowing_m_FSL(r_host::Real, m200::Real, model::FSLModel) = _pdf_r_knowing_m_FSL(r_host, m200, model) / normalisation_factor(model)  

density_rm_FSL(r_host::Real, m200::Real, model::FSLModel) = _pdf_rm_FSL(r_host, m200, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)
density_m_knowing_r_FSL(m200::Real, r_host::Real, model::FSLModel) = _pdf_m_knowing_r_FSL(m200, r_host, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)
density_r_knowing_m_FSL(r_host::Real, m200::Real, model::FSLModel) = _pdf_r_knowing_m_FSL(r_host, m200, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)

pdf_r_FSL(r_host::Real, model::FSLModel) = _pdf_r_FSL(r_host, model) / normalisation_factor(model)
density_r_FSL(r_host::AbstractFloat, model::FSLModel) = _pdf_r_FSL(r_host, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)

pdf_m_FSL(m200::Real, model::FSLModel) = _pdf_m_FSL(m200, model) / normalisation_factor(model)
density_m_FSL(m200::Real, model::FSLModel) = _pdf_m_FSL(m200, model) * unevolved_number_subhalos(model)

test_FSL_1(model::FSLModel) = QuadGK.quadgk(lnr -> _pdf_r_FSL(exp(lnr), model) * exp(lnr),log(1e-2 * model.context.host.halo.rs), log(model.context.host.rt), rtol = 1e-3)[1]
test_FSL_2(model::FSLModel) = QuadGK.quadgk(lnm -> _pdf_m_FSL(exp(lnm), model) * exp(lnm),log(model.params.m_min), log(model.context.m200_host), rtol = 1e-3)[1]
test_FSL_3(model::FSLModel) = 4.0 * π *QuadGK.quadgk(lnr -> exp(lnr)^2 * density_r_FSL(exp(lnr), model) * exp(lnr),log(1e-2 * model.context.host.halo.rs), log(model.context.host.rt), rtol = 1e-3)[1]

#############################################################
# Definition of the unevolved distribution functions

@doc raw""" 
    subhalo_mass_function_template_MT(x, γ1, α1, γ2, α2, β, ζ)

Template function for the subhalo mass function fitted on merger trees:

``m_Δ^{\rm host} \frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}} = \left(\gamma_1 x^{-\alpha_1} + \gamma_2 x^{-\alpha_2}\right)  e^{-\beta x^\zeta}``

The first argument, `x::Real`, is the ratio of the subhalo over the host mass ``m_Δ^{\rm sub} / m_Δ^{\rm host}.``
"""
function subhalo_mass_function_template_MT(x::Real, γ1::Real,  α1::Real, γ2::Real, α2::Real, β::Real, ζ::Real)
    return (γ1*x^(-α1) + γ2*x^(-α2)) * exp(-β * x^ζ )
end


pdf_virial_mass(mΔ_sub::T, mΔ_host::T, params::FSLParamsPL{T}) where {T<:AbstractFloat} = mΔ_sub > params.m_min ? (params.α_m-1) * mΔ_host^(params.α_m - 1) * 1/( (mΔ_host/params.m_min)^(params.α_m - 1) - 1) * mΔ_sub^(-params.α_m) : zero(T)
pdf_virial_mass(mΔ_sub::T, mΔ_host::T, params::FSLParamsMT{T}) where {T<:AbstractFloat} = mΔ_sub > params.m_min ? subhalo_mass_function_template_MT(mΔ_sub / mΔ_host, params.γ_1, params.α_1, params.γ_2, params.α_2, params.β, params.ζ) / mΔ_host / unevolved_number_subhalos(mΔ_host, params) : 0.0


@doc raw""" 
    pdf_virial_mass(mΔ_sub, model)

Probability distribution function for model (of type `FSLModel`).
If model.params is of type `FSLParamsPL` outputs a normalised power-law.
If model.params is of type `FSLParamsMT` outputs a normalised double power-law with exponentil cutoff as fitted over merger tree simulations
"""
pdf_virial_mass(mΔ_sub::Real, model::FSLModel) = pdf_virial_mass(mΔ_sub, model.context.m200_host, model.params)


function unevolved_number_subhalos(mΔ_host::Real, params::FSLParamsMT)
    
    _integral(x::Real, γ::Real, α::Real, β::Real, ζ::Real) = γ  / ζ  * ( x^(1.0-α) * SpecialFunctions.expint( (α-1.0)/ζ + 1.0,  β * x^ζ ) - SpecialFunctions.expint( (α-1.0)/ζ + 1.0,  β ))
    return _integral( params.m_min / mΔ_host, params.γ_1, params.α_1, params.β, params.ζ) + _integral(params.m_min / mΔ_host, params.γ_2, params.α_2, params.β, params.ζ)

end


std_mass_concentration(m::Real, ::Type{SCP12}) = 0.14 * log(10.0)
std_mass_concentration(m::Real, ::Type{T}) where {T<:MassConcentrationModel} = std_mass_concentration(m, T)

function pdf_concentration(c200::Real, m200::Real, z::Real = 0, cosmo::Cosmology = planck18, ::Type{T} = SCP12) where {T<:MassConcentrationModel}
   
    σ_c = std_mass_concentration(m200, T)
    median_c = median_concentration(m200, z, cosmo, T)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    return 1.0 / Kc / c200 / sqrt(2.0 * π) / σ_c * exp(-(log(c200) - log(median_c))^2 / 2.0 / σ_c^2)
end

pdf_concentration(c200::Real, m200::Real, model::FSLModel) = pdf_concentration(c200, m200, model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

function ccdf_concentration(c200::T,  m200::T, z::T, cosmo::Cosmology{T, BKG}, ::Type{M}) where {M<:MassConcentrationModel, T<:AbstractFloat, BKG<:BkgCosmology{T}}
    
    σ_c = std_mass_concentration(m200, M)
    median_c = median_concentration(m200, z, cosmo, M)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    # Careful, need to check that the difference of erf makes sense
    return 1.0 - (SpecialFunctions.erf(log(median_c)/(sqrt(2) * σ_c)) - SpecialFunctions.erf(log(median_c/c200)/(sqrt(2) * σ_c)))/ (2 * Kc)
end


pdf_position(r::T, host::HostModel{T, U}, cosmo::Cosmology{T, BKG} = planck18) where {T<:AbstractFloat, U<:Real, BKG<:BkgCosmology{T}} = 4 * π  * r^2 * ρ_halo(r, host.halo) / mΔ(host.halo, 200, cosmo)
pdf_position(r::AbstractFloat, model::FSLModel) =  4 * π  * r^2 * ρ_halo(r, model.context.host.halo) / model.context.m200_host

