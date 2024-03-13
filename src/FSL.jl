export subhalo_mass_function_template
export mass_function_merger_tree
export FSLParams, FSLParamsPL, FSLParamsMT, FSLContext, FSLOptions, FSLModel
export dflt_FSLParamsPL, dflt_FSLParamsMT, dflt_FSLContext, dflt_FSLOptions, dflt_FSLModel
export min_concentration, min_concentration_calibration, ccdf_concentration, ccdf_concentration_calibration
export mass_fraction, normalisation_factor, number_subhalos, unevolved_number_subhalos
export pdf_virial_mass, pdf_position, pdf_rmc_FSL, pdf_rm_FSL, pdf_r_FSL, pdf_m_FSL, density_rmc_FSL, density_rm_FSL, density_r_FSL, density_m_FSL
export test_FSL_1, test_FSL_2, test_FSL_3, get_hash_tidal_scale


abstract type FSLParams{T<:Real} end


struct FSLParamsPL{T<:Real} <: FSLParams{T}
    
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

struct FSLParamsMT{T<:Real} <: FSLParams{T}
    
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

FSLParamsPL(α_m::Real, ϵ_t::Real, m_min::Real, mass_frac::Real = 0.11, x1_frac::Real = 2.2e-6, x2_frac::Real = 8.8e-4, z::Real = 0.0) = FSLParamsPL(promote(α_m, ϵ_t, m_min, mass_frac, x1_frac, x2_frac, z)...)
FSLParamsMT(γ_1::Real, α_1::Real, γ_2::Real, α_2::Real, β::Real, ζ::Real, ϵ_t::Real, m_min::Real, z::Real = 0.0) = FSLParamsMT(promote(γ_1, α_1, γ_2, α_2, β, ζ, ϵ_t, m_min, z)...)

struct FSLContext{T<:Real}
    
    cosmo::Cosmology{T}
    host::HostModel{T}
    subhalo_profile::HaloProfile{<:Real}
    
    θ::T # angle between the subhalo's trajectory and the disk
    q::T # fraction of crossing with an impact parameter b > b_0(q)
    
    # Precomputed values
    m200_host::T               # defined from the HostHalo
    pmtab::PseudoMassTable     # defined from the HaloProfile
end

function FSLContext(cosmo::Cosmology{<:Real} = planck18, host::HostModel{<:Real} = milky_way_MM17_g1, sp::HaloProfile{<:Real} = nfwProfile; θ::Real=π/3, q::Real = 0.2) 
    pmtab = PseudoMassTable(sp)
    return FSLContext(cosmo, host, sp, θ, q, mΔ(host.halo, 200, cosmo), pmtab)
end

# options to run the code
struct FSLOptions
    disk::Bool
    stars::Bool
    mc_model::Symbol
    use_tables::Bool
    c_max::Real
end

FSLOptions(disk::Bool = true, stars::Bool = false, mc_model::Symbol = :SC12, use_tables::Bool = true) = FSLOptions(disk, stars, mc_model, use_tables, 500.0)

mutable struct FSLModel{T<:Real}

    const params::FSLParams{T}
    const context::FSLContext{T}
    const options::FSLOptions

    tidal_scale::Union{Nothing, Function}
    min_concentration::Union{Nothing, Function}
    min_concentration_mt::Union{Nothing, Function}
    min_concentration_calibration::Union{Nothing, Function}
end

FSLModel(params::FSLParams{<:Real}, context::FSLContext{<:Real}, options::FSLOptions) = FSLModel(params, context, options, nothing, nothing, nothing, nothing)

# definition of length and iterator on our struct
# allows to use f.(x, y) where y is of type FSLModel
Base.length(::FSLModel) = 1
Base.iterate(iter::FSLModel) = (iter, nothing)
Base.iterate(::FSLModel, state::Nothing) = nothing

# redefinition of getproperty to instantiate tables
function Base.getproperty(obj::FSLModel, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        (s === :tidal_scale) && setfield!(obj, s, _load_tidal_scale(obj))
        (s !== :tidal_scale) && setfield!(obj, s, _load(obj, s))
    end

    return getfield(obj, s)
end


# definition of a constant model
const dflt_FSLOptions::FSLOptions   = FSLOptions(true, false, :SCP12, true, 500.0)
const dflt_FSLContext::FSLContext   = FSLContext(planck18, milky_way_MM17_g1, nfwProfile)
const dflt_FSLParamsPL::FSLParamsPL = FSLParamsPL(1.95, 1e-2, 1e-6, 0.0)
const dflt_FSLParamsMT::FSLParamsMT = FSLParamsMT(0.019, 1.94, 0.464, 1.58, 24.0, 3.4, 1e-2, 1e-6, 0.0)

dflt_FSLModel::FSLModel = FSLModel(dflt_FSLParamsPL, dflt_FSLContext, dflt_FSLOptions)


""" tidal scale in terms of the parameters of the model, r_host, c200 and m200 """
tidal_scale(r_host::Real, c200::Real, m200::Real, model::FSLModel{<:Real}) = tidal_scale(r_host, 
        halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=200.0, ρ_ref = model.context.cosmo.bkg.ρ_c0), 
        model.context.host, model.params.z, 200, model.context.cosmo,  pmtab = model.context.pmtab,
        q = model.context.q, θ = model.context.θ, disk = model.options.disk, stars = model.options.stars, use_tables = model.options.use_tables) 


""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) """
function min_concentration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)

    max_c = model.options.c_max
    res = -1.0

    # here we compare xt = rt/rs to the parameter ϵ_t
    function _to_bisect(c200::Real) 
        
        # if we use interpolation tables then we use the interpolation tables        
        (model.options.use_tables) && (return model.tidal_scale(r_host, c200, m200) / model.params.ϵ_t - 1.0)

        subhalo = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=200.0, ρ_ref = model.context.cosmo.bkg.ρ_c0)
        return tidal_scale(r_host, subhalo, model.context.host, model.params.z, 200, model.context.cosmo) / model.params.ϵ_t - 1.0
    end

    try
        res = Roots.find_zero(c200 -> _to_bisect(c200), (1.0, max_c), Roots.Bisection(), xrtol=1e-3)
    catch e
        
        if isa(e, ArgumentError)
            # if the problem is that at large c we still have xt < ϵ_t then the min concentration is set to the max value of c
            (_to_bisect(max_c) <= 0.0) && (return max_c)
            (_to_bisect(1.0) >= 0.0) && (return 1.0)
        end

        msg = "Impossible to compute min_concentration for rhost = "  * string(r_host) * " Mpc, m200 (cosmo) = " * string(m200) * " Msun | [min, mean, max] = " * string(_to_bisect.([1.0, (max_c + 1.0)/2.0, max_c])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end
end


""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) not including baryons """
function min_concentration_calibration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)

    max_c = model.options.c_max
    res = -1.0

    # here we compare xt = rt/rs to the parameter ϵ_t
    # for calibration on DM-only simulation only the Jacobi radius matters and ϵ_t = 1
    function _to_bisect(c200::Real) 
        subhalo = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=200.0, ρ_ref = model.context.cosmo.bkg.ρ_c0)
        return min(jacobi_scale_DM_only(r_host, subhalo, model.context.host), c200)  - 1.0
    end

    try
        res = Roots.find_zero(c200 -> _to_bisect(c200), (1.0, max_c), Roots.Bisection(), xrtol=1e-3)
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
function ccdf_concentration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)
    
    c_min = model.options.use_tables ? model.min_concentration(r_host, m200) : min_concentration(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

end

""" complementary cumulative distribution function of the concentration at min_concentration_DM_only """
function ccdf_concentration_calibration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)
    
    c_min = model.options.use_tables ? model.min_concentration_calibration(r_host, m200) : min_concentration_calibration(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

end

""" fraction of mass in the form of subhalos per number of subhalos and without normalisation"""
function _mass_fraction(m1::Real, m2::Real, model::FSLModel{<:Real}; calibration::Bool = false)
    
    _to_integrate_on_m(r_host::Real, m::Real) = m * pdf_virial_mass(m, model.context.m200_host, model.params) * (calibration ? ccdf_concentration_calibration(r_host, m, model) : ccdf_concentration(r_host, m, model)) 
    _to_integrate_on_r(r_host::Real) = QuadGK.quadgk(lnm -> _to_integrate_on_m(r_host, exp(lnm))*exp(lnm), log(m1), log(m2), rtol = 1e-3)[1] * pdf_position(r_host, model)
    
    r_min = 1e-2 * model.context.host.halo.rs
    r_max = model.context.host.rt

    return QuadGK.quadgk(lnr -> _to_integrate_on_r(exp(lnr))*exp(lnr), log(r_min), log(r_max), rtol = 1e-3)[1] / model.context.m200_host

end

mass_fraction( model::FSLModel{<:Real}, m1::Real = 2.2e-6 * model.context.m200_host, m2::Real = 8.8e-4 * model.context.m200_host) = _mass_fraction(m1, m2, model) * unevolved_number_subhalos(model)

""" normalisation factor for the probabitlity density """
function normalisation_factor(model::FSLModel{<:Real})

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
function unevolved_number_subhalos(model::FSLModel{<:Real})
    
    if isa(model.params, FSLParamsMT)
        return unevolved_number_subhalos(model.context.m200_host, model.params)
    else
        mass_frac_num = _mass_fraction(model.params.x1_frac * model.context.m200_host, model.params.x2_frac * model.context.m200_host, model, calibration = true)
        return model.params.mass_frac / mass_frac_num
    end

end

""" number of subhalos surviving tidal disruptions """
number_subhalos(model::FSLModel{<:Real}) = unevolved_number_subhalos(model) * normalisation_factor(model)

_pdf_rmc_FSL(r_host::Real, m200::Real, c200::Real, model::FSLModel) = pdf_position(r_host, model) * pdf_virial_mass(m200, model) * pdf_concentration(c200, m200, model) 
_pdf_rm_FSL(r_host::Real, m200::Real, model::FSLModel) = pdf_position(r_host, model) * pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_m_knowing_r_FSL(m200::Real, r_host::Real, model::FSLModel) = pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_r_knowing_m_FSL(r_host::Real, m200::Real, model::FSLModel) =  pdf_virial_mass(m200, model) * ccdf_concentration(r_host, m200, model)
_pdf_r_FSL(r_host::Real, model::FSLModel) = QuadGK.quadgk(lnm -> pdf_virial_mass(exp(lnm), model) * ccdf_concentration(r_host, exp(lnm), model) * exp(lnm), log(model.params.m_min), log(model.context.m200_host), rtol = 1e-3)[1] * pdf_position(r_host, model) 
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
density_r_FSL(r_host::Real, model::FSLModel) = _pdf_r_FSL(r_host, model) * unevolved_number_subhalos(model) / (4.0 * π * r_host^2)

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


pdf_virial_mass(mΔ_sub::Real, mΔ_host::Real, params::FSLParamsPL{<:Real}) = mΔ_sub > params.m_min ? (params.α_m-1.0) * mΔ_host^(params.α_m - 1.0) * 1.0/( (mΔ_host/params.m_min)^(params.α_m - 1.0) - 1.0 ) * mΔ_sub^(-params.α_m) : 0.0
pdf_virial_mass(mΔ_sub::Real, mΔ_host::Real, params::FSLParamsMT{<:Real}) = mΔ_sub > params.m_min ? subhalo_mass_function_template_MT(mΔ_sub / mΔ_host, params.γ_1, params.α_1, params.γ_2, params.α_2, params.β, params.ζ) / mΔ_host / unevolved_number_subhalos(mΔ_host, params) : 0.0


@doc raw""" 
    pdf_virial_mass(mΔ_sub, model)

Probability distribution function for model (of type `FSLModel`).
If model.params is of type `FSLParamsPL` outputs a normalised power-law.
If model.params is of type `FSLParamsMT` outputs a normalised double power-law with exponentil cutoff as fitted over merger tree simulations
"""
pdf_virial_mass(mΔ_sub::Real, model::FSLModel{<:Real}) = pdf_virial_mass(mΔ_sub, model.context.m200_host, model.params)


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

pdf_concentration(c200::Real, m200::Real, model::FSLModel{<:Real}) = pdf_concentration(c200, m200, model.params.z, model.context.cosmo, (@eval $(model.options.mc_model)))

function ccdf_concentration(c200::Real,  m200::Real, z::Real, cosmo::Cosmology, ::Type{T}) where {T<:MassConcentrationModel}
    
    σ_c = std_mass_concentration(m200, T)
    median_c = median_concentration(m200, z, cosmo, T)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    # Careful, need to check that the difference of erf makes sense
    return 1.0 - (SpecialFunctions.erf(log(median_c)/(sqrt(2) * σ_c)) - SpecialFunctions.erf(log(median_c/c200)/(sqrt(2) * σ_c)))/ (2 * Kc)
end


pdf_position(r::Real, host::HostModel, cosmo::Cosmology = planck18) = 4 * π  * r^2 * ρ_halo(r, host.halo) / mΔ(host.halo, 200, cosmo)
pdf_position(r::Real, model::FSLModel) =  4 * π  * r^2 * ρ_halo(r, model.context.host.halo) / model.context.m200_host


#############################################################

cache_location::String = ".cache/"

const _NPTS_R = 50
const _NPTS_M = 50
const _NPTS_C = 50

function _save_tidal_scale(model::FSLModel)
    
    rs = model.context.host.halo.rs
    rt = model.context.host.rt
    
    @info "| Saving tidal_scale in cache" 
    r = 10.0.^range(log10(1e-2 * rs), log10(rt), _NPTS_R)
    c = 10.0.^range(0, log10(model.options.c_max), _NPTS_C)
    m = nothing

    # Loading all the tabulated values for the host in order  
    # to avoid clashes between different threads in the following
    preload!(model.context.host)
    preload!(model.context.pmtab)

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

    JLD2.jldsave(cache_location * "tidal_scale_" * get_hash_tidal_scale(model) * ".jld2" ; r = r, m = m, c = c, y = y)
    return true
end


""" create a hash id for the saved tidal scale data """
function get_hash_tidal_scale(model::FSLModel)
    hash_tuple = (model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z, model.options.disk, model.options.stars)
    (model.options.stars) && (hash_tuple = (hash_tuple..., (model.context.q, model.context.θ)...))
    return string(hash(hash_tuple),  base=16)
end


## Possibility to interpolate the model
function _load_tidal_scale(model::FSLModel)

    hash_tuple = (model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z, model.options.disk, model.options.stars)
    (model.options.stars) && (hash_tuple = (hash_tuple..., (model.context.q, model.context.θ)...))
    hash_value = hash(hash_tuple)

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = "tidal_scale" * "_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save_tidal_scale(model)

    data    = JLD2.jldopen(cache_location * file)

    _r = data["r"]
    _m = data["m"]
    _c = data["c"]
    _y = data["y"]

    if !(model.options.stars)
        log10_y = Interpolations.interpolate((log10.(_r), log10.(_c)), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))
        
        return (r::Real, c::Real, m::Real = 1.0) -> 
        begin
            res = 10.0^log10_y(log10(r), log10(c))
            (res === NaN) && return -Inf
            return res 
        end

    else
        log10_y = Interpolations.interpolate((log10.(_r), log10.(_c), log10.(_m)), log10.(_y),  Interpolations.Gridded(Interpolations.Linear()))
        
        return (r::Real, c::Real, m::Real) -> 
        begin
            res = 10.0^log10_y(log10(r), log10(c), log10(m))
            (res === NaN) && return -Inf
            return res 
        end
    end


end

function _save(model::FSLModel, s::Symbol)
    
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

    if (s === :min_concentration_calibration)
        hash_value = hash((model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z))
    else
        hash_value = hash((model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.params.ϵ_t, model.options.c_max, model.params.z, model.options.disk, model.options.stars))
    end

    JLD2.jldsave(cache_location * string(s)  * "_" * string(hash_value, base=16) * ".jld2" ; r = r, m = m, y = y)

    return true

end


## Possibility to interpolate the model
function _load(model::FSLModel, s::Symbol)

    if (s === :min_concentration_calibration)
        hash_value = hash((model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.options.c_max, model.params.z))
    else
        hash_value = hash((model.context.host.name, model.context.cosmo.name, model.context.subhalo_profile.name, model.params.ϵ_t, model.options.c_max, model.params.z, model.options.disk, model.options.stars))
    end
    
    
    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * "_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save(model, s)

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

#####################################