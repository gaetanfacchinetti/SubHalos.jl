export subhalo_mass_function_template
export mass_function_merger_tree
export FSLParams, FSLContext, FSLOptions, FSLModel
export dflt_FSLParams, dflt_FSLContext, dflt_FSLOptions, dflt_FSLModel
export min_concentration, min_concentration_DM_only, ccdf_concentration, ccdf_concentration_DM_only

struct FSLParams{T<:Real}
    
    # basic parameters of the model
    m_min::T
    α_m::T
    ϵ_t::T

    # to be properly implemented in the future
    z::T

    # other properties
    max_concentration::T
end

FSLParams(m_min::Real, α_m::Real, ϵ_t::Real, z::Real = 0.0, max_concentration::Real = 500.0) = FSLParams(promote(m_min, α_m, ϵ_t, z, max_concentration)...)

struct FSLContext{T<:Real}
    cosmo::Cosmology{T}
    host::HostModel{T}
    subhalo_profile::HaloProfile{<:Real}
end

# options to run the code
struct FSLOptions
    tidal_effects::Symbol
    mass_function_model::Symbol
    use_tables::Bool
    mass_concentration_model::Symbol
end

mutable struct FSLModel{T<:Real}

    params::FSLParams{T}
    const context::FSLContext{T}
    const options::FSLOptions

    min_concentration::Union{Nothing, Function}
    min_concentration_DM_only::Union{Nothing, Function}
end


FSLModel(params::FSLParams{<:Real}, context::FSLContext{<:Real}, options::FSLOptions) = FSLModel(params, context, options, nothing, nothing)

# definition of length and iterator on our struct
# allows to use f.(x, y) where y is of type FSLModel
Base.length(::FSLModel) = 1
Base.iterate(iter::FSLModel) = (iter, nothing)
Base.iterate(::FSLModel, state::Nothing) = nothing

# redefinition of getproperty to instantiate tables
function Base.getproperty(obj::FSLModel, s::Symbol)

    # we load the data if necessary
    if getfield(obj, s) === nothing
        setfield!(obj, s, _load(obj, s))
    end

    return getfield(obj, s)
end


# definition of a constant model
const dflt_FSLOptions::FSLOptions = FSLOptions(:all, :simu, true, :SCP12)
const dflt_FSLContext::FSLContext = FSLContext(planck18, milky_way_MM17_g1, nfwProfile)
const dflt_FSLParams::FSLParams = FSLParams(1e-6, -1.95, 1e-2, 0.0, 500.0)

dflt_FSLModel::FSLModel = FSLModel(dflt_FSLParams, dflt_FSLContext, dflt_FSLOptions)


""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) """
function min_concentration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)

    max_c = model.params.max_concentration
    res = -1.0

    # here we compare xt = rt/rs to the parameter ϵ_t
    function _to_bisect(c200::Real) 
        subhalo = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=200.0, ρ_ref = model.context.cosmo.bkg.ρ_c0)
        return tidal_scale(r_host, subhalo, model.context.host, model.params.z, model.context.cosmo) / model.params.ϵ_t - 1.0
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


""" minimal concentration of surviving halos at position r_host (Mpc) with mass m200 (Msun) not including baryons"""
function min_concentration_DM_only(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)

    max_c = model.params.max_concentration
    res = -1.0

    # here we compare xt = rt/rs to the parameter ϵ_t
    function _to_bisect(c200::Real) 
        subhalo = halo_from_mΔ_and_cΔ(model.context.subhalo_profile, m200, c200, Δ=200.0, ρ_ref = model.context.cosmo.bkg.ρ_c0)
        return jacobi_scale_DM_only(r_host, subhalo, model.context.host) / model.params.ϵ_t - 1.0
    end

    try
        res = Roots.find_zero(c200 -> _to_bisect(c200), (1.0, max_c), Roots.Bisection(), xrtol=1e-3)
    catch e
        if isa(e, ArgumentError)
            # if the problem is that at large c we still have xt < ϵ_t then the min concentration is set to the max value of c
            (_to_bisect(max_c) <= 0.0) && (return max_c)
            (_to_bisect(1.0) >= 0.0) && (return 1.0)
        end

        msg = "Impossible to compute min_concentration_DM_only for rhost = "  * string(r_host) * " Mpc, m200 (cosmo) = " * string(m200) * " Msun | [min, mean, max] = " * string(_to_bisect.([1.0, (max_c + 1.0)/2.0, max_c])) * "\n" * e.msg 
        throw(ArgumentError(msg))
    end
end

""" complementary cumulative distribution function of the concentration at min_concentration """
function ccdf_concentration(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)
    
    c_min = model.options.use_tables ? model.min_concentration(r_host, m200) : min_concentration(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mass_concentration_model)))

end

""" complementary cumulative distribution function of the concentration at min_concentration_DM_only """
function ccdf_concentration_DM_only(r_host::Real, m200::Real, model::FSLModel{<:Real} = dflt_FSLModel)
    
    c_min = model.options.use_tables ? model.min_concentration_DM_only(r_host, m200) : min_concentration_DM_only(r_host, m200, model)
    return ccdf_concentration(c_min, m200,  model.params.z, model.context.cosmo, (@eval $(model.options.mass_concentration_model)))

end




#############################################################
# Defnitions of basic functions

@doc raw""" 
    subhalo_mass_function_template(x, γ1, α1, γ2, α2, β, ζ)

Template function for the subhalo mass function:

``m_Δ^{\rm host} \frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}} = \left(\gamma_1 x^{-\alpha_1} + \gamma_2 x^{-\alpha_2}\right)  e^{-\beta x^\zeta}``

The first argument, `x::Real`, is the ratio of the subhalo over the host mass ``m_Δ^{\rm sub} / m_Δ^{\rm host}.``
"""
function subhalo_mass_function_template(x::Real, γ1::Real,  α1::Real, γ2::Real, α2::Real, β::Real, ζ::Real)
    return (γ1*x^(-α1) + γ2*x^(-α2)) * exp(-β * x^ζ )
end


@doc raw"""
    pdf_virial_mass(mΔ_sub, mΔ_host)

Example of subhalo mass function fitted on merger tree results
(Facchinetti et al., in prep.)

# Arguments
- `mΔ_sub::Real` : subhalo virial mass (in Msun)
- `mΔ_host::Real`: host virial mass (in Msun)

# Returns
- ``\frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}}``
"""
function density_virial_mass(mΔ_sub::Real, mΔ_host::Real, z_acc::Real = 0) 
    return subhalo_mass_function_template(mΔ_sub / mΔ_host, 0.019, 1.94, 0.464, 1.58, 24.0, 3.4)/mΔ_host
end

std_mass_concentration(m::Real, ::Type{SCP12}) = 0.14 * log(10.0)
std_mass_concentration(m::Real, ::Type{T}) where {T<:MassConcentrationModel} = std_mass_concentration(m, T)

function pdf_concentration(c200::Real, m200::Real, z::Real = 0, cosmo::Cosmology = planck18, ::Type{T} = SCP12) where {T<:MassConcentrationModel}
   
    σ_c = std_mass_concentration(m200, T)
    median_c = median_concentration(m200, z, cosmo, T)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    return 1.0 / Kc / c200 / sqrt(2.0 * π) / σ_c * exp(-(log(c200) - log(median_c))^2 / 2.0 / σ_c^2)
end

function ccdf_concentration(c200::Real,  m200::Real, z::Real, cosmo::Cosmology, ::Type{T}) where {T<:MassConcentrationModel}
    
    σ_c = std_mass_concentration(m200, T)
    median_c = median_concentration(m200, z, cosmo, T)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    # Careful, need to check that the difference of erf makes sense
    return 1.0 - (SpecialFunctions.erf(log(median_c)/(sqrt(2) * σ_c)) - SpecialFunctions.erf(log(median_c/c200)/(sqrt(2) * σ_c)))/ (2 * Kc)
end



#m200 in Msol
pdf_position(r::Real, host::HostModel, cosmo::Cosmology = planck18) = 4 * π  * r^2 * ρ_halo(r, host.halo) / mΔ(host.halo, 200, cosmo.bkg.ρ_c0)
density_virial_mass(m200::Real, host::HostModel, z = 0, cosmo::Cosmology = planck18) = pdf_virial_mass(m200, mΔ(host.halo, 200, cosmo.bkg.ρ_c0), z) 

pdf_FSL(r::Real, m200::Real, c200::Real) = pdf_position(r, host, cosmo) 

number_subhalos(host::HostModel, z=0, cosmo::Cosmology = planck18) =  QuadGK.quadgk(lnm -> exp(lnm))


#####################################

cache_location::String = ".cache/"

const _NPTS_R = 200
const _NPTS_M = 100

function _save(model::FSLModel, s::Symbol)
    
    rs = model.context.host.halo.rs
    rt = model.context.host.rt
    
    @info "| Saving " * string(s) * " in cache" 
    r = 10.0.^range(log10(1e-3 * rs), log10(rt), _NPTS_R)
    m = nothing

    if ((model.options.tidal_effects in [:jacobi_disk, :jacobi]) || (s === :min_concentration_DM_only))
        # here we do not have any dependance on the mass
        # skip saving the table in terms of the mass for efficiency
        y = @eval $s.($(Ref(r))[], 1.0, $(Ref(model))[])
    else
        m = 10.0.^range(-15, 12, _NPTS_M)
        y = @eval $s.($(Ref(r))[], $(Ref(m))[]', $(Ref(model))[])
    end

    hash_value = (model.context.host.name, model.context.cosmo.name, model.params.ϵ_t, model.params.max_concentration, model.params.z, model.options.tidal_effects)
    JLD2.jldsave(cache_location * string(s)  * "_" * string(hash(hash_value), base=16) * ".jld2" ; r = r, m = m, y = y)

    return true

end


## Possibility to interpolate the model
function _load(model::FSLModel, s::Symbol)

    hash_value = hash((model.context.host.name, model.context.cosmo.name, model.params.ϵ_t, model.params.max_concentration, model.params.z, model.options.tidal_effects))

    !(isdir(cache_location)) && mkdir(cache_location)
    filenames  = readdir(cache_location)
    file       = string(s) * "_" * string(hash_value, base=16) * ".jld2" 

    !(file in filenames) && _save(model, s)

    data    = JLD2.jldopen(cache_location * file)

    r = data["r"]
    m = data["m"]
    y = data["y"]

    if ((model.options.tidal_effects in [:jacobi_disk, :jacobi]) || (s === :min_concentration_DM_only))
        log10_y = Interpolations.interpolate((log10.(r),), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
        return (r::Real, m::Real = 1.0) -> 10.0^log10_y(log10(r))
    else
        log10_y = Interpolations.interpolate((log10.(r), log10.(m)), log10.(y),  Interpolations.Gridded(Interpolations.Linear()))
        return (r::Real, m::Real) -> 10.0^log10_y(log10(r), log10(m))
    end

end

#####################################