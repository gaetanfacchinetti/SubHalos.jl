module SubHalos

import QuadGK, Roots

import CosmoTools : median_concentration, SCP12, MassConcentrationModel, gravitational_potential
import CosmoTools : ρ_halo, mΔ, planck18, MPC_TO_KM, orbital_frequency, G_NEWTON, KM_TO_MPC
import HostHalos : HostModel, number_circular_orbits

include("./TidalStripping.jl")


export subhalo_mass_function_template
export mass_function_merger_tree


struct FSLParams
    Mmin::T
    αm::T
    ϵt::T
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

    Kc = 0.5 * erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    return 1.0 / Kc / c200 / sqrt(2.0 * π) / σ_c * exp(-(log(c200) - log(median_c)) / sqrt(2.0) / σ_c^2)
end


#m200 in Msol
pdf_position(r::Real, host::HostModel, cosmo::Cosmology = planck18) = 4 * π  * r^2 * ρ_halo(r, host.halo) / mΔ(host.halo, 200, cosmo.bkg.ρ_c0)
density_virial_mass(m200::Real, host::HostModel, z = 0, cosmo::Cosmology = planck18) = pdf_virial_mass(m200, mΔ(host.halo, 200, cosmo.bkg.ρ_c0), z) 

number_subhalos(host::HostModel, z=0, cosmo::Cosmology = planck18) =  QuadGK.quadgk(lnm -> exp(lnm))


end
