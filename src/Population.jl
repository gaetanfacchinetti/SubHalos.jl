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

export pdf_concentration, rand_concentration, rand_concentration!


########################
## CONCENTRATION DISTRIBUTION

std_mass_concentration(m::T, ::Type{SCP12}) where {T<:AbstractFloat} = T(0.14 * log(10))
std_mass_concentration(m::T, ::Type{MCM}) where {T<:AbstractFloat, MCM<:MassConcentrationModel} = std_mass_concentration(m, MCM)


function pdf_concentration(m200::T, z::T = T(0), cosmo::Cosmology{T, <:BkgCosmology{T}} = dflt_cosmo(T), ::Type{MCM} = SCP12) where {T<:AbstractFloat, MCM<:MassConcentrationModel}
   
    σ_c = std_mass_concentration(m200, MCM)
    median_c = median_concentration(m200, z, cosmo, MCM)

    return LogNormal(log(median_c), σ_c)
end


# draw concentration over a log normal distribution
function rand_concentration(
    n::Int, 
    m200::T, 
    z::T = T(0), 
    cosmo::Cosmology{T, <:BkgCosmology{T}} = dflt_cosmo(), 
    ::Type{MCM} = SCP12, 
    rng::Random.AbstractRNG = Random.default_rng()) where {
        T<:AbstractFloat, 
        MCM<:MassConcentrationModel
        }
    
    σ_c = std_mass_concentration(m200, MCM)
    log_median_c = log(median_concentration(m200, z, cosmo, MCM))

    return rand(LogNormal(log_median_c, σ_c), n, rng)
end


function rand_concentration!(
    c200::AbstractArray{T}, 
    m200::T, 
    z::T = T(0), 
    cosmo::Cosmology{T, <:BkgCosmology{T}} = dflt_cosmo(), 
    ::Type{MCM} = SCP12, 
    rng::Random.AbstractRNG = Random.default_rng()) where {
        T<:AbstractFloat, 
        MCM<:MassConcentrationModel
        }
    
    σ_c = std_mass_concentration(m200, MCM)
    log_median_c = log(median_concentration(m200, z, cosmo, MCM))

    rand!(LogNormal(log_median_c, σ_c), c200, rng)
end


########################
## MASS DISTRIBUTION

abstract type VirialMassDistribution{T<:AbstractFloat} end

struct PowerDistribution{T<:AbstractFloat} <: VirialMassDistribution{T}
    # basic parameters of the model
    α_m::T
    m_min::T
end

struct PowerExpDistribution{T<:AbstractFloat} <: VirialMassDistribution{T}
    γ_1::T
    α_1::T
    γ_2::T
    α_2::T
    β::T
    ζ::T
    m_min::T
end

@doc raw""" 
    subhalo_mass_function_template_MT(x, γ1, α1, γ2, α2, β, ζ)

Template function for the subhalo mass function fitted on merger trees:

``m_Δ^{\rm host} \frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}} = \left(\gamma_1 x^{-\alpha_1} + \gamma_2 x^{-\alpha_2}\right)  e^{-\beta x^\zeta}``

The first argument, `x::Real`, is the ratio of the subhalo over the host mass ``m_Δ^{\rm sub} / m_Δ^{\rm host}.``
"""
function subhalo_mass_function_template_MT(x::T, γ1::T,  α1::T, γ2::T, α2::T, β::T, ζ::T) where {T<:AbstractFloat}
    return (γ1*x^(-α1) + γ2*x^(-α2)) * exp(-β * x^ζ )
end

function unevolved_number_subhalos(mΔ_host::T, params::PowerDistribution{T}) where {T<:AbstractFloat}
    
    _integral(x::Real, γ::Real, α::Real, β::Real, ζ::Real) = γ  / ζ  * ( x^(1.0-α) * SpecialFunctions.expint( (α-1.0)/ζ + 1.0,  β * x^ζ ) - SpecialFunctions.expint( (α-1.0)/ζ + 1.0,  β ))
    return _integral( params.m_min / mΔ_host, params.γ_1, params.α_1, params.β, params.ζ) + _integral(params.m_min / mΔ_host, params.γ_2, params.α_2, params.β, params.ζ)

end


function (f::PowerDistribution)(mΔ_sub::T, mΔ_host::T) where {T<:AbstractFloat}

    if mΔ_sub > f.m_min
        return (f.α_m-1) * mΔ_host^(f.α_m - 1) * 1/( (mΔ_host/f.m_min)^(f.α_m - 1) - 1) * mΔ_sub^(-f.α_m)
    end

    return zero(T)

end

function (f::PowerExpDistribution)(mΔ_sub::T, mΔ_host::T) where {T<:AbstractFloat}

    if mΔ_sub > f.m_min
        return subhalo_mass_function_template_MT(mΔ_sub / mΔ_host, f.γ_1, f.α_1, f.γ_2, f.α_2, f.β, f.ζ) / mΔ_host / unevolved_number_subhalos(mΔ_host, f)f
    end

    return zero(T)
end


pdf_virial_mass(mΔ_sub::T, mΔ_host::T, dist::VirialMassDistribution{T}) where {T<:AbstractFloat} = dist(mΔ_sub, mΔ_host)





struct SubHaloPopulationModel{
    T<:AbstractFloat, 
    VMD<:VirialMassDistribution{T},
    HI<:HostInterpolation{T, <:HaloProfile, <:BulgeModel{T}, <:GasModel{T}, <:GasModel{T}, <:StellarModel{T, <:StellarMassModel{T}}},
    SP<:HaloProfile,
    HPI<:HaloProfileInterpolation{T, SP, <:PseudoMass{T, SP}, <:VelocityDispersion{T, SP}},
    C<:Cosmology{T, <:BkgCosmology{T}}
    } 

    vmd::VMD
    host::HI
    subhalo_profile::HPI
    cosmo::C
    
    m200_host::T
   
    disk::Bool
    stars::Bool
end


function SubHaloPopulationModel(
    vmd::VirialMassDistribution{T}, 
    host::HostModelType{T}, 
    subhalo_profile::HaloProfile,
    cosmo::Cosmology{T, <:BkgCosmology{T}};
    disk::Bool = true,
    stars::Bool = true
    ) where {T<:AbstractFloat}

    # create HaloProfileInterpolation from the HaloProfile
    subhalo_profile_interp = HaloProfileInterpolation(subhalo_profile, T)
    
    # create HostInterpolation from the input host
    host_interp = HostInterpolation(host)

    # precompute the virial mass of the host
    m200 = mΔ(host.halo, T(200), cosmo)

    return SubHaloPopulationModel(vmd, host_interp, subhalo_profile_interp, cosmo, m200, disk, stars)

end


function SubHaloPopulationModel(
    vmd::VirialMassDistribution{T}, 
    host_interp::HostInterpolationType{T}, 
    subhalo_profile_interp::HaloProfileInterpolationType{T},
    cosmo::Cosmology{T, <:BkgCosmology{T}};
    disk::Bool = true,
    stars::Bool = true
    ) where {T<:AbstractFloat}

    # precompute the virial mass of the host
    m200 = mΔ(host_interp.halo, T(200), cosmo)

    return SubHaloPopulationModel(vmd, host_interp, subhalo_profile_interp, cosmo, m200, disk, stars)

end