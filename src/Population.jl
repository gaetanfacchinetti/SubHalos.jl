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


std_mass_concentration(m::AbstractFloat, ::Type{SCP12}) = 0.14 * log(10.0)
std_mass_concentration(m::AbstractFloat, ::Type{MCM}) where {MCM<:MassConcentrationModel} = std_mass_concentration(m, MCM)


function pdf_concentration(c200::T, m200::T, z::T = T(0), cosmo::Cosmology{T, <:BkgCosmology{T}} = dflt_cosmo(), ::Type{MCM} = SCP12) where {T<:AbstractFloat, MCM<:MassConcentrationModel}
   
    σ_c = std_mass_concentration(m200, MCM)
    median_c = median_concentration(m200, z, cosmo, MCM)

    Kc = 0.5 * SpecialFunctions.erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    return 1.0 / Kc / c200 / sqrt(2.0 * π) / σ_c * exp(-(log(c200) - log(median_c))^2 / 2.0 / σ_c^2)
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

    r = Random.randn(rng, T, n)
    res = Vector{T}(undef, n)

    @inbounds for i in 1:n
        res[i] =  exp(muladd(σ_c, r[i], log_median_c))
    end

    return res
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

    Random.randn!(rng, c200)
    c200 .= exp.(muladd.(σ_c, c200, log_median_c))

end