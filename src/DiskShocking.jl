
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


export adiabatic_correction, compute_disk_shocking_kick!

adiabatic_correction(η::T) where {T<:AbstractFloat} = (1+η^2)^(-3/2)

""" correction factor for adiabatic shocks (Gnedin)"""
function adiabatic_correction(
    x::Vector{T}, 
    xt::T, 
    ρs::T,
    rs::T,
    hd_km::T,
    σz_kms::T,
    halo_profile::HPI, 
    )::Vector{T} where {
        T<:AbstractFloat, 
        P<:HaloProfile{<:Real}, 
        HPI<:HaloProfileInterpolationType{T, P}
        }
    
    td = hd_km / σz_kms # in s
    typical_velocity_kms = sqrt(4 * π * ρs * rs^2 / convert_lengths(MegaParsecs, KiloMeters, T) * constant_G_NEWTON(KiloMeters, Msun, Seconds, T)) # in km / s
    v_disp = typical_velocity_kms * halo_profile.velocity_dispersion.(log10.(x), log10(xt)) # km / s
    ωd = v_disp ./ (x * convert_lengths(rs, MegaParsecs, KiloMeters))  # in 1 / s
       
    return adiabatic_correction.(td * ωd)
end


function compute_disk_shocking_kick!(
    Δv::Array{T, 4},
    x::Vector{T}, 
    ψ::Vector{T}, 
    φ::Vector{T},
    r_host::T,
    v_kms::T,
    xt::T, 
    θ::T, 
    subhalo::H, 
    halo_profile::HPI, 
    host::HI
    ) where {
        T<:AbstractFloat,
        P<:HaloProfile{<:Real},
        H<:Halo{T, P}, 
        HPI<:HaloProfileInterpolationType{T, P}, 
        HI<:HostInterpolationType{T}
    }

    rs = subhalo.rs

    if v_kms == 0
        # first we need to make sure the problem is not comming from interpolations
        # around the edges of the host (in which case, there is no baryons anymore
        # and we can simply return 0.0
        (r_host >= T(0.98) * host.rt) && (Δv .= T(0))
        return nothing
    end

    # gd -> disc acceleration in km / s^2
    disc_acceleration = 2 * π * constant_G_NEWTON(MegaParsecs, Msun, GigaYears, T) * σ_baryons(r_host, host) * convert_lengths(MegaParsecs, KiloMeters, T) / convert_times(GigaYears, Seconds, T)^2 # in km / s^2
    
    # typical disk shocking velocity kick in km / s
    δv_ds = 2 .* disc_acceleration .* convert_lengths(rs, MegaParsecs, KiloMeters) / sqrt(3) / v_kms
    
    # adiabatic correction for fast rotating particles
    σz_kms = T(sqrt(2 / π)) * velocity_dispersion_spherical_kms(r_host, host) # in km/s
    hd_km = convert_lengths(host.stars.thick_zd, MegaParsecs, KiloMeters) # in km
    adiab_corr = sqrt.(adiabatic_correction(x, xt, subhalo.ρs, rs, hd_km, σz_kms, halo_profile))
    
    # precomputation of some trigonometric functions
    sθ = sin(θ)
    cθ = cos(θ)
    cφ = cos.(φ)

    # prefactor of the entire expression
    pref =  δv_ds  * (cos.(θ .+ ψ) .+ sin(θ) * sin.(ψ) .*  (1 .- cφ')) / abs(cθ)
    
    # loop over the values of x, broadcasting over ψ and φ
    @inbounds for i ∈ 1:length(x)
        val = pref .* x[i] .* adiab_corr[i]
        Δv[i, :, :, 1] .= val * cθ
        Δv[i, :, :, 2] .= -val * sθ
    end

    return nothing

end
