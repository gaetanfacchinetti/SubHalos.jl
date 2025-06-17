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

export OneCrossingArrays, allocate_one_crossing, inv_μ, tidal_scale_one_crossing!, tidal_scale

# structure holding all arrays used for one crossing computation
struct OneCrossingArrays{T<:AbstractFloat}
    log10_x_over_xt::Vector{T}
    
    ψ::Vector{T}
    φ::Vector{T}
    α::Vector{T}
    
    sψ::Vector{T}
    cψ::Vector{T}
    sψ2::Vector{T}
    cψ2::Vector{T}

    sφ::Vector{T}
    cφ::Vector{T}
    sφ2::Vector{T}
    cφ2::Vector{T}

    sα::Vector{T}
    cα::Vector{T}

    dlog10x::Vector{T}
    dcψ::Vector{T}
    dφ::Vector{T}

    rand_vec::Vector{T}

    m_stars::Vector{T}
    β::Vector{T}    
    mpm::Vector{T}
    Δv_ds::Array{T, 4}
    Δv_se::Array{T, 4}
    Δv::Array{T, 3}
    
    σ_sub_kms::Vector{T} 
    ve_kms::Vector{T}
    frac::Array{T, 3}
end


function inv_μ(μ::T)::T where {T<:AbstractFloat}
    
    if μ < T(4e-7)
        return sqrt(T(2)*μ)
    end

    if μ > T(7.1)
        return exp(y+1)-1
    end

    try
        return exp.(Roots.find_zero(lnx->(μ_halo(exp(lnx), nfwProfile)- μ), (log(T(1e-2)), log(T(1e+10))), Roots.Bisection(), rtol=T(1e-5)))
    catch
        if μ_halo(exp(T(log(1e-2))), nfwProfile)- μ > 0
            return T(0.999e-2)
        end
        #println("f1 = $(μ_halo(exp(log(1e-2)), nfwProfile)- μ), f2 = $(μ_halo(exp(log(1e+10)), nfwProfile)- μ)")
        rethrow()
    end
end


function allocate_one_crossing(n_stars::Int, nx::Int = 15, nψ::Int = 10, nφ::Int = 10,  ::Type{T} = Float64) where {T<:AbstractFloat}

    γ = T(0.5)
    nx_over_2 = floor(Int, nx/2)

    log10_x_over_xt = T.(vcat(range(-5, log10(T(0.2)), nx_over_2), log10.(range((T(0.2))^(1/γ), T(0.99999), nx_over_2+1)[2:end].^γ)))
    
    ψ = collect(range(T(0), T(π), nψ))
    φ = collect(range(T(0), T(2*π), nφ))
    
    sψ = sin.(ψ)
    cψ = cos.(ψ)
    sψ2 = sψ.^2
    cψ2 = cψ.^2

    sφ = sin.(φ)
    cφ = cos.(φ)
    sφ2 = sφ.^2
    cφ2 = cφ.^2

    dlog10x = [log10_x_over_xt[i+1] - log10_x_over_xt[i] for i in 1:nx-1]
    dcψ = [cψ[i+1] - cψ[i] for i in 1:nψ-1]
    dφ = [φ[i+1] - φ[i] for i in 1:nφ-1]

    rand_vec  = Vector{T}(undef, 3 * n_stars)

    m_stars = Vector{T}(undef, n_stars)
    β       = Vector{T}(undef, n_stars)
    α       = Vector{T}(undef, n_stars)
    sα      = Vector{T}(undef, n_stars)
    cα      = Vector{T}(undef, n_stars)
    mpm     = Vector{T}(undef, n_stars)
    
    Δv_se = Array{T, 4}(undef, nx, nψ, nφ, 3)
    Δv_ds = Array{T, 4}(undef, nx, nψ, nφ, 2)

    Δv    = Array{T, 3}(undef, nx, nψ, nφ)
    
    σ_sub_kms = Vector{T}(undef, nx)
    ve_kms    = Vector{T}(undef, nx)
    frac      = Array{T, 3}(undef, nx, nψ, nφ)

    return OneCrossingArrays(log10_x_over_xt, ψ, φ, α, sψ, cψ, sψ2, cψ2, sφ, cφ, sφ2, cφ2, sα, cα, dlog10x, dcψ, dφ, rand_vec, m_stars, β, mpm, Δv_ds, Δv_se, Δv, σ_sub_kms, ve_kms, frac)

end

function fraction_particles(v::T, ve::T, σ::T)::T where {T<:AbstractFloat}

    y = ve/σ/sqrt(2)
    ex = exp(-y^2)
    
    num = ex * v - sqrt(π/2) * σ * SpecialFunctions.erf(v/sqrt(2)/σ)
    denom = 2 * (ex * ve - sqrt(π/2) * σ * SpecialFunctions.erf(y) )

    return num/denom

end


function tidal_scale_one_crossing!(
    xt::T,
    r_host::T,
    v_subhalo_kms::Vector{T},
    subhalo::H,
    hpi::HPI,
    host::HI,
    n_stars::Int,
    tables::OneCrossingArrays{T}
) where {
    T<:AbstractFloat,
    H<:HaloType{T},
    HPI<:HaloProfileInterpolationType{T},
    HI<:HostInterpolationType{T}
}
    log10_xt = log10(xt)
    log10_x = @. tables.log10_x_over_xt + log10_xt
    x = exp10.(log10_x)
    x2 = @. x^2

    # Random sampling
    rand_vec = tables.rand_vec
    Random.rand!(rand_vec)

    β_max = maximum_impact_parameter(r_host, host) / subhalo.rs
    
    @views begin
        tables.β .= @. sqrt(rand_vec[1:n_stars]) * β_max
        tables.α .= @. 2 * π * rand_vec[n_stars+1:2*n_stars]
        tables.sα .= sin.(tables.α)
        tables.cα .= cos.(tables.α)
        rand_stellar_mass!(tables.m_stars, rand_vec[2n_stars+1:end], host())
        tables.mpm .= hpi.pseudo_mass.(log10.(tables.β), log10_xt)
    end


    # Velocities
    norm_v_subhalo_kms = LinearAlgebra.norm(v_subhalo_kms)
    θ = acos(v_subhalo_kms[3] / norm_v_subhalo_kms)
    v_star_kms = circular_velocity_kms(r_host, host)
    v_rel_kms = similar(v_subhalo_kms)
    @. v_rel_kms = v_subhalo_kms
    v_rel_kms[2] -= v_star_kms
    norm_v_rel_kms = LinearAlgebra.norm(v_rel_kms)

    δv0 = 2 * constant_G_NEWTON(KiloMeters, Msun, Seconds, T) / (subhalo.rs * MPC_TO_KM) / norm_v_rel_kms

    # Disk shocking
    compute_disk_shocking_kick!(tables.Δv_ds, x, tables.ψ, tables.φ, r_host, norm_v_subhalo_kms, xt, θ, subhalo, hpi, host)

    # Constants
    η_subhalo_2 = (norm_v_subhalo_kms / norm_v_rel_kms)^2
    η_star_2 = (v_star_kms / norm_v_rel_kms)^2
    η_subhalo_stars = norm_v_subhalo_kms * v_star_kms / norm_v_rel_kms^2
    γ_subhalo_star = 1 / sqrt(1 + (norm_v_subhalo_kms / v_star_kms)^2)
    γ_star_subhalo = 1 / sqrt(1 + (v_star_kms / norm_v_subhalo_kms)^2)

    σ_prefactor = sqrt(T(4) * π * subhalo.ρs * subhalo.rs^2 / convert_lengths(MegaParsecs, KiloMeters, T) * constant_G_NEWTON(KiloMeters, Msun, Seconds, T))
    
    @views for i in eachindex(x)
        log10_xi = log10_x[i]
        xi = x[i]
        tables.σ_sub_kms[i] = σ_prefactor * hpi.velocity_dispersion(log10_xi, log10_xt)
        tables.ve_kms[i] = escape_velocity_kms(xi * subhalo.rs, xt * subhalo.rs, subhalo)
    end

    ni, nj, nk = size(tables.Δv_se)
    frac_average_angle = similar(x)

    
    @inbounds for i in 1:ni
        
        frac_average_angle[i] = zero(T)
        xi = x[i]
        x2i = x2[i]
        ve_i = tables.ve_kms[i]
        σ_i = tables.σ_sub_kms[i]

        @inbounds for j in 1:nj, k in 1:nk

            cψ, sψ = tables.cψ[j], tables.sψ[j]
            cψ2, sψ2 = tables.cψ2[j], tables.sψ2[j]
            cφ, sφ = tables.cφ[k], tables.sφ[k]
            sφ2 = tables.sφ2[k]

            xev2 = x2i * (cψ2 * η_subhalo_2 + sψ2 * sφ2 * η_star_2 - 2 * cψ * sψ * sφ * η_subhalo_stars)

            sum1 = zero(T); sum2 = zero(T); sum3 = zero(T)

            pref1_x = xi * (cψ * (η_subhalo_2 - 1) - sψ * sφ * η_subhalo_stars)
            pref2_x = - xi * sψ * cφ 
            pref3_x = xi * (sψ * sφ * (η_star_2 - 1) - cψ * η_subhalo_stars)

            @inbounds @simd for l in 1:n_stars

                β, m_star, mpm, cα, sα = tables.β[l], tables.m_stars[l], tables.mpm[l], tables.cα[l], tables.sα[l]

                one_over_denom = 1 / (x2i + β^2 - xev2 + 2 * xi * β * (sψ * cφ * cα + sψ * sφ * sα * γ_star_subhalo + cψ * cα * γ_subhalo_star))

                # each component along nv, n1, and n2
                sum1 += m_star * ( (pref1_x - β * sα * γ_subhalo_star) * one_over_denom + mpm * sα * γ_subhalo_star )
                sum2 += m_star * ( (pref2_x - β * cα) * one_over_denom + mpm * cα)
                sum3 += m_star * ( (pref3_x - β * sα * γ_star_subhalo) * one_over_denom + mpm * sα * γ_star_subhalo )
            
            end

            Δv_se = view(tables.Δv_se, i, j, k, :)
            Δv_ds = view(tables.Δv_ds, i, j, k, :)
            
            Δv_se[1] = δv0 * sum1
            Δv_se[2] = δv0 * sum2
            Δv_se[3] = δv0 * sum3

            Δv1 = Δv_se[1] + Δv_ds[1]
            Δv2 = Δv_se[2] + Δv_ds[2]
            Δv3 = Δv_se[3]

            tables.Δv[i, j, k]   = sqrt(Δv1^2 + Δv2^2 + Δv3^2)
            tables.frac[i, j, k] = fraction_particles(max(-ve_i, ve_i - tables.Δv[i, j, k]), ve_i, σ_i) - fraction_particles(-ve_i, ve_i, σ_i)
        end

        for j in 1:nj-1, k in 1:nk-1
            frac_average_angle[i] += -T(0.25) * (
                tables.frac[i, j+1, k+1] + tables.frac[i, j+1, k] +
                tables.frac[i, j, k+1] + tables.frac[i, j, k]
            ) * tables.dcψ[j] * tables.dφ[k] / (4π)
        end
    end

    
    frac_average_angle .= min.(one(T), frac_average_angle)

    μt = zero(T)
    for i in 1:ni-1
        ρ1 = ρ_halo(x[i+1], subhalo.hp)
        ρ2 = ρ_halo(x[i], subhalo.hp)
        x1 = x[i+1]; x2_ = x[i]
        μt += T(0.5) * (
            frac_average_angle[i+1] * ρ1 * x1^3 +
            frac_average_angle[i]   * ρ2 * x2_^3
        ) * tables.dlog10x[i] * log(T(10))
    end


    μt = min(μt, μ_halo(xt, subhalo.hp))
    #println(μt, " ", μ_halo(xt, subhalo.hp))
    
    return T(inv_μ(μt))
end


function tidal_scale( 
    xt::T,
    r_host::T,
    subhalo::H, 
    hpi::HPI,
    host::HI,
    n_stars::Int;
    z::T = T(0),
    cosmo::C = dflt_cosmo(T)
    ) where {
        T<:AbstractFloat,
        H<:HaloType{T}, 
        HPI<:HaloProfileInterpolationType{T}, 
        HI<:HostInterpolationType{T},
        C<:Cosmology{T, <:BkgCosmology{T}}}

    arrays = allocate_one_crossing(n_stars, 20, 10, 10, T)
    n_cross = 2 * number_circular_orbits(r_host, host, z, cosmo.bkg)
    v_subhalo_kms = rand_3D_velocity_kms(1, r_host, host)[:, 1]

    norm_v_subhalo_kms = LinearAlgebra.norm(v_subhalo_kms)
    θ = acos(v_subhalo_kms[3] / norm_v_subhalo_kms)

    xt_array = Vector{T}(undef, 0)

    append!(xt_array, tidal_scale_one_crossing!(xt, r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays))
    (xt_array[end] <= T(1e-2)) && (return xt_array, θ, norm_v_subhalo_kms)
    
    for i in 2:n_cross
        v_subhalo_kms .= - v_subhalo_kms
        append!(xt_array, tidal_scale_one_crossing!(xt_array[end], r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays))
        (xt_array[end] <= T(1e-2)) && (return xt_array, θ, norm_v_subhalo_kms)
    end

    return xt_array, θ, norm_v_subhalo_kms
end