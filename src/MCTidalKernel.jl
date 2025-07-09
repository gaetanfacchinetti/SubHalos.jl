export OneCrossingArrays, allocate_one_crossing, inv_μ, tidal_scale_one_crossing!

## Main function that compute the tidal scale for one crossing of the disk

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

    dlog10x::Vector{T}
    dcψ::Vector{T}
    dφ::Vector{T}

    rand_vec::Vector{T}

    m_stars::Vector{T}
    β::Vector{T} 
    β2::Vector{T}
    β_cα::Vector{T}
    β_sα::Vector{T}   
    mpm_cα::Vector{T}
    mpm_sα::Vector{T}
    Δv::Array{T, 3}
    
    σ_sub_kms::Vector{T} 
    ve_kms::Vector{T}
    frac::Array{T, 3}
end



function allocate_one_crossing(n_stars::Int, nx::Int = 15, nψ::Int = 10, nφ::Int = 10,  ::Type{T} = Float64) where {T<:AbstractFloat}

    γ = T(0.5)
    nx_over_2 = floor(Int, nx/2)

    log10_x_over_xt = T.(vcat(range(-5, log10(T(0.2)), nx_over_2), log10.(range((T(0.2))^(1/γ), T(0.99999), nx_over_2+1)[2:end].^γ)))
    
    ψ = collect(range(T(0), T(π), nψ+1))[1:end-1]
    φ = collect(range(T(0), T(2*π), nφ+1))[1:end-1]
    
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


    α = Vector{T}(undef, n_stars)

    # β, m_stars, mpm, sα, cα (in this order)
    #encounter_properties = Matrix{T}(undef, 5, n_stars)
    m_stars = Vector{T}(undef, n_stars)
    β       = Vector{T}(undef, n_stars)
    β2      = Vector{T}(undef, n_stars)
    β_sα    = Vector{T}(undef, n_stars)
    β_cα    = Vector{T}(undef, n_stars)
    mpm_sα  = Vector{T}(undef, n_stars)
    mpm_cα  = Vector{T}(undef, n_stars)
    

    Δv    = Array{T, 3}(undef, nx, nψ, nφ)
    
    σ_sub_kms = Vector{T}(undef, nx)
    ve_kms    = Vector{T}(undef, nx)
    frac      = Array{T, 3}(undef, nx, nψ, nφ)

    return OneCrossingArrays(log10_x_over_xt, ψ, φ, α, sψ, cψ, sψ2, cψ2, sφ, cφ, sφ2, cφ2,
        dlog10x, dcψ, dφ, rand_vec, m_stars, β, β2, β_cα, β_sα, mpm_cα, mpm_sα, Δv, σ_sub_kms, ve_kms, frac)

end


function inv_μ(μ::T)::T where {T<:AbstractFloat}
    
    if μ < T(4e-7)
        return sqrt(T(2)*μ)
    end

    if μ > T(7.1)
        return exp(y+1)-1
    end

    try
        return exp.(Roots.find_zero(lnx->(μ_halo(exp(lnx), nfwProfile)- μ), (log(T(1e-2)), log(T(1e+10))), Roots.Bisection(), rtol=T(1e-8)))
    catch
        if μ_halo(exp(T(log(1e-2))), nfwProfile)- μ > 0
            return T(0.999e-2)
        end
        #println("f1 = $(μ_halo(exp(log(1e-2)), nfwProfile)- μ), f2 = $(μ_halo(exp(log(1e+10)), nfwProfile)- μ)")
        rethrow()
    end
end

function fraction_particles(v::T, ve::T, σ::T)::T where {T<:AbstractFloat}

    y = ve/σ/sqrt(2)
    ex = exp(-y^2)
    
    num = ex * v - sqrt(π/2) * σ * SpecialFunctions.erf(v/sqrt(2)/σ)
    denom = 2 * (ex * ve - sqrt(π/2) * σ * SpecialFunctions.erf(y) )

    return num/denom

end

""" correction factor for adiabatic shocks (Gnedin)"""
function adiabatic_correction(
    x::T, 
    xt::T, 
    ρs::T,
    rs::T,
    hd_km::T,
    σz_kms::T,
    halo_profile::HPI, 
    )::T where {
        T<:AbstractFloat, 
        P<:HaloProfile{<:Real}, 
        HPI<:HaloProfileInterpolationType{T, P}
        }
    
    td = hd_km / σz_kms # in s
    typical_velocity_kms = sqrt(4 * π * ρs * rs^2 / convert_lengths(MegaParsecs, KiloMeters, T) * constant_G_NEWTON(KiloMeters, Msun, Seconds, T)) # in km / s
    v_disp = typical_velocity_kms * halo_profile.velocity_dispersion(log10(x), log10(xt)) # km / s
    ωd = v_disp / (x * convert_lengths(rs, MegaParsecs, KiloMeters))  # in 1 / s
           
    return (1+(td * ωd)^2)^(-3/2)
end


function tidal_scale_one_crossing!(
    xt::T,
    r_host::T,
    v_subhalo_kms::Vector{T},
    subhalo::H,
    hpi::HPI,
    host::HI,
    n_stars::Int,
    tables::OneCrossingArrays{T},
    disk::Bool = true,
    stars::Bool = true
    ) where {
        T<:AbstractFloat,
        H<:HaloType{T},
        HPI<:HaloProfileInterpolationType{T},
        HI<:HostInterpolationType{T}
    }

    ######## 
    # INITIALISATION

    log10_xt = log10(xt)
    log10_x = @. tables.log10_x_over_xt + log10_xt
    x = exp10.(log10_x)
    x2 = @. x^2

    n_penetrative = 0

    # populate the entry of the tables with 
    # drawn realisations if stellar encounters
    # are taken into account
    if stars

        # Random sampling
        rand_vec = view(tables.rand_vec, 1:3*n_stars)
        Random.rand!(rand_vec)

        β_max = maximum_impact_parameter(r_host, host) / subhalo.rs
        
        @views begin

            @. tables.β[1:n_stars]      = sqrt(rand_vec[1:n_stars]) * β_max
            @. tables.α[1:n_stars]      = 2 * π * rand_vec[n_stars+1:2*n_stars]
            @. tables.β2[1:n_stars]     = tables.β[1:n_stars]^2
            @. tables.β_cα[1:n_stars]   = tables.β[1:n_stars] * cos(tables.α[1:n_stars])
            @. tables.β_sα[1:n_stars]   = tables.β[1:n_stars] * sin(tables.α[1:n_stars])
            @. tables.mpm_cα[1:n_stars] = hpi.pseudo_mass(log10.(tables.β[1:n_stars]), log10_xt) * cos(tables.α[1:n_stars])
            @. tables.mpm_sα[1:n_stars] = hpi.pseudo_mass(log10.(tables.β[1:n_stars]), log10_xt) * sin(tables.α[1:n_stars])
            
            rand_stellar_mass!(tables.m_stars[1:n_stars], rand_vec[2*n_stars+1:end], host())
        end

        n_penetrative = sum(tables.β[1:n_stars] .< xt)

    end

    # Velocities in the basis (ex, ey, ez)
    norm_v_subhalo_kms = LinearAlgebra.norm(v_subhalo_kms)
    v_star_kms = circular_velocity_kms(r_host, host)
    v_rel_kms = similar(v_subhalo_kms)
    @. v_rel_kms = v_subhalo_kms
    v_rel_kms[2] -= v_star_kms
    norm_v_rel_kms = LinearAlgebra.norm(v_rel_kms)

    σ_prefactor =  sqrt(T(4) * π * subhalo.ρs * subhalo.rs^2 / convert_lengths(MegaParsecs, KiloMeters, T) * constant_G_NEWTON(KiloMeters, Msun, Seconds, T)) 

    @views for i in eachindex(x)
        tables.σ_sub_kms[i] = σ_prefactor * hpi.velocity_dispersion(log10_x[i], log10_xt)
        tables.ve_kms[i] = escape_velocity_kms(x[i] * subhalo.rs, xt * subhalo.rs, subhalo)
    end


    # constants for stellar encounters
    δv0_se = stars ? 2 * constant_G_NEWTON(KiloMeters, Msun, Seconds, T) / convert_lengths(subhalo.rs, MegaParsecs, KiloMeters) / norm_v_rel_kms : zero(T)
   
    # constants for disk shocking
    disc_acceleration = disk ? 2 * π * constant_G_NEWTON(MegaParsecs, Msun, GigaYears, T) * σ_baryons(r_host, host) * convert_lengths(MegaParsecs, KiloMeters, T) / convert_times(GigaYears, Seconds, T)^2 : zero(T) # in km / s^2
    δv0_ds = disk ? 2 .* disc_acceleration .* convert_lengths(subhalo.rs, MegaParsecs, KiloMeters) / sqrt(3) / (- v_subhalo_kms[3]) : zero(T)
    σz_kms = disk ? T(sqrt(2 / π)) * velocity_dispersion_spherical_kms(r_host, host) : zero(T) # in km/s
    hd_km = disk ? convert_lengths(host.stars.thick_zd, MegaParsecs, KiloMeters) : zero(T) # in km
    

    ######## 
    # CONSTRUCT THE ROTATION MATRICES

    rotation_eps_n = Matrix{T}(undef, 3, 3)
    rotation_e_n   = Matrix{T}(undef, 3, 3)

    u = v_subhalo_kms ./ norm_v_subhalo_kms
    ν = v_star_kms / norm_v_subhalo_kms

    # angles and trigonometry
    sθ = sqrt(1-u[3]^2)
    cθ = u[3]
    cλ = u[1] / sθ
    sλ = u[2] / sθ
    cω = cθ / sqrt(1 + ν^2 - 2*ν * sθ * sλ)
    sω = sqrt(1-cω^2)
    cτ = sθ * cλ / sqrt(sθ^2 + ν^2 - 2*ν * sθ * sλ)
    sτ = (sθ * sλ - ν) / sqrt(sθ^2 + ν^2 - 2*ν * sθ * sλ)

    # rotation from the basis (nx, ny, nz) to (ex, ey, ez)
    # Ri   .= Ri1z * Ri2y * Ri3z
    rotation_e_n_1z::Matrix{T} = [cλ -sλ 0; sλ cλ 0; 0 0 1]
    rotation_e_n_2y::Matrix{T} = [cθ 0 sθ; 0 1 0; -sθ 0 cθ]
    rotation_e_n .= rotation_e_n_1z * rotation_e_n_2y

    # rotation from the basis (ex, ey, ez) to (epsx, epxy, epsz)
    # N   .=  N2y * N1z
    rotation_eps_e_1z::Matrix{T} = [cτ sτ 0; -sτ cτ 0; 0 0 1]
    rotation_eps_e_2y::Matrix{T} = [cω 0 -sω; 0 1 0; sω 0 cω]

    # rotation from the basis (nx, ny, nz) to (epsx, epsy, epsz)
    rotation_eps_n .= rotation_eps_e_2y * rotation_eps_e_1z * rotation_e_n 

    # decompose the rotation matrices for faster memory access
    r_eps_n_11 = rotation_eps_n[1, 1]
    r_eps_n_12 = rotation_eps_n[1, 2]
    r_eps_n_13 = rotation_eps_n[1, 3]
    r_eps_n_21 = rotation_eps_n[2, 1]
    r_eps_n_22 = rotation_eps_n[2, 2]
    r_eps_n_23 = rotation_eps_n[2, 3]
    r_eps_n_31 = rotation_eps_n[3, 1]
    r_eps_n_32 = rotation_eps_n[3, 2]
    r_eps_n_33 = rotation_eps_n[3, 3]

    r_e_n_33 = rotation_e_n[3, 3]
    r_e_n_31 = rotation_e_n[3, 1]
    r_e_n_32 = rotation_e_n[3, 2]

    
    ######## 
    # MAIN LOOP OVER POSITIONS AND STELLAR ENCOUNTERS

    ni, nj, nk = size(tables.Δv)
    frac_average_angle = similar(x)
  
    # loop over the radius x
    @inbounds for i ∈ 1:ni
        
        frac_average_angle[i] = zero(T)
        xi = x[i]
        x2i = x2[i]
        ve_i = tables.ve_kms[i]
        σ_i = tables.σ_sub_kms[i]

        # adiabatic correction factor if computing disk shocking
        adiab_corr = disk ? sqrt(adiabatic_correction(xi, xt, subhalo.ρs, subhalo.rs, hd_km, σz_kms, hpi)) : zero(T) # may need to improve this for memory allocation

        # loop over the angles
        @inbounds for j ∈ 1:nj, k ∈ 1:nk

            sum1 = zero(T)
            sum2 = zero(T)

            Δv_se_1 = zero(T)
            Δv_se_2 = zero(T)
            Δv_se_3 = zero(T)
            
            Δv_ds_1 = zero(T)
            Δv_ds_2 = zero(T)
            Δv_ds_3 = zero(T)

            cψ, sψ = tables.cψ[j], tables.sψ[j]
            cφ, sφ = tables.cφ[k], tables.sφ[k]
            

            # only perform the heavy computation for stellar encounter
            if stars

                # vector position in the basis (epsx, epsy, epsz)
                # the "natural" basis for computations with stellar encounters        
                x_epsx = xi * (r_eps_n_11 * cψ + sψ * ( r_eps_n_12 * cφ + r_eps_n_13 * sφ))
                x_epsy = xi * (r_eps_n_21 * cψ + sψ * ( r_eps_n_22 * cφ + r_eps_n_23 * sφ))
                x_epsz = xi * (r_eps_n_31 * cψ + sψ * ( r_eps_n_32 * cφ + r_eps_n_33 * sφ))
                
                x2_reps2 = x2i - x_epsz^2
                
                # loop over the encounters
                # most expensive part of the code
                # need to be made as efficient as possible
                @inbounds @simd for l ∈ 1:n_stars

                    # access all relavant and precomputed variables
                    β2      = tables.β2[l]
                    β_cα    = tables.β_cα[l]
                    β_sα    = tables.β_sα[l]
                    m_star  = tables.m_stars[l]
                    mpm_cα  = tables.mpm_cα[l] 
                    mpm_sα  = tables.mpm_sα[l]

                    one_over_denom = 1 / (x2_reps2 + β2 + 2 * (x_epsx * β_cα  + x_epsy * β_sα) )

                    # each component along epsx and epsy
                    sum1 += m_star * ( - (x_epsx + β_cα) * one_over_denom  + mpm_cα )
                    sum2 += m_star * ( - (x_epsy + β_sα) * one_over_denom  + mpm_sα )

                end

                # change frame with the appropriate rotation matrix 
                # use the transpose of rotation_eps_n here as we go from (eps_x, eps_y, eps_z) to (n_x, n_y, n_z)
                Δv_se_1 = δv0_se * (r_eps_n_11 * sum1 + r_eps_n_21 * sum2)
                Δv_se_2 = δv0_se * (r_eps_n_12 * sum1 + r_eps_n_22 * sum2)
                Δv_se_3 = δv0_se * (r_eps_n_13 * sum1 + r_eps_n_23 * sum2)

            end


            # add the contribution of disk shocking
            if disk
                
                # disk shocking along ez "natural" direction for disk shocking
                # last part is the decompostion along the z axis of the vector x (or r)
                val_3 = δv0_ds * adiab_corr * xi * (r_e_n_31 * cψ + sψ * (r_e_n_32 * cφ + r_e_n_33 * sφ))

                # change frame with the appropriate rotation matrix
                # use the transpose of rotation_e_n here as we go from (e_x, e_y, e_z) to (n_x, n_y, n_z)
                Δv_ds_1 = r_e_n_31 * val_3
                Δv_ds_2 = r_e_n_32 * val_3
                Δv_ds_3 = r_e_n_33 * val_3  

            end

            # sum the two contributions (stars and disk)
            Δv1 = Δv_se_1 + Δv_ds_1
            Δv2 = Δv_se_2 + Δv_ds_2
            Δv3 = Δv_se_3 + Δv_ds_3
        
            tables.Δv[i, j, k]  = sqrt(Δv1^2 + Δv2^2 + Δv3^2)
            tables.frac[i, j, k] = fraction_particles(max(-ve_i, ve_i - tables.Δv[i, j, k]), ve_i, σ_i) - fraction_particles(-ve_i, ve_i, σ_i)
        end

        #println("i = $i -> $(tables.Δv[i, 5, 5]), $(tables.frac[i, 5, 5])")

        for j in 1:nj-1, k in 1:nk-1
            frac_average_angle[i] += -T(0.25) * (
                tables.frac[i, j+1, k+1] + tables.frac[i, j+1, k] +
                tables.frac[i, j, k+1] + tables.frac[i, j, k]
            ) * tables.dcψ[j] * tables.dφ[k] / (4 * π)
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
    
    return T(inv_μ(μt)), μt, n_penetrative
end