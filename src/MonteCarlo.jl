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


#######################
## STAR PROPERTIES

export w_parallel, w_perp, cdf_η, inverse_cdf_η, average_inverse_relative_speed, average_inverse_relative_speed_sqr
export draw_velocity_kick_complex, VelocityKickDraw, draw_velocity_kick, draw_parameter_B, ccdf_ΔE, ccdf_ΔE_CL
export mean_velocity_kick_approx_sqr, median_ΔE_CL_approx, ccdf_ΔE_CL_approx
export Δv2, mean_ΔE, reduced_Δv2, reduced_mean_ΔE, median_ΔE
export _save_inverse_cdf_η, _load_inverse_cdf_η

average_inverse_relative_speed(σ::T, v_star::T) where {T<:AbstractFloat} = SpecialFunctions.erf(v_star/(sqrt(2.0) * σ))/v_star
average_inverse_relative_speed_sqr(σ::T, v_star::T) where {T<:AbstractFloat} = sqrt(2.0)* SpecialFunctions.dawson(v_star/(sqrt(2.0) * σ)) / (σ * v_star)


function cdf_η(η::T, σ::T, v_star::T, mstar_avg::T, v_avg::T, host::HM) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}}
    
    x = v_star / (sqrt(2.0) * σ)
    
    function integrand(m::T) where {T<:AbstractFloat}
        y = v_avg / (sqrt(2.0) * σ) * m / mstar_avg / η
        res = (exp(-(x+y)^2)*(-1 + exp(4.0*x*y))/sqrt(π)/x + SpecialFunctions.erfc(y-x) + SpecialFunctions.erfc(x+y))/2.0
        (res === NaN) && return 0.0
        return res
    end

    return QuadGK.quadgk(lnm -> integrand(exp(lnm)) * stellar_mass_function(exp(lnm), host) * exp(lnm), log(1e-7), log(10.0^1.8), rtol=1e-10)[1]
end


function inverse_cdf_η(rnd::T, σ::T, v_star::T, mstar_avg::T, v_avg::T, host::HM) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}} 
    return exp(Roots.find_zero(lnu -> cdf_η(exp(lnu), σ, v_star, mstar_avg, v_avg, host) - rnd, (log(1e-8), log(1e+6)), Roots.Bisection(), rtol=1e-10, atol=1e-10)) 
end


""" β = b / rs,  \vec{β} = β * (cos(α) n_1 + sin(α) n_2 ), xp = x_ψ = x sin(ψ)"""
w_parallel(xp::T, α::T, β::T, xt::T, shp::HaloProfile = nfwProfile) where {T<:AbstractFloat} = (pseudo_mass_I(β, xt, shp) * sin(α) - (xp * β + β^2 * sin(α))/(xp^2 + β^2 + 2*xp*β*sin(α)) )
w_perp(xp::T, α::T, β::T, xt::T, shp::HaloProfile = nfwProfile) where {T<:AbstractFloat} = (pseudo_mass_I(β, xt, shp) * cos(α) - (β^2 * cos(α))/(xp^2 + β^2 + 2*xp*β*sin(α)) )

function w(xp::T, α::T, β::T, xt::T, shp::HaloProfile = nfwProfile) where {T<:AbstractFloat}
    
    sα   = sin(α)
    cα   = cos(α)
    denom = (xp^2 + β^2 + 2*xp*β*sα)
    pm    = pseudo_mass_I(β, xt, shp)
  
    (pm * sα - (xp * β + β^2 * sα)/denom), (pm * cα - (β^2 * cα)/denom) 
end


function draw_velocity_kick(rp::T, subhalo::H, r_host::T, host::HM; use_tables::Bool = true) where {T<:AbstractFloat, U<:Real, S<:Real, H<:Halo{T, S}, HM<:HostModel{T, U}} 
    
    # initialisation for a given value of r
    n     = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    b_m   = use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)
    inv_η = _load_inverse_cdf_η(r_host, T)

    rt = jacobi_radius(r_host, subhalo, host)
    rs = subhalo.rs

    (rp > rt) && return false

    # Randomly sampling the distributions
    θb = T(2) * π * rand(n, T)
    β  = sqrt.(rand(n, T))
    η  = inv_η.(rand(n, T))

    v_parallel = w_parallel.(rp / rs, θb, b_m * β / rs, rt / rs, subhalo.hp) .* η ./ β # assuming b_min = 0 here
    v_perp     = w_perp.(rp / rs, θb, b_m * β / rs, rt / rs, subhalo.hp) .* η ./ β     # assuming b_min = 0 here
    
    return v_parallel, v_perp
end


function draw_parameter_B(r_host::T, host::HM, θ::T, nstars::Int; use_tables::Bool = true) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}}  
    
    σ         = (use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)) / T(MPC_TO_KM)
    v_star    = (use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)) / T(MPC_TO_KM)
    n_stars   = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    
    return draw_parameter_B(r_host, σ, v_star, host, n_stars; use_tables = use_tables)
end

function draw_parameter_B(r_host::T, σ::T, v_star::T, host::HM, n_stars::Int; use_tables::Bool = true) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}} 
    
    # initialisation for a given value of r
    b_m   = use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)
    inv_η = _load_inverse_cdf_η(r, host)
    mstar_avg = moments_stellar_mass(1, host)
    v_avg     = 1.0/average_inverse_relative_speed(σ, v_star) # Mpc / s

    println("average velocity = ", v_avg * MPC_TO_KM, " km/s | mstat_avg = ", mstar_avg)

    # Randomly sampling the distributions
    pref = 2*G_NEWTON * mstar_avg / v_avg / b_m^2 * s

    return pref * T(inv_η.(rand(n_stars, T)) ./ rand(n_stars, T))
end



""" nmem maximal number of iteration for the memory"""
function draw_velocity_kick(xp::VT, subhalo::H, r_host::T, host::HM, θ::T = T(π/3.0); nrep::Int = 1, nmem::Int = 10000, use_tables::Bool = true) where {T<:AbstractFloat, U<:Real, S<:Real, HM<:HostModel{T, U}, VT<:Vector{T}, H<:Halo{T, S}} 

    # initialisation for a given value of r
    rs = subhalo.rs
    nstars = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
    b_ms   = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host) ) / rs
    inv_η  = _load_inverse_cdf_η(r_host, host)
    xt     = jacobi_radius(r_host, subhalo, host) / rs

    all(xp .> xt) && return false

    nxp = length(xp) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    

    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end

    v_parallel = Matrix{T}(undef, nxp, nrep)
    v_perp     = Matrix{T}(undef, nxp, nrep)

    irep = 1

    @info "nturn" nturn

    Threads.@threads for it in 1:nturn

        # at the last step we only take the reminder number of draws necessary
        (it == nturn) && (ndraw = (nstars*nrep) % nmem)

        nchunks = (ndraw÷nstars)

        # randomly sampling the distributions
        θb = T(2) * π * rand(ndraw, T)'
        β  = sqrt.(rand(ndraw, T))'
        η  = inv_η.(rand(ndraw, T))'

        # assuming b_min = 0 here

        sθb   = sin.(θb)
        y     = xp ./ (b_ms .* β) 
        denom = @. y^2 + 1 + 2 * y * sθb
        pm    = pseudo_mass_I.(b_ms * β, xt, subhalo.hp)

        _v_parallel = @. (pm * sθb - (y + sθb)/denom) * η / β
        _v_perp     = @. (pm  -  1/denom) * cos(θb) * η / β


        # summing all the contributions
        for j in 1:nchunks
            v_parallel[:, irep] = sum(_v_parallel[:, (j-1)*nstars+1:j*nstars], dims = 2)'
            v_perp[:, irep]     = sum(_v_perp[:, (j-1)*nstars+1:j*nstars], dims = 2)'      
            irep = irep + 1  
        end
    
    end

    return v_parallel, v_perp
end





@doc raw""" 
    
    draw_velocity_kick_complex(z, subhalo, xt, T; nrep, ηb) where {T<:HostModel}

Returns a Monte-Carlo draw of the total velocity kick felt by the particles at
complex positions `z::Vector{Complex{S}}` where S<:Real`, where 

    ``z = r * \sin\psi * exp(-i\varphi)`` 

inside the `subhalo::Halo` with dimensionless tidal radius `xt::Real = rt/rs` (where `rt` and `rs` 
respectively are the tidal and scale radii). The number of crossed star is given by the distance 
of the crossing from the center of the host galaxy `r_host::Real`. The properties of the host are 
encoded into `T<:HostModel`. 

The Monte-Carlo draw is done over `nrep::Int` iterations and we can play with `nmem::Int` the
maximal size of arrays to minimise the impact on the memory. Note that because the routine is
parallelised and a small value of `nmem` is prefered to make the most of the parallelisation.

Returns
-------
- `Δw::Matrix{Complex{S}}` where `S<:Real``
    Δv_x  + i Δv_y

"""
function draw_velocity_kick_complex(
    z::VC, 
    subhalo::H, 
    r_host::T, 
    host::HM,
    θ::T = T(π/3.0),
    xt::Union{T, Nothing} = nothing; 
    nrep::Int = 1, nmem::Int = 10000,
    ηb::T = T(0.0), use_tables::Bool = true
    ) where {T<:AbstractFloat, S <:Real, U<:Real, HM<:HostModel{T, U}, H<:Halo{T, S}, VC<:Vector{Complex{T}}} 

    # initialisation for a given value of r
    rs = subhalo.rs
    nstars = use_tables ? floor(Int, host.number_stellar_encounters(r_host) * 0.5 / cos(θ)) : number_stellar_encounters(r_host, host, θ)
    β_max  = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host) ) / rs
    β_min  = β_max * ηb
    inv_η  = _load_inverse_cdf_η(r_host, host)

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, host) / rs)

    all(abs.(z) .> xt) && return false
    (nmem < nstars) && return false

    nz = length(z) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    
    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end
    

    @info "nturn" nturn

    ######## initialisation
    # the idea is to make all quantities thread safe 
    # initialise the cut into chunks
    # different behaviour for the last part
    nchunks = [nmem÷nstars for i in 1:nturn]
    nchunks[nturn] = ((nstars*nrep) % nmem) ÷ nstars
    ndraws = [ndraw for i in 1:nturn]
    ndraws[nturn] = (nstars*nrep) % nmem
    
    chunks = Matrix{Union{UnitRange{Int64}, Missing}}(missing, nturn, Int(nmem÷nstars))
    irep   = Matrix{Union{Int64, Missing}}(missing, nturn, Int(nmem÷nstars)) # index of the repetition

    for it in 1:nturn
        chunks[it, 1:nchunks[it]] .= [(j-1)*nstars+1:j*nstars for j in 1:nchunks[it]]
        irep[it, 1:nchunks[it]] .= sum(nchunks[1:it-1]) .+ [j for j in 1:nchunks[it]]
    end

    # initialisation of the result matrix
    Δw = Matrix{Complex{T}}(undef, nz, nrep)
    ###### end of initialisation

    ###### entring the main loop
    Threads.@threads for it in 1:nturn

        # randomly sampling the distributions
        β_norm = (sqrt.(rand(ndraws[it]) .* (β_max^2 - β_min^2) .+ β_min^2))
        β  = β_norm .* exp.(- 2.0im  * π * rand(ndraws[it]))  # b/b_max assuming b_min = 0
        η  = inv_η.(rand(ndraws[it]))
        pm = pseudo_mass_I.(β_norm, xt, subhalo.hp) ./ β

        δw = η .* (pm .- 1.0 ./ (z' .+ β))

        # summing all the contributions
        for j in 1:nchunks[it]
            Δw[:, irep[it, j]] .= sum(δw'[:, chunks[it, j]], dims = 2)
        end
    end
    ##### end of the main loop

    return Δw
end


@doc raw""" 
    
    draw_velocity_kick_complex_approx(z, subhalo, xt, T; nrep, ηb) where {T<:HostModel}

See `StellarEncounters.draw_velocity_kick_complex`. Same function but for an approximate experssion
of the velocity kick per star (the equivalent definition defined in arXiv:2201.09788)

"""
function draw_velocity_kick_complex_approx(   
    x::VT, 
    subhalo::H, 
    r_host::T, 
    host::HM,
    θ::T = T(π/3.0),
    xt::Union{T, Nothing} = nothing;
    nrep::Int = 1, nmem::Int = 10000,
    ηb::T = 0.0, use_tables::Bool = true
    ) where {T<:AbstractFloat, S <:Real, U<:Real, VT<:Vector{T}, H<:Halo{T, S}, HM<:HostModel{T, U}}

     # initialisation for a given value of r_host
    rs = subhalo.rs
    nstars = use_tables ? floor(Int, host.number_stellar_encounters(r_host) * 0.5 / cos(θ)) : number_stellar_encounters(r_host, host, θ)
    β_max    = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host) ) / rs
    β_min    = β_max * ηb
    inv_η  = _load_inverse_cdf_η(r_host, host)

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, host) / rs)

    all(abs.(x) .> xt) && return false
    (nmem < nstars) && return false

    nx = length(x) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    
    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end
    
    @info "nturn" nturn

    ######## initialisation
    # the idea is to make all quantities thread safe 
    # initialise the cut into chunks
    # different behaviour for the last part
    nchunks = [nmem÷nstars for i in 1:nturn]
    nchunks[nturn] = ((nstars*nrep) % nmem) ÷ nstars
    ndraws = [ndraw for i in 1:nturn]
    ndraws[nturn] = (nstars*nrep) % nmem
    
    chunks = Matrix{Union{UnitRange{Int64}, Missing}}(missing, nturn, nmem÷nstars)
    irep   = Matrix{Union{Int64, Missing}}(missing, nturn, nmem÷nstars) # index of the repetition

    for it in 1:nturn
        chunks[it, 1:nchunks[it]] .= [(j-1)*nstars+1:j*nstars for j in 1:nchunks[it]]
        irep[it, 1:nchunks[it]] .= sum(nchunks[1:it-1]) .+ [j for j in 1:nchunks[it]]
    end

    # initialisation of the result matrix
    Δw = Matrix{Complex{T}}(undef, nx, nrep)
    ###### end of initialisation

    ###### entring the main loop
    Threads.@threads for it in 1:nturn

        # randomly sampling the distributions
        β = (sqrt.(rand(ndraws[it]) .* (β_max^2 - β_min^2) .+ β_min^2)) # b/b_max assuming b_min = 0
        η  = inv_η.(rand(ndraws[it]))
        pm = pseudo_mass_I.(β, xt, subhalo.hp)

        δw = η ./ β .* sqrt.(pm.^2 .+ 3.0 .* (1.0 .- 2.0.*pm) ./ (3.0 .+ 2.0 .* (x'./ β).^2)) .* exp.(- 2.0im  * π * rand(ndraws[it]))

        # summing all the contributions
        for j in 1:nchunks[it]
            Δw[:, irep[it, j]] .= sum(δw'[:, chunks[it, j]], dims = 2)
        end
    end
    ##### end of the main loop


    return Δw

end

mutable struct VelocityKickDraw{T<:AbstractFloat, S<:Real, U<:Real, H<:Halo{T, S}, HM<:HostModel{T, U}}
    subhalo::H
    rt::T
    r_host::T
    host::HM
    x::Union{Vector{T}, StepRangeLen{T}}
    ψ::Union{Vector{T}, StepRangeLen{T}, Nothing}
    φ::Union{Vector{T}, StepRangeLen{T}, Nothing}
    Δw::Array{Complex{T}}
    δv0::T
    ηb::T
    v::Array{Complex{T}}
    mean_ΔE_approx::Union{Vector{<:T}, Nothing}
end


""" x = r/rs """
function draw_velocity_kick_complex(
    subhalo::H, 
    host::HM,
    x::Union{Vector{T}, StepRangeLen{T}}, 
    ψ::Union{Vector{T}, StepRangeLen{T}, Nothing} = nothing, 
    φ::Union{Vector{T}, StepRangeLen{T}, Nothing} = nothing,
    rt::Union{T, Nothing} = nothing;
    θ::T = T(π/3.0),
    r_host::T = T(8e-3), nrep::Int = 1, nmem::Int = 10000, 
    dflt_ψ::T = T(π/2.0), dflt_φ::T = T(0.0), approx::Bool = false, 
    ηb::T = T(0.0),
    use_tables::Bool = true) where {T<:AbstractFloat, S<:Real, U<:Real, H<:Halo{T, S}, HM<:HostModel{T, U}}
    

    (rt === nothing) && (rt = jacobi_radius(r_host, subhalo, host))

    if !approx 

        (ψ === nothing) && (ψ = [dflt_ψ]) 
        (φ === nothing) && (φ = [dflt_φ])

        nx = length(x) 
        nψ = length(ψ)
        nφ = length(φ)

        # converting x, ψ and φ into a complex number
        function _z_vs_coord(x::T, ψ::T, φ::T)
            (φ == T(2*π)) && return (T(1.0) + 0.0im) * x * sin(ψ)
            return x * sin(ψ) * exp(-im*φ)
        end
        
        z_array = [_z_vs_coord(_x, _ψ, _φ) for _x in x, _ψ in ψ, _φ in φ]
        linear  =  LinearIndices(z_array)
        z       = z_array[linear[:]]

        Δw = draw_velocity_kick_complex(z, subhalo, r_host, host, θ, rt/subhalo.rs; nrep = nrep, nmem = nmem, ηb = ηb, use_tables = use_tables)

        res_array = Array{Complex{T}, 4}(undef, nx, nψ, nφ, nrep)
        
        for k=1:nφ, j=1:nψ
            Threads.@threads for i=1:nx
                #res_array[i, j, k, :] = Δw[i + nx*(j-1) + nx*nψ*(k-1), :]
                res_array[i, j, k, :] = Δw[linear[i, j, k], :]
            end
        end

        (nψ == 1 && nφ != 1) && (res_array = res_array[:, 1, :, :])
        (nψ != 1 && nφ == 1) && (res_array = res_array[:, :, 1, :])
        (nψ == 1 && nφ == 1) && (res_array = res_array[:, 1, 1, :])
    else
        res_array = draw_velocity_kick_complex_approx(x, subhalo, r_host, host, θ, rt/subhalo.rs; nrep = nrep, nmem = nmem,  ηb = ηb, use_tables = use_tables)
    end

    # computing the normalisation
    σ_host    = (use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)) / T(MPC_TO_KM)
    v_star    = (use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)) / T(MPC_TO_KM)
    mstar_avg = moments_stellar_mass(1, host)
    v_avg = 1.0/average_inverse_relative_speed(σ_host, v_star) # Mpc / s

    mean_ΔE_approx = nothing 
    
    if approx && (ηb != T(0.0)) 
        n_stars = use_tables ? host.number_stellar_encounters(r_host) * 0.5 / cos(θ) : number_stellar_encounters(r_host, host, θ)
        mean_ΔE_approx = n_stars .*  mean_velocity_kick_approx_sqr(x, subhalo, r_host, host, rt/subhalo.rs) / 2.0
    end

    δv0 = 2 * G_NEWTON * mstar_avg / v_avg  / subhalo.rs * MPC_TO_KM  # in km / s

    Δv = δv0 .* res_array # in km /s

    # velocity of the particles insinde the halo
    σ_sub = velocity_dispersion_kms.(x * subhalo.rs, rt, subhalo) # in km / s
    v = σ_sub .* (randn(size(Δv)...) .- 1.0im .* randn(size(Δv)...)) # in km / s

    return VelocityKickDraw(subhalo, rt, r_host, host, x, ψ, φ, res_array, δv0, ηb, v, mean_ΔE_approx)
    
end




""" result in (km/s)^2 """
function mean_velocity_kick_approx_sqr(   
    x::VT, 
    subhalo::H, 
    r_host::T, 
    host::HM,
    xt::Union{T, Nothing} = nothing; 
    use_tables::Bool = true,
    use_sharp_truncation::Bool = false,
    use_smooth_truncation::Bool = false,
    q::T = 0.2,
    θ::T = T(π/3.0)
    ) where {T<:AbstractFloat, S <:Real, U<:Real, H<:Halo{T, S}, HM<:HostModel{T, U}, VT<:Vector{T}}


    # initialisation for a given value of r_host
    rs = subhalo.rs
    β_max  = (use_tables ? host.maximum_impact_parameter(r_host) : maximum_impact_parameter(r_host, host)) / rs
    β_min  = T(1e-15) * β_max

    n_stars = T(0);
    ηb_0 = T(0);

    if use_sharp_truncation || use_smooth_truncation
        n_stars = use_tables ? floor(Int, host.number_stellar_encounters(r_host) * 0.5 / cos(θ)) : number_stellar_encounters(r_host, host, θ)
    end

    β_0 = copy(β_min)

    if use_sharp_truncation
        ηb_0 = sqrt(1.0-(1.0-q)^(1.0/n_stars))
        β_0 = β_max * ηb_0
    end

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, host) / rs)
    σ_host    = (use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host)) / T(MPC_TO_KM)
    v_star    = (use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host)) / T(MPC_TO_KM)

    _pm(β::T) = pseudo_mass_I(β, xt, subhalo.hp)
    
    function _to_integrate(β::T) 
        
        weight = T(1)
        
        if use_smooth_truncation
            weight = β / β_max > 0.01 ?  1 - ((1-(β/β_max)^2) / (1-(β_min/β_max)^2))^n_stars : 1 + (n_stars * (β/β_max)^2 - 1) / (1-(β_min/β_max)^2)^n_stars
        end

        return (_pm(β)^2 .+ 3.0*(1.0 - 2.0 *_pm(β))./(3.0 .+ 2.0 .* (x/β).^2)) * weight
    
    end


    integ = QuadGK.quadgk(lnβ -> _to_integrate(exp(lnβ)), log(β_0), log(β_max), rtol=1e-6)[1]
    res = 8.0 * moments_stellar_mass(2, host) * average_inverse_relative_speed_sqr(σ_host, v_star) / rs^2 / (β_max^2 - β_min^2) .* integ
    
    return res .* G_NEWTON^2 * T(MPC_TO_KM)^2

end


function ccdf_ΔE_CL_approx(
    ΔE::Union{VT, T}, 
    x::T, 
    subhalo::H, 
    r_host::T, 
    host::HM,
    xt::Union{T, Nothing} = nothing;
    use_tables::Bool = true,
    use_sharp_truncation::Bool = false,
    use_smooth_truncation::Bool = false,
    q::T = 0.2,
    θ::T = T(π/3.0) 
    ) where {T<:AbstractFloat, S <:Real, U<:Real, H<:Halo{T, S}, HM<:HostModel{T, U}, VT<:Vector{T}}

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, host) / subhalo.rs)

    σ_sub   = velocity_dispersion_kms.(x * subhalo.rs, xt * subhalo.rs, subhalo)
    dv2     = mean_velocity_kick_approx_sqr([x], subhalo, r_host, host, xt, use_tables = use_tables, use_sharp_truncation = use_sharp_truncation, use_smooth_truncation = use_smooth_truncation, q = q, θ = θ)[1]
    n_stars = use_tables ? floor(Int, host.number_stellar_encounters(r_host) * 0.5 / cos(θ)) : number_stellar_encounters(r_host, host, θ)

    s       = sqrt.(n_stars * dv2 / 8.0) ./ σ_sub
    ξ       = sqrt.(1.0 .+ s.^2)./s

    if isa(ΔE, T)
        (ΔE > 0) && return @. (1.0+ξ)/(2.0*ξ)*exp(-ΔE/(2.0*σ_sub^2)*(ξ-1.0))
        (ΔE <=0) && return @. 1.0-(ξ-1.0)/(2.0*ξ) *exp(ΔE/(2.0*σ_sub^2)*(1.0+ξ))
    else
        res = Vector{T}(undef, length(ΔE))
        mask = ΔE .> T(0.0)
        res[mask] = @. (1.0+ξ)/(2.0*ξ)*exp(-ΔE[mask]/(2.0*σ_sub^2)*(ξ-1.0))
        res[.!mask] = @. 1.0-(ξ-1.0)/(2.0*ξ) *exp(ΔE[.!mask]/(2.0*σ_sub^2)*(1.0+ξ))
        return res
    end
end


function ccdf_ΔE_CL_approx(
    ΔE::Union{Vector{T}, T}, 
    draw::VelocityKickDraw;
    use_tables::Bool = true,
    use_sharp_truncation::Bool = false,
    use_smooth_truncation::Bool = false,
    q::T = 0.2,
    θ::T = T(π/3.0)) where {T<:AbstractFloat}
    return ccdf_ΔE_CL_approx(ΔE, draw.x, draw.subhalo, draw.r_host, draw.host, draw.rt/draw.subhalo.rs, use_tables = use_tables, use_sharp_truncation = use_sharp_truncation, use_smooth_truncation = use_smooth_truncation, q = q, θ = θ)
end



function median_ΔE_CL_approx(
    x::VT, 
    subhalo::H, 
    r_host::T, 
    host::HM,
    xt::Union{T, Nothing} = nothing; 
    use_tables::Bool = true,
    use_sharp_truncation::Bool = false,
    use_smooth_truncation::Bool = false,
    q::T = T(0.2),
    θ::T = T(π/3.0)
    ) where {T<:AbstractFloat, S<:Real, U<:Real, H<:Halo{T, S}, VT<:Vector{T}, HM<:HostModel{T, U}}

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, host) / subhalo.rs)

    σ_sub   = velocity_dispersion_kms.(x * subhalo.rs, xt * subhalo.rs, subhalo)
    n_stars = use_tables ? floor(Int, host.number_stellar_encounters(r_host) * 0.5 / cos(θ)) : number_stellar_encounters(r_host, host, θ)
    dv2     = mean_velocity_kick_approx_sqr(x, subhalo, r_host, host, xt, use_tables = use_tables, use_sharp_truncation = use_sharp_truncation, use_smooth_truncation = use_smooth_truncation, q = q, θ = θ)
    
    s       = sqrt.(n_stars * dv2 / 8.0) ./ σ_sub
    ξ       = sqrt.(1.0 .+ s.^2)./s

    return 2.0 .* (σ_sub.^2) ./ (ξ .-1.0).*log.(1.0 .+ 1.0 ./ ξ)
end






function _reduced_array_on_angle(array::Union{Array{<:Complex}, Array{<:Real}}, draw::VelocityKickDraw, angle::Union{Symbol, Nothing} = nothing)
    
    if (((angle !== :ψ) && (angle !== :φ)) || (angle === nothing))
        return array
    end

    angle_array  = getproperty(draw, angle)
    (length(angle_array) == 1) && throw(ArgumentError("Only one value of angle " * angle * " in the draw cannot be reduced"))

    # be default angle is ψ so the dimension on 
    # which we average/reduce is the second one
    # if angle is φ then we need to be carefull
    dim = 2  
    norm = 1.0  
    if angle === :φ 
        (length(draw.ψ) > 1)  && (dim = 3)
        norm = 2.0 * π
    end
   
    n_angle = length(angle_array)

    # average using trapeze rule 
    if angle == :ψ
        a = selectdim((array .* (sin.(angle_array))'), dim, 1:(n_angle-1))
        b = selectdim((array .* (sin.(angle_array))'), dim, 2:n_angle)
    elseif angle == :φ 
        a = selectdim((array), dim, 1:(n_angle-1))
        b = selectdim((array), dim, 2:n_angle)
    end

    return selectdim(sum((a.+b) ./ 2.0 .* diff(angle_array)' , dims = dim), dim, 1) ./ norm
end

function _reduced_array_on_angles(array::Union{Array{<:Complex}, Array{<:Real}}, draw::VelocityKickDraw; angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)
    
    (typeof(angle) === Symbol) && (return _reduced_array_on_angle(array, draw, angle))
    
    (angle === nothing) && (return array)
    (length(angle) < 1)  && (return array)
    
    (length(angle) == 1) && (return _reduced_array_on_angle(array, draw, angle[1]))
    (length(angle) == 2) && (return _reduced_array_on_angle(_reduced_array_on_angle(array, draw, angle[1]), draw, angle[2]))

end



## some basic properties associated with the draw of the velocity kick
Δv(Δw::Array{<:Complex}, δv0::AbstractFloat) = δv0^2 .* Δw
Δv(draw::VelocityKickDraw) = Δv(draw.Δw, draw.δv0)
Δv2(Δw::Array{<:Complex}, δv0::AbstractFloat) = δv0^2 .* abs2.(Δw)
Δv2(draw::VelocityKickDraw) = Δv2(draw.Δw, draw.δv0)
ΔE(Δw::Array{<:Complex}, v::Array{<:Complex}, δv0::AbstractFloat) = Δv2(Δw, δv0) ./ 2.0 + real(v .* Δv(Δw, δv0))
ΔE(draw::VelocityKickDraw) = ΔE(draw.Δw, draw.v, draw.δv0)
mean_ΔE(Δw::Array{<:Complex}, δv0::AbstractFloat) = selectdim(Statistics.mean(Δv2(Δw, δv0), dims = length(size(Δw))), length(size(Δw)), 1) ./ 2.0
mean_ΔE(draw::VelocityKickDraw) = mean_ΔE(draw.Δw, draw.δv0)

## reduced (i.e. angle averaged) versions of the properties
reduced_Δv(draw::VelocityKickDraw; angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)  = _reduced_array_on_angles(Δv(draw), draw, angle = angle)
reduced_Δv2(draw::VelocityKickDraw; angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing) =  _reduced_array_on_angles(Δv2(draw), draw, angle = angle)
reduced_ΔE(draw::VelocityKickDraw; angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)  = _reduced_array_on_angles(ΔE(draw), draw, angle = angle)
reduced_mean_ΔE(draw::VelocityKickDraw; angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing) = _reduced_array_on_angles(mean_ΔE(draw), draw, angle = angle)


function _reduced_array_size(draw::VelocityKickDraw; reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)
    
    s = size(draw.Δw)
    l = length(s)
    
    if reduce_angle === nothing
        return s
    else
        
        if typeof(reduce_angle) === Symbol || (typeof(reduce_angle) === Vector{Symbol} && length(reduce_angle) == 1)
            (l == 4) && return (s[1:2], s[end])
            (l == 3) && return (s[1], s[end])
        end
        if (typeof(reduce_angle) === Vector{Symbol} && length(reduce_angle) == 2)
            (l == 4) && return (s[1], s[end])
        end
    end

    throw(ArgumentError("Trying to reduce an array of the wrong size"))
end



@inline function ccdf_ΔE(ΔE_input::Real, ΔE_distrib::Union{Array{<:Real}, SubArray{<:Real}})
    sorted = ndims(ΔE_distrib) > 1 ? sort(ΔE_distrib, dims=ndims(ΔE_distrib)) : sort(ΔE_distrib)
    return sum(sorted .> ΔE_input, dims=ndims(ΔE_distrib)) / size(ΔE_distrib, ndims(ΔE_distrib))
end


function ccdf_ΔE(ΔE_input::Real, draw::VelocityKickDraw; reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing,
    indices::Union{Tuple{Vararg{Union{Int, UnitRange{Int}}}}, Union{Int, UnitRange{Int}}, Nothing} = nothing)

    if indices === nothing
        return ccdf_ΔE(ΔE_input, reduced_ΔE(draw, angle = reduce_angle))
    end

    isa(indices, Union{Int, UnitRange{Int}}) && (indices = (indices..., ))
    (length(indices) >= length(_reduced_array_size(draw, reduce_angle = reduce_angle))) && throw(ArgumentError("Indices should not have dimension larger than the reduced array."))
    indices = (indices..., :)

    return  ccdf_ΔE(ΔE_input, reduced_ΔE(draw, angle = reduce_angle)[indices...])
end


function median_ΔE(draw::VelocityKickDraw; reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)
    
    dim_size = _reduced_array_size(draw, reduce_angle = reduce_angle)[1:end-1]
    res      = Array{Union{Missing, Float64}}(undef, dim_size...)

    for index in CartesianIndices(dim_size)
        _to_bisect(log10ΔE::Real) = (ccdf_ΔE(10.0.^log10ΔE, draw, reduce_angle = reduce_angle, indices = Tuple(index)) .- 0.5)[1]
        try
            res[index] = 10.0.^(Roots.find_zero(_to_bisect, (-15, 0), Roots.Bisection()))
        catch 
            res[index] = missing
        end
    end

    return res 
end


""" deprecated because confusing"""
function ccdf_ΔE_CL(ΔE::Real, draw::VelocityKickDraw; reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)
    
    sigma = velocity_dispersion_kms.(draw.x * draw.subhalo.rs, draw.rt, draw.subhalo)
    _m_ΔE = draw.mean_ΔE_approx !== nothing ? draw.mean_ΔE_approx : reduced_mean_ΔE(draw, angle = reduce_angle)
    s     = sqrt.(_m_ΔE)./(2.0.*sigma)
    ξ     = sqrt.(1.0 .+ s.^2)./s

    (ΔE  > 0) && return @. (1.0+ξ)/(2.0*ξ)*exp(-ΔE/(2.0*sigma^2)*(ξ-1.0))
    (ΔE <=0) && return @. 1.0-(ξ-1.0)(2.0*ξ) *exp(ΔE/(2.0*sigma^2)*(1.0+ξ))
end




function _ccdf_ΔE_array(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw, 
    ccdf_ΔE_single::Function; reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing)

    s = _reduced_array_size(draw, reduce_angle = reduce_angle)
    res_ΔE = Array{Float64}(undef, s[1:end-1]..., length(ΔE))

    for index in eachindex(ΔE)
        selectdim(res_ΔE, length(s), index) .= ccdf_ΔE_single(ΔE[index], draw, reduce_angle = reduce_angle)
    end

    return res_ΔE
end



ccdf_ΔE(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw;  reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing) = _ccdf_ΔE_array(ΔE, draw, ccdf_ΔE, reduce_angle= reduce_angle)
ccdf_ΔE_CL(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw;  reduce_angle::Union{Vector{Symbol}, Symbol, Nothing} = nothing) = _ccdf_ΔE_array(ΔE, draw, ccdf_ΔE_CL, reduce_angle=reduce_angle)




function _save_inverse_cdf_η(r_host::Real, host::HM; use_tables::Bool = true) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}}
     
    σ         = (use_tables ? host.velocity_dispersion_spherical_kms(r_host) : velocity_dispersion_spherical_kms(r_host, host))
    v_star    = (use_tables ? host.circular_velocity_kms(r_host) : circular_velocity_kms(r_host, host))
    mstar_avg = moments_stellar_mass(1, host)
    v_avg     = 1.0/average_inverse_relative_speed(σ, v_star) # km/s

    rnd_array = 10.0.^range(-8, -1e-12, 1000)

    # -------------------------------------------
    # Checking if the file does not already exist
    hash_value = hash((r_host, host.name))
    file = "cdf_eta_" * string(hash_value, base=16) * ".jld2" 
 
    if file in readdir(".cache/")
        existing_data = jldopen(".cache/cdf_eta_" * string(hash_value, base=16) * ".jld2")
        (existing_data["r_host"] == r_array) && @info "| file to save is already cached" && return nothing
    end
    # -------------------------------------------

    inv_cdf = inverse_cdf_η.(rnd_array, σ, v_star, mstar_avg, v_avg, host)
    JLD2.jldsave(".cache/cdf_eta_" * string(hash_value, base=16) * ".jld2"; rnd = rnd_array, inverse_cdf_eta = inv_cdf, r_host = r_host)

    return true
end



## Possibility to interpolate the model
function _load_inverse_cdf_η(r_host::T, host::HM) where {T<:AbstractFloat, U<:Real, HM<:HostModel{T, U}}
    """ change that to a save function """

    hash_value = hash((r_host, host.name))
    filenames = readdir(".cache/")
    file = "cdf_eta_" * string(hash_value, base=16) * ".jld2" 
    !(file in filenames) && _save_inverse_cdf_η(r_host, host)
    
    data = JLD2.jldopen(".cache/" * file)
    rnd_array = data["rnd"]
    inv_cdf = data["inverse_cdf_eta"]
        
    log10inv_cdf = Interpolations.interpolate((log10.(rnd_array),), log10.(inv_cdf),  Interpolations.Gridded(Interpolations.Linear()))
   
    function inv_cdf_η(rnd::AbstractFloat) 
        try
            return 10.0^log10inv_cdf(log10(rnd)) 
        catch e
            println(rnd)
            return false
        end
    end

    return inv_cdf_η

end


#######################
## FUNCTION FOR TESTS -> TO BE CLEANED
