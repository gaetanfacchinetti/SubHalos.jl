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

export jacobi_scale, jacobi_radius,jacobi_scale_DM_only, tidal_scale, baryons_tidal_scale

################################################
## Jacobi Radius

#struct BissectionJacobiTS
#    r_host::T
#    ρs::T
#end


@doc raw"""
    jacobi_scale(r, ρs, hp, ρ_host, m_host)

Jacobi scale radius for a subhalo of scale density `ρs` (in Msol/Mpc^3) with a HaloProfile `hp`
at a distance r from the host centre  such that the sphericised mass density of the host at r is `ρ_host` 
and the sphericised enclosed mass inside the sphere of radius r is `m_host`.

More precisely, returns `xt` solution to
``\frac{xt^3}{\mu(xt)} - \frac{\rho_{\rm s}}{\rho_{\rm host}(r)} \frac{\hat \rho}{1-\hat \rho}``
with `` reduced sperical host density being
`` = \frac{4\pi}{3} r^3 \frac{\rho_{\rm host}(r)}{m_{\rm host}(r)}``
"""
function jacobi_scale(r_host::T, ρs::T, hp::HaloProfile{S}, ρ_host::T, m_host::T)::T where {T<:AbstractFloat, S<:Real}
    
    reduced_ρ =  T(4 * π) * r_host^3 *  ρ_host / T(3.0) / m_host
    _to_bisect(xt::Real) = xt^3 / convert(T, μ_halo(xt, hp)) - (ρs / ρ_host) * reduced_ρ / (T(1) - reduced_ρ)
    
    res = zero(T)

    try
        res = exp(Roots.find_zero(lnxt -> _to_bisect(exp(lnxt)), (log(T(1e-10)), log(T(1e+4))), Roots.Bisection(), xrtol = T(1e-10))) 
    catch e
        msg = "Impossible to compute the jacobi scale at rhost = " * r_host * " Mpc for " * string(hp) *  " | c200 (planck18) = " * string(cΔ_from_ρs(ρs, hp, 200, planck18)) * " [min, med, max] = " * _to_bisect.([1e-10, sqrt(1e-10 * 1e+4), 1e+4]) * "\n" * e.msg
        throw(ArgumentError(msg))
    end

    return res
end


jacobi_scale(r_host::T, ρs::T, hp::HaloProfile{S}, host::HostModel{T}) where {T<:AbstractFloat, S<:Real} = jacobi_scale(r_host, ρs, hp, host.ρ_host_spherical(r_host), host.m_host_spherical(r_host))
jacobi_scale(r_host::T, subhalo::Halo{T, S}, ρ_host::T, m_host::T) where {T<:AbstractFloat, S<:Real}  = jacobi_scale(r_host, subhalo.ρs, subhalo.hp,  ρ_host, m_host)
jacobi_scale(r_host::T, subhalo::Halo{T, S}, host::HostModel{T, U}) where {T<:AbstractFloat, S<:Real, U<:Real} = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)
jacobi_radius(r_host::T, subhalo::Halo{T, S}, host::HostModel{T, U}) where {T<:AbstractFloat, S<:Real, U<:Real}  = subhalo.rs * jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)


jacobi_scale_DM_only(r_host::T, subhalo::Halo{T, S}, host::HostModel{T, U}) where {T<:AbstractFloat, S<:Real, U<:Real} = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, ρ_halo(r_host, host.halo), m_halo(r_host, host.halo))
jacobi_radius_DM_only(r_host::T, subhalo::Halo{T, S}, host::HostModel{T, U}) where {T<:AbstractFloat, S<:Real, U<:Real} = subhalo.rs * jacobi_scale_DM_only(r_host, subhalo, host)


#########################################
# Total baryonic tides

struct BissectionBaryonsTS{T<:AbstractFloat, S<:Real, U<:Real}
    x_init::T
    r_host::T
    subhalo::Halo{T, S}
    host::HostModel{T, U}
    pp::ProfileProperties{T, S}
    q::T
    θ::T 
    disk::Bool
    stars::Bool 
    use_tables::Bool
end


# functor to avoid putting to use to much memory
function (f::BissectionBaryonsTS{T, S, U})(lnx::T)::T where {T<:AbstractFloat, S<:Real, U<:Real}

    ΔE = zero(T)
    x = exp(lnx)

    f.disk && (ΔE += angle_average_energy_shock(x * f.subhalo.rs, f.x_init * f.subhalo.rs, f.r_host, f.subhalo, f.host))
    f.stars && (ΔE += T(0.7) * average_energy_kick_stars(x, f.x_init, β_min(f.q, f.r_host, f.subhalo.rs, f.host, f.θ, f.use_tables), f.r_host, f.subhalo.rs, f.host, f.pp, f.θ, f.use_tables))

    return ΔE / abs(gravitational_potential(x * f.subhalo.rs, f.x_init * f.subhalo.rs, f.subhalo)) - T(1)
    
end

""" Tidal radius after one crossing of the disk in units of the scale radius """
function baryons_tidal_scale(
    x_init::T, 
    r_host::T, 
    subhalo::Halo{T, S}, 
    host::HostModel{T, U}, 
    pp::ProfileProperties{T, S}, 
    q::T, 
    θ::T, 
    disk::Bool, 
    stars::Bool, 
    use_tables::Bool
    )::T where {T<:AbstractFloat, S<:Real, U<:Real}
   
    # if no baryons effect just return x_init
    !(disk || stars) && (return x_init)
  
    f = BissectionBaryonsTS{T, S, U}(x_init, r_host, subhalo, host, pp, q, θ, disk, stars, use_tables)

    res = T(-1.0) 
    
    try
        res = exp(Roots.find_zero(f, (log(T(1e-15)), log(T(x_init))), Roots.Bisection(), xrtol = T(1e-8)))
    catch e
        if isa(e, ArgumentError)
            
            # if even at the smallest value the energy kick is above the potential then put xt = 0
            (f(T(1e-15)) >= zero(T)) && (return zero(T))
            (T(r_host) > T(0.98*host.rt)) && (return x_init)

            msg = "Impossible to compute the baryons_tidal scale for (x_init, r_host, q, θ, disk, stars, use_tables) = (" * 
            string(x_init) * ", " * string(r_host) * ", " * string(q) * ", " * string(θ) * ", " * string(disk) * ", " * 
            string(stars) * ", " * string(use_tables) * ")\n" * string(subhalo) * "\n [vals] = " *
            string(f.(10.0.^range(-15, log10(x_init), 20))) * "\n" * e.msg
            throw(ArgumentError(msg))
        else
            rethrow()
        end
    end

    return res

end


""" Tidal radius after n_cross crossing of the disk in units of the scale radius """
function baryons_tidal_scale(
    x_init::T, 
    r_host::T, 
    subhalo::Halo{T, S}, 
    host::HostModel{T, U}, 
    n_cross::Int, 
    pp::ProfileProperties{T, S}, 
    q::T, 
    θ::T, 
    disk::Bool, 
    stars::Bool, 
    use_tables::Bool
    )::T where {T<:AbstractFloat, S<:Real, U<:Real}
    

    # if no baryons effect just return x_init
    !(disk || stars) && (return x_init)

    xt = x_init

    for i in 1:n_cross
        xt = baryons_tidal_scale(xt, r_host, subhalo, host, pp, q, θ, disk, stars, use_tables)
        (xt <= T(2e-15)) && (return zero(T))

    end

    return xt
end

function tidal_scale(
    r_host::T, 
    subhalo::Halo{T, S}, 
    host::HostModel{T, U} = milky_way_MM17_g1, 
    z::T = T(0.0), 
    Δ::T = T(200), 
    cosmo::Cosmology{T} = planck18; 
    pp::ProfileProperties{T, S}, 
    q::T = T(0.2), 
    θ::T = T(π/3), 
    disk::Bool = true, 
    stars::Bool = false, 
    use_tables::Bool = true
    ) where {T<:AbstractFloat, S<:Real, U<:Real}


    xt = min(jacobi_scale(r_host, subhalo, host), convert(T, cΔ(subhalo, Δ, cosmo)))

    if r_host >= host.rt
        return xt
    end

    if (disk || stars)
        n_cross = 2 * number_circular_orbits(r_host, host, z, cosmo.bkg, use_tables = use_tables)
        return baryons_tidal_scale(xt, r_host, subhalo, host, n_cross, pp, q, θ, disk, stars, use_tables)
    end

    return T(-1.0)

end

tidal_radius(r_host::Real, subhalo::Halo{<:Real}, z::Real = 0,  Δ::Real = 200, cosmo::Cosmology = planck18) = tidal_scale(r_host, subhalo, z, Δ, cosmo) * subhalo.rs



