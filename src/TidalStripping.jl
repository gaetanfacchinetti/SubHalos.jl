export jacobi_scale, jacobi_scale_DM_only, disk_shocking_tidal_radius, tidal_scale, angle_average_energy_shock, baryons_tidal_scale

################################################
## Jacobi Radius

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
function jacobi_scale(r_host::Real, ρs::Real, hp::HaloProfile, ρ_host::Real, m_host::Real)
    
    reduced_ρ =  4 * π * r_host^3 *  ρ_host / 3.0 / m_host
    _to_bisect(xt::Real) = xt^3 / μ_halo(xt, hp) - (ρs / ρ_host) * reduced_ρ / (1.0 - reduced_ρ)
    
    res = 0.0

    try
        res = exp(Roots.find_zero(lnxt -> _to_bisect(exp(lnxt)), (log(1e-10), log(1e+4)), Roots.Bisection(), xrtol = 1e-3)) 
    catch e
        msg = "Impossible to compute the jacobi scale at rhost = " * r_host * " Mpc for " * string(hp) *  " | c200 (planck18) = " * string(cΔ_from_ρs(ρs, hp, 200, planck18)) * " [min, med, max] = " * _to_bisect.([1e-10, sqrt(1e-10 * 1e+4), 1e+4]) * "\n" * e.msg
        throw(ArgumentError(msg))
    end

    return res
end


jacobi_scale(r_host::Real, ρs::Real, hp::HaloProfile, host::HostModel{<:Real}) = jacobi_scale(r_host, ρs, hp, host.ρ_host_spherical(r_host), host.m_host_spherical(r_host))
jacobi_scale(r_host::Real, subhalo::Halo{<:Real}, ρ_host::Real, m_host::Real) = jacobi_scale(r_host, subhalo.ρs, subhalo.hp,  ρ_host, m_host)
jacobi_scale(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)

jacobi_scale_DM_only(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, ρ_halo(r_host, host.halo), m_halo(r_host, host.halo))

jacobi_radius(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = subhalo.rs * jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)

jacobi_radius_DM_only(r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = subhalo.rs * jacobi_scale_DM_only(r_host, subhalo.ρs, subhalo.hp, host)


#########################################
# Total baryonic tides

""" Tidal radius after one crossing of the disk in units of the scale radius """
function baryons_tidal_scale(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, q::Real, θ::Real, disk::Bool, stars::Bool, use_tables::Bool)
   
    # if no baryons effect just return x_init
    !(disk || stars) && (return x_init)

  
    function _to_bisect(x::Real) 
        ΔE = 0

        disk && (ΔE += angle_average_energy_shock(x * subhalo.rs, r_host, subhalo, host))
        stars && (ΔE += 0.7 * average_energy_kick_stars(x, x_init, β_min(q, r_host, subhalo, host, θ, use_tables), r_host, subhalo, host, θ, use_tables))

        return ΔE / abs(gravitational_potential(x * subhalo.rs, x_init * subhalo.rs, subhalo)) - 1.0
    end

    res = -1.0 
    
    try
        res = exp(Roots.find_zero(lnx -> _to_bisect(exp(lnx)), (log(1e-15), log(x_init)), Roots.Bisection(), xrtol = 1e-6))
    catch e
        if isa(e, ArgumentError)
            
            # if even at the smallest value the energy kick is above the potential then put xt = 0
            (_to_bisect(1e-15) >= 0) && (return 0)

            msg = "Impossible to compute the baryons_tidal scale for (x_init, r_host, q, θ, disk, stars, use_tables) = (" * 
            string(x_init) * ", " * string(r_host) * ", " * string(q) * ", " * string(θ) * ", " * string(disk) * ", " * 
            string(stars) * ", " * string(use_tables) * ")" * string(subhalo) * " [vals] = " *
            string(_to_bisect.(10.0.^range(-15, log10(x_init), 20))) * "\n" * e.msg
            throw(ArgumentError(msg))
        else
            throw(e)
        end
    end

    return res

end


""" Tidal radius after n_cross crossing of the disk in units of the scale radius """
function baryons_tidal_scale(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, n_cross::Int, q::Real, θ::Real, disk::Bool, stars::Bool, use_tables::Bool)

    # if no baryons effect just return x_init
    !(disk || stars) && (return x_init)

    xt = x_init

    for i in 1:n_cross
        xt = baryons_tidal_scale(xt, r_host, subhalo, host, q, θ, disk, stars, use_tables)
        (xt == 0) && (return 0)
    end

    return xt
end

function tidal_scale(r_host::Real, subhalo::Halo{<:Real}, host::HostModel = milky_way_MM17_g1, z::Real = 0.0, Δ::Real = 200, cosmo::Cosmology = planck18,
    q::Real=0.2, θ::Real = π/3, disk::Bool = true, stars::Bool = false, use_tables::Bool = true)

    xt = min(jacobi_scale(r_host, subhalo, host), cΔ(subhalo, Δ, cosmo))
    n_cross = 2 * number_circular_orbits(r_host, host, z, cosmo.bkg)
    return baryons_tidal_scale(xt, r_host, subhalo, host, n_cross,  q, θ, disk, stars, use_tables)

end

tidal_radius(r_host::Real, subhalo::Halo{<:Real}, z::Real = 0,  Δ::Real = 200, cosmo::Cosmology = planck18) = tidal_scale(r_host, subhalo, z, Δ, cosmo) * subhalo.rs


## IMPLEMENT σ_baryons
