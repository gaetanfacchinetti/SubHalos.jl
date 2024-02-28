export jacobi_scale, jacobi_scale_DM_only, disk_shocking_tidal_radius, tidal_scale, angle_average_energy_shock

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
        msg = "Impossible to compute the jacobi scale for " * string(hp) *  " | c200 (planck18) = " * string(cΔ_from_ρs(ρs, hp, 200, planck18)) * "\n" * e.msg
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


################################################
## Disk shocking

adiabatic_correction(η::Real) = (1+η^2)^(-3/2)

""" correction factor for adiabatic shocks (Gnedin)"""
function adiabatic_correction(r_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    hd = host.stars.thick_zd # in Mpc
    σz = host.circular_velocity_kms(r_host) / sqrt(2) # in km/s
    td = hd * MPC_TO_KM / σz # in s
    ωd = orbital_frequency(r_sub, subhalo) # in 1/s
    η = td * ωd
    return adiabatic_correction(η)
end

""" Energy gained by particle mass in a single crossing of th disk (in Mpc / s)^2 """
function angle_average_energy_shock(r_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    disc_acceleration = 2.0 * π * G_NEWTON * host.σ_baryons(r_host) # in Mpc / s^2
    σz = host.circular_velocity_kms(r_host) * KM_TO_MPC / sqrt(2) # in Mpc / s
    return 2 * (disc_acceleration / σz * r_sub)^2 * adiabatic_correction(r_sub, r_host, subhalo, host) / 3.0;
end

""" Tidal radius after one crossing of the disk in units of the scale radius """
function disk_shocking_tidal_radius(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
   
    _to_bisect(x::Real) = angle_average_energy_shock(x * subhalo.rs, r_host, subhalo, host) / abs(gravitational_potential(x * subhalo.rs, x_init * subhalo.rs, subhalo)) - 1.0
    
    res = -1.0 
    
    try
        res = exp(Roots.find_zero(lnx -> _to_bisect(exp(lnx)), (log(1e-15), log(x_init)), Roots.Bisection(), xrtol = 1e-6))
    catch e
        if isa(e, ArgumentError)
            
            # if even at the smallest value the energy kick is above the potential then put xt = 0
            (_to_bisect(1e-15) >= 0) && (return 0)

            msg = "Impossible to compute the jacobi scale for " * string(subhalo) * "\n" * e.msg
            throw(ArgumentError(msg))
        else
            throw(e)
        end
    end

    return res

end

""" Tidal radius after n_cross crossing of the disk in units of the scale radius """
function disk_shocking_tidal_radius(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, n_cross::Int)

    xt = x_init

    for i in 1:n_cross
        xt = disk_shocking_tidal_radius(xt, r_host, subhalo, host)
        (xt == 0) && (return 0)
    end

    return xt
end

#########################################
# Total baryonic tides

# Need to add stellar encounters here afterwards
function baryonic_tides(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, n_cross::Int)
    return disk_shocking_tidal_radius(x_init, r_host, subhalo, host, n_cross)
end

function tidal_scale(r_host::Real, subhalo::Halo{<:Real}, host::HostModel = milky_way_MM17_g1, z::Real = 0.0, Δ::Real = 200, cosmo::Cosmology = planck18)

    xt = min(jacobi_scale(r_host, subhalo, host), cΔ(subhalo, Δ, cosmo))
    n_cross = 2 * number_circular_orbits(r_host, host, z, cosmo.bkg)
    return baryonic_tides(xt, r_host, subhalo, host, n_cross)

end

tidal_radius(r_host::Real, subhalo::Halo{<:Real}, z::Real = 0,  Δ::Real = 200, cosmo::Cosmology = planck18) = tidal_scale(r_host, subhalo, z, Δ, cosmo) * subhalo.rs


## IMPLEMENT σ_baryons
