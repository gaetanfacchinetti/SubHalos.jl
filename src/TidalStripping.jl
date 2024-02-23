export jacobi_scale, disk_shocking_tidal_radius, tidal_scale

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
function jacobi_scale(r::Real, ρs::Real, hp::HaloProfile, ρ_host::Real, m_host::Real)
    reduced_ρ =  4 * π * r^3 *  ρ_host / 3.0 / m_host
    to_zero(xt::Real) = xt^3/μ_halo(xt, hp) - ρs/ρ_host * reduced_ρ / (1.0 - reduced_ρ)
    return exp(Roots.find_zero(lnxt -> to_zero(exp(lnxt)), (-5, +5), Roots.Bisection())) 
end


jacobi_scale(r::Real, ρs::Real, hp::HaloProfile, host::HostModel{<:Real}) = jacobi_scale(r, ρs, hp, host.ρ_host_spherical(r), host.m_host_spherical(r))
jacobi_scale(r::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = jacobi_scale(r, subhalo.ρs, subhalo.hp, host)
jacobi_scale(r::Real, subhalo::Halo{<:Real}, ρ_host::Real, m_host::Real) = jacobi_scale(r, subhalo.ρs, subhalo.hp,  ρ_host, m_host)

jacobi_radius(r::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}) = subhalo.rs * jacobi_scale(r, subhalo.ρs, subhalo.hp, host)


################################################
## Disk shocking

adiabatic_correction(η::Real) = (1+η^2)^(-3/2)

function adiabatic_correction(r_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    hd = host.stars.thick_zd # in Mpc
    σz = host.circular_velocity_kms(r_host) / sqrt(2) # in km/s
    td = hd * MPC_TO_KM / σz # in s
    ωd = orbital_frequency(r_sub, subhalo) # in 1/s
    η = td * ωd
    return adiabatic_correction(η)
end

## NEED TO IMPLEMENT σ_baryons
""" Energy gained by particle mass in a single crossing of th disk (in Mpc / s)^2 """
function angle_average_energy_shock(r_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    disc_acceleration = 2.0 * π * G_NEWTON * host.σ_baryons(r_host) # in Mpc / s^2
    σz = host.circular_velocity_kms(r_host) * KM_TO_MPC / sqrt(2) # in Mpc / s
    return 2 * (disc_acceleration / σz * r_sub)^2 * adiabatic_correction(r_sub, r_host, subhalo, host) / 3.0;
end

""" Tidal radius after one crossing of the disk in units of the scale radius """
function disk_shocking_tidal_radius(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    _to_bissect(x::Real) = angle_average_energy_shock(x * subhalo.rs, r_host, subhalo, host) / abs(gravitational_potential(x * subhalo.rs, x_init * subhalo.rs, subhalo)) -1.0
    return exp(Roots.find_zero(lnx -> _to_bissect(exp(lnx)), (log(1e-6), log(x_init)), Roots.Bisection()))
end

""" Tidal radius after n_cross crossing of the disk in units of the scale radius """
function disk_shocking_tidal_radius(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, n_cross::Int)

    xt = x_init

    for i in 1:n_cross
        (xt < 1e-6) && return 0.0
        xt = disk_shocking_tidal_radius(xt, r_host, subhalo, host)
    end

    return xt
end

#########################################
# Total baryonic tides

# Need to add stellar encounters here afterwards
function baryonic_tides(x_init::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real}, n_cross::Int)
    return disk_shocking_tidal_radius(x_init, r_host, subhalo, host, n_cross)
end

function tidal_scale(r_host::Real, subhalo::Halo{<:Real}, host::HostModel = milky_way_MM17_g1, z::Real = 0.0, cosmo::Cosmology = planck18)

    xt = jacobi_scale(r_host, subhalo, host)
    n_cross = number_circular_orbits(r_host, host, z, cosmo.bkg)
    return baryonic_tides(xt, r_host, subhalo, host, n_cross)

end

tidal_radius(r_host::Real, subhalo::Halo{<:Real}, z::Real = 0, cosmo::Cosmology = planck18) = tidal_scale(r_host, subhalo, z, cosmo) * subhalo.rs


## IMPLEMENT σ_baryons
