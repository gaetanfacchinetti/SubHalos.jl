export  disk_shocking_tidal_radius, angle_average_energy_shock

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


