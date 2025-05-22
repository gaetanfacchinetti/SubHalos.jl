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

export  disk_shocking_tidal_radius, angle_average_energy_shock

adiabatic_correction(η::Real) = (1+η^2)^(-3/2)

""" correction factor for adiabatic shocks (Gnedin)"""
function adiabatic_correction(r_sub::Real, rt_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real})
    
    hd = host.stars.thick_zd # in Mpc
    σz = sqrt(2.0 / π) * host.velocity_dispersion_spherical_kms(r_host) # in km/s
    td = hd * MPC_TO_KM / σz # in s
    ωd = velocity_dispersion(r_sub, rt_sub, subhalo) / r_sub  # in 1 / s
    η = td * ωd
    
    return adiabatic_correction(η)

    # σz = host.circular_velocity_kms(r_host) / sqrt(2) # in km/s
    # ωd = circular_velocity(r_sub, subhalo)/ r_sub * sqrt(3.0/2.0) # in 1/s (isothermal orbital frequency) #orbital_frequency(r_sub, subhalo) # in 1/s
    # ωd = velocity_dispersion(r_sub, rt_sub, subhalo) / r_sub
     #ωd = sqrt(4 * π * ρs * rs^2 * G_NEWTON) * velocity_dispersion(r_sub / rs, rt_sub / rs) / r_sub 
end

""" Energy gained by particle mass in a single crossing of th disk (in Mpc / s)^2 """
function angle_average_energy_shock(r_sub::Real, rt_sub::Real, r_host::Real, subhalo::Halo{<:Real}, host::HostModel{<:Real},)
    vz =  sqrt(2.0 / π) * host.velocity_dispersion_spherical_kms(r_host) * KM_TO_MPC # in Mpc / s
    
    if vz == 0
        # first we need to make sure the problem is not comming from interpolations
        # around the edges of the host (in which case, there is no baryons anymore
        # and we can simply return 0.0
        (r_host >= 0.98 * host.rt) && (return 0.0)
        return Inf
    end

    disc_acceleration = 2.0 * π * G_NEWTON * host.σ_baryons(r_host) # in Mpc / s^2
    return 2 * (disc_acceleration * r_sub / vz )^2 * adiabatic_correction(r_sub, rt_sub, r_host, subhalo, host) / 3.0;
end


