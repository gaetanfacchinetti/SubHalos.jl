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

module SubHalos

import QuadGK, Roots, JLD2,  Interpolations, SpecialFunctions, Flux, Optimisers, Statistics

import CosmoTools: median_concentration, SCP12, MassConcentrationModel, gravitational_potential, Cosmology, BkgCosmology
import CosmoTools: planck18, MPC_TO_KM, orbital_frequency, G_NEWTON, KM_TO_MPC, HaloProfile, αβγProfile, plummerProfile
import CosmoTools: Halo, μ_halo, nfwProfile, halo_from_mΔ_and_cΔ, cΔ, cΔ_from_ρs,  ρ_halo, mΔ, m_halo, circular_velocity, velocity_dispersion, velocity_dispersion_kms
import HostHalos: HostModel, number_circular_orbits, milky_way_MM17_g1, number_circular_orbits, load!, σ_stars, ρ_stars, stellar_mass_function
import HostHalos: maximum_impact_parameter, number_stellar_encounters, moments_stellar_mass, velocity_dispersion_spherical_kms, circular_velocity_kms


import CosmoTools: get_halo_profile_type, get_cosmology_type
import HostHalos: get_host_halo_type

include("./StellarEncounters.jl")
include("./CachingProfile.jl")
include("./DiskShocking.jl")
include("./TidalStripping.jl")
include("./InterpolateFSL.jl")
include("./FSL.jl")
include("./CachingFSL.jl")
include("./TidalMass.jl")
include("./MonteCarlo.jl")


end
