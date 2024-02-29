module SubHalos

import QuadGK, Roots, JLD2,  Interpolations, SpecialFunctions

import CosmoTools: median_concentration, SCP12, MassConcentrationModel, gravitational_potential, Cosmology
import CosmoTools: planck18, MPC_TO_KM, orbital_frequency, G_NEWTON, KM_TO_MPC, HaloProfile, αβγProfile, plummerProfile
import CosmoTools: Halo, μ_halo, nfwProfile, halo_from_mΔ_and_cΔ, cΔ, cΔ_from_ρs,  ρ_halo, mΔ, m_halo
import HostHalos: HostModel, number_circular_orbits, milky_way_MM17_g1, number_circular_orbits, preload!, σ_stars, ρ_stars 
import HostHalos: maximum_impact_parameter, number_stellar_encounters, moments_stellar_mass

include("./StellarEncounters.jl")
include("./DiskShocking.jl")
include("./TidalStripping.jl")
include("./FSL.jl")
include("./TidalMass.jl")

end
