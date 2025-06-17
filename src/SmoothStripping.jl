

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

export jacobi_radius, jacobi_scale, jacobi_radius_DM_only, jacobi_scale_DM_only

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
        res = exp(Roots.find_zero(lnxt -> _to_bisect(exp(lnxt)), (log(T(1e-10)), log(T(1e+4))), Roots.Bisection(), xrtol = T(1e-7))) 
    catch e
        msg = "Impossible to compute the jacobi scale at rhost = " * r_host * " Mpc for " * string(hp) *  " | c200 (planck18) = " * string(cΔ_from_ρs(ρs, hp, 200, planck18)) * " [min, med, max] = " * _to_bisect.([1e-10, sqrt(1e-10 * 1e+4), 1e+4]) * "\n" * e.msg
        throw(ArgumentError(msg))
    end

    return res
end


jacobi_scale(r_host::T, ρs::T, hp::HaloProfile, host::HostInterpolationType{T}) where {T<:AbstractFloat} = jacobi_scale(r_host, ρs, hp, ρ_host_spherical(r_host, host), m_host_spherical(r_host, host))
jacobi_scale(r_host::T, subhalo::HaloType{T}, ρ_host::T, m_host::T) where {T<:AbstractFloat}  = jacobi_scale(r_host, subhalo.ρs, subhalo.hp,  ρ_host, m_host)
jacobi_scale(r_host::T, subhalo::HaloType{T}, host::HostInterpolationType{T}) where {T<:AbstractFloat} = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)
jacobi_radius(r_host::T, subhalo::HaloType{T}, host::HostInterpolationType{T}) where {T<:AbstractFloat}  = subhalo.rs * jacobi_scale(r_host, subhalo.ρs, subhalo.hp, host)


jacobi_scale_DM_only(r_host::T, subhalo::HaloType{T}, host::HostInterpolationType{T}) where {T<:AbstractFloat} = jacobi_scale(r_host, subhalo.ρs, subhalo.hp, ρ_halo(r_host, host.halo), m_halo(r_host, host.halo))
jacobi_radius_DM_only(r_host::T, subhalo::HaloType{T}, host::HostInterpolationType{T}) where {T<:AbstractFloat} = subhalo.rs * jacobi_scale_DM_only(r_host, subhalo, host)

