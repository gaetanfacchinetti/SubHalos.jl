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

export tidal_scale, MCResult, mc_save, mc_load, append!, convert

function _tidal_scale( 
    xt::T,
    r_host::T,
    subhalo::H, 
    hpi::HPI,
    host::HI,
    n_stars::Int;
    z::T = T(0),
    cosmo::C = dflt_cosmo(T),
    stars::Bool = true,
    disk::Bool = true
    ) where {
        T<:AbstractFloat,
        H<:HaloType{T}, 
        HPI<:HaloProfileInterpolationType{T}, 
        HI<:HostInterpolationType{T},
        C<:Cosmology{T, <:BkgCosmology{T}}}

    arrays = allocate_one_crossing(n_stars, 20, 10, 10, T)
    n_cross = 2 * number_circular_orbits(r_host, host, z, cosmo.bkg)
    v_subhalo_kms = rand_3D_velocity_kms(1, r_host, host)[:, 1]

    norm_v_subhalo_kms = LinearAlgebra.norm(v_subhalo_kms)
    θ = acos(v_subhalo_kms[3] / norm_v_subhalo_kms)

    xt_array = Vector{T}(undef, 0)

    append!(xt_array, tidal_scale_one_crossing!(xt, r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays, stars, disk))
    (xt_array[end] <= T(1e-2)) && (return xt_array, θ, norm_v_subhalo_kms)
    
    for i in 2:n_cross
        v_subhalo_kms .= - v_subhalo_kms
        append!(xt_array, tidal_scale_one_crossing!(xt_array[end], r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays, stars, disk))
        (xt_array[end] <= T(1e-2)) && (return xt_array, θ, norm_v_subhalo_kms)
    end

    return xt_array, θ, norm_v_subhalo_kms
end


function tidal_scale_hist( 
    r_host::T,
    m200::T,
    hpi::HPI,
    host::HI;
    n = 1,
    z::T = T(0),
    cosmo::C = dflt_cosmo(T),
    stars::Bool = true,
    disk::Bool = true,
    c200::T = -T(1),
    n_stars::Int = 0,
    n_cross::Int = 0
    ) where {
        T<:AbstractFloat,
        HPI<:HaloProfileInterpolationType{T}, 
        HI<:HostInterpolationType{T},
        C<:Cosmology{T, <:BkgCosmology{T}}}


    # evaluate the number of stellar encounters per crossings and total number of crossings
    n_stars = n_stars > 0 ? n_stars : number_stellar_encounters(r_host, host)
    n_cross = n_cross > 0 ? n_cross : 2 * number_circular_orbits(r_host, host, z, cosmo.bkg)

    # draw a given subhalo from the concentration distribution
    if c200 < T(0)
        c_array = rand_concentration(n, m200, z)
    else
        c_array = [c200 for _ ∈ 1:n]
    end

    # allocate memory (to avoid memory leakage)
    arrays = allocate_one_crossing(n_stars, 20, 10, 10, T)

    # prepare output arrays
    xt_array = Vector{Vector{T}}(undef, 0)
    mt_array = Vector{Vector{T}}(undef, 0)
    xt_one_sub = Vector{T}(undef, 0)
    mt_one_sub = Vector{T}(undef, 0)
    θ_array  = Vector{T}(undef, 0)
    v_kms_array = Vector{T}(undef, 0)

    for i ∈ 1:n

        # initialise the xt for this subhalo
        xt_one_sub = []
        mt_one_sub = []
        
        # create the subhalo object from the input mass and drawn concentration
        subhalo = halo_from_mΔ_and_cΔ(hpi.hp, m200, c_array[i])

        # get the jacobi scale first
        xt = jacobi_scale(r_host, subhalo, host)

        # draw a 3D velocity for the subhalo and infer norm and direction
        v_subhalo_kms = rand_3D_velocity_kms(1, r_host, host)[:, 1]
        norm_v_subhalo_kms = LinearAlgebra.norm(v_subhalo_kms)
        θ = acos(v_subhalo_kms[3] / norm_v_subhalo_kms)

        _xt, _μt, _ = tidal_scale_one_crossing!(xt, r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays, stars, disk)
        _mt = 4 * π * subhalo.rs^3 * subhalo.ρs * _μt
        
        append!(xt_one_sub, _xt)
        append!(mt_one_sub, _mt)

        (_xt <= T(1e-2)) && continue
        
        for i in 2:n_cross
            v_subhalo_kms .= - v_subhalo_kms
            
            _xt, _μt, _ = tidal_scale_one_crossing!(xt_one_sub[end], r_host, v_subhalo_kms, subhalo, hpi, host, n_stars, arrays, stars, disk)
            _mt = 4 * π * subhalo.rs^3 * subhalo.ρs * _μt

            append!(xt_one_sub, _xt)
            append!(mt_one_sub, _mt)

            (_xt <= T(1e-2)) && break
        end

        # add the new values to the collections
        push!(xt_array, xt_one_sub)
        push!(mt_array, mt_one_sub)
        push!(θ_array, θ)
        push!(v_kms_array, norm_v_subhalo_kms)

    end


    return xt_array, mt_array, θ_array, v_kms_array, c_array

end


Base.@kwdef struct MCResult{T<:AbstractFloat, U<:Int}
    
    r_host::T
    m200::T

    stars::Bool
    disk::Bool

    xt::Vector{T}
    mt::Vector{T}
    θ::Vector{T}
    v::Vector{T}
    c::Vector{T}
    
    ns::Vector{U}
    nc::Vector{U}
    nt::Vector{U}
    np::Vector{Vector{U}}

end

function Base.append!(res::MCResult{T, U}, add::MCResult{V, U}) where {T, V, U}
    
    for name in fieldnames(MCResult)
        
        f1 = getfield(res, name)
        f2 = getfield(add, name)
        
        if f1 isa AbstractVector
            
            if V != T && f2 isa AbstractVector{V}
                append!(f1, T.(f2))   # append whole vector contents
            else
                append!(f1, f2)
            end
        end

    end

    return res

end

function Base.convert(::Type{T}, res::MCResult{V, U}) where {T, V, U}
    
    if T == V
        return res
    end

    params = Dict{Symbol, Any}()

    for field in fieldnames(MCResult)

        value = getfield(res, field)
        
        if value isa Union{V, AbstractVector{V}}
            params[field] = T.(value)
        else
            params[field] = value
        end
        
    end

    return MCResult{T, U}(; params...)

end

function mc_save(filename::String, result::MCResult)

    try
        JLD2.jldsave(filename; Dict(field => getfield(result, field)  for field in fieldnames(MCResult))...)
    catch e
        println("Impossible to save the result of the monte carlo")
        rethrow(e)
    end

end

function mc_load(filename::String)

    try
        data = JLD2.jldopen(filename)
        return MCResult(; Dict(Symbol(k) => data[k] for k in keys(data))...)
    catch e
        println("Impossible to load the result of the monte carlo")
        rethrow(e)
    end

end



function tidal_scale( 
    r_host::T,
    m200::T,
    hpi::HPI,
    host::HI,
    n::Int = 1,
    disk::Bool = true,
    stars::Bool = true,
    z::T = T(0),
    cosmo::C = dflt_cosmo(T);
    nx::Int = 16,
    nψ::Int = 8,
    nφ::Int = 16,
    n_cross::Int = -1,
    n_stars::Int = -1,
    v_kms::T = -T(1),
    seed::Int = -1,
    ) where {
        T<:AbstractFloat,
        HPI<:HaloProfileInterpolationType{T}, 
        HI<:HostInterpolationType{T},
        C<:Cosmology{T, <:BkgCosmology{T}}}


    # fix the random seed
    if seed > 0
        Random.seed!(seed)
    end

    # draw a given subhalo from the concentration distribution
    c_array = rand_concentration(n, m200, z)

    # initialise 3 velocity vectors to v_kms in input
    # assuming that they only have a component normal to the disk
    v_subhalo_kms = fill(v_kms, 3, n)
    v_subhalo_kms[1, :] .= T(0)
    v_subhalo_kms[2, :] .= T(0)
    v_kms_array = fill(v_kms, n)

    # if the value of v_kms < 0 (as default) draw realistic velocities
    if v_kms < 0
        # draw a 3D velocity for the subhalo and infer norm and direction
        v_subhalo_kms .= rand_3D_velocity_kms(n, r_host, host)
        v_kms_array .= sqrt.(sum(abs2, v_subhalo_kms; dims=1))[1, :]
    end
    
    # evaluate the angle of the subhalo direction with the normal of the disk
    θ_array = acos.(v_subhalo_kms[3, :] ./ v_kms_array)

    # evaluate the number of stellar encounters per crossings
    ns_array = fill(n_stars, n)
    
    if n_stars < 0
        ns_array .= round.(Int, number_stellar_encounters(r_host, host) .* 0.5 ./ abs.(cos.(θ_array)))
    end

    # default number of crossings
    nc_array = fill(n_cross, n)

    # prepare output arrays
    xt_array = Vector{T}(undef, n)
    mt_array = Vector{T}(undef, n)
    nt_array = Vector{Int}(undef, n)
    np_array = Vector{Vector{Int}}(undef, n)
    mt_one_sub = zero(T)


    # main loop of the Monte Carlo over different 
    # realisation of subhalos properties
    for i ∈ 1:n

        # allocate memory (to avoid memory leakage)
        arrays = allocate_one_crossing(ns_array[i], nx, nψ, nφ, T)
        
        # create the subhalo object from the input mass and drawn concentration
        subhalo = halo_from_mΔ_and_cΔ(hpi.hp, m200, c_array[i])

        # get the jacobi scale first
        xt_one_sub = jacobi_scale(r_host, subhalo, host)

        # evaluate total number of crossing from the velocity
        if n_cross < 0
            period = 2 * π * convert_lengths(r_host, MegaParsecs, KiloMeters) / v_kms_array[i] # in seconds
            nc_array[i] = 2 * floor(Int, convert_times(age_host(z, host, cosmo.bkg), GigaYears, Seconds) / period)
        end

        # if no crossings of the disk we simply output the jacobi radius
        # if no contributions from disk shocking or stars output the jacobi radius
        if nc_array[i] == 0 || (!disk && !stars)
            xt_array[i] = xt_one_sub
            mt_array[i] = 4 * π * subhalo.rs^3 * subhalo.ρs * μ_halo(xt_one_sub, hpi.hp)
            nt_array[i] = 0
            continue
        end

        # number of crossings
        j = 0

        # initialise a temporary holder for 
        # the number of penetrative encounters
        np_array_temp = Vector{Int}(undef, 0)

        #println("$xt_one_sub $r_host $(ns_array[i]) $m200 $(nc_array[i]) $(v_kms_array[i]) $(c_array[i])")

        # loop on disk crossings
        while j < nc_array[i] && xt_one_sub > T(1e-2)

            xt_one_sub, μt_one_sub, np_one_sub = tidal_scale_one_crossing!(
                xt_one_sub, 
                r_host, 
                v_subhalo_kms[:, i], 
                subhalo, 
                hpi, 
                host, 
                ns_array[i], 
                arrays, 
                disk, 
                stars)

            # add the number of penetrative encounters
            push!(np_array_temp, np_one_sub)

            # tidal mass corresponding to the tidal scale
            mt_one_sub = 4 * π * subhalo.rs^3 * subhalo.ρs * μt_one_sub

            # reverse velocity for next crossing
            # x and y component do not change because on the second
            # crossing the basis has turned
            v_subhalo_kms[3, i] = - v_subhalo_kms[3, i]

            # increment the number of crossings
            j = j+1

        end

        # add the new values to the collections
        xt_array[i] = xt_one_sub
        mt_array[i] = mt_one_sub
        nt_array[i] = j
        np_array[i] = np_array_temp
     
    end

    return MCResult(
        r_host = r_host, 
        m200 = m200, 
        stars = stars, 
        disk = disk, 
        xt = xt_array, 
        mt = mt_array, 
        θ = θ_array,
        v = v_kms_array,
        c = c_array,
        ns = ns_array, 
        nc = nc_array, 
        nt = nt_array, 
        np = np_array)

end