abstract type Distribution{T<:AbstractFloat} end


###################
## Simple power law

struct PowerLaw{T<:AbstractFloat} <: Distribution{T}
    α::T
    min::T
    max::T
end

function (dist::PowerLaw)(x::T) where {T<:AbstractFloat}

    if x >= dist.min && x <= dist.max
        return -(dist.α+1) * dist.max^(-(dist.α + 1)) * 1 / ( (dist.max/dist.min)^(-(dist.α + 1)) - 1) * x^(dist.α)
    end

    return zero(T)

end

cdf(dist::PowerLaw{T}, x::T) where {T<:AbstractFloat} = ( (x/dist.min)^(dist.α - 1) - 1) / ( (dist.max/dist.min)^(dist.α - 1) - 1)

function inv_cdf(dist::PowerLaw{T}, u::T) where {T<:AbstractFloat}
    
    @assert (u >= 0 && u <= 1) "Argument u should be between 0 and 1"

    y = (dist.max / dist.min)^(-(dist.α +1))

    return dist.min .* (y/(y*(1-u) +  u)).^(1/(-(dist.α +1)))

end


function rand(dist::PowerLaw{T}, n::Int = 1, rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    
    r = Random.rand(rng, T, n)
    res = Vector{T}(undef, n)

    @inbounds for i in 1:n
        res[i] = inv_cdf(dist, r[i])
    end

    return res
end

function rand!(dist::PowerLaw{T}, x::AbstractArray{T}, rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    
    Random.rand!(rng, x)
    
    for i in eachindex(x)
        x[i] = inv_cdf(dist, x[i])
    end

    return x
end

###################
## Double power law

struct DoublePowerExp{T<:AbstractFloat} <: Distribution{T}
    γ_1::T
    α_1::T
    γ_2::T
    α_2::T
    β::T
    ζ::T
end


###################
## (Natural) log normal


struct LogNormal{T<:AbstractFloat} <: Distribution{T}
    med::T
    σ::T
end


function (dist::LogNormal{T})(x::T) where {T<:AbstractFloat}
    K = SpecialFunctions.erfc( - dist.med / (sqrt(2) * dist.σ) ) / 2 # normalisation constant
    return 1 / K / x / sqrt(2 * π) / dist.σ * exp(-(log(x) - dist.med)^2 / 2 / dist.σ^2)
end

cdf(dist::SubHalos.LogNormal{T}, x::T) where {T<:AbstractFloat} = SpecialFunctions.erfc( - (log(x)  - dist.med) / sqrt(2) / dist.σ ) / 2
cdf_inv(dist::SubHalos.LogNormal{T}, y::T) where {T<:AbstractFloat} = exp(dist.med - sqrt(2) * dist.σ * SpecialFunctions.erfcinv(2*y))

function rand(dist::LogNormal{T}, n::Int = 1, rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    
    r = Random.randn(rng, T, n)
    res = Vector{T}(undef, n)

    @inbounds for i in 1:n
        res[i] = exp(muladd(dist.σ, r[i], dist.med))
    end

    return res
end


function rand!(dist::LogNormal{T}, x::AbstractArray{T}, rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    Random.randn!(rng, x)
    x .= exp.(muladd.(dist.σ, x, dist.med))
end


pdf(dist::Distribution{T}, x::T) where {T<:AbstractFloat} = dist(x)
