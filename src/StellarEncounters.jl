export pseudo_mass, solve_xt_new

###################################
_pseudo_mass(bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = 1.0 - quadgk(lnx-> sqrt(exp(lnx)^2 - bs^2)  * ρ_halo(exp(lnx), shp) * exp(lnx)^2 , log(bs), log(xt), rtol=1e-10)[1] / μ_halo(xt, shp)

function pseudo_mass(bs::Real, xt::Real, shp::HaloProfile = nfwProfile)

    (xt <= bs)  && return 1.0
    (bs < 1e-5) && return _pseudo_mass(bs, xt, shp)

    ((typeof(shp) <: αβγProfile) && (shp == plummerProfile)) && return (1.0 - (1.0 / (1.0 + bs^2)) * (1 - bs^2 / (xt^2))^(1.5))

    if ((typeof(shp) <: αβγProfile) && (shp == nfwProfile))
        (bs > 1)  && return 1.0 + (sqrt(xt * xt - bs * bs) / (1 + xt) - acosh(xt / bs) + (2. / sqrt(bs * bs - 1)) * atan(sqrt((bs - 1) / (bs + 1)) * tanh(0.5 * acosh(xt / bs)))) / μ_halo(xt, shp)
        (bs == 1) && return 1.0 - (-2 * sqrt((xt - 1) / (xt + 1)) + 2 * asinh(sqrt((xt - 1) / 2))) / μ_halo(xt, shp)
        (bs < 1)  && return 1.0 + (sqrt(xt * xt - bs * bs) / (1 + xt) - acosh(xt / bs) + (2. / sqrt(1 - bs * bs)) * atanh(sqrt((1 - bs) / (bs + 1)) * tanh(0.5 * acosh(xt / bs)))) / μ_halo(xt, shp)
    end 

    # For whatever different profile
    return _pseudo_mass(bs, xt, shp)

end

pseudo_mass(b::Real, rt::Real, sh::Halo) = pseudo_mass(b/sh.rs, rt/sh.rs, sh.hp)

####################################

""" result in (Msol^{-1}) from Chabrier 2003 """
function stellar_mass_function_C03(m::Real)
    
    (m <= 1) && (return 0.158 * exp(-(log10(m) - log10(0.079))^2 / (2. * 0.69^2)) / m / 0.6046645064846679) 
    (0 < log10(m) && log10(m) <= 0.54) && (return 4.4e-2 *  m^(-5.37) / 0.6046645064846679)
    (0.54 < log10(m) && log10(m) <= 1.26) && (return 1.5e-2 * m^(-4.53) / 0.6046645064846679)
    (1.26 < log10(m) && log10(m) <= 1.80) && (return 2.5e-4 * m^(-3.11) / 0.6046645064846679)

    return 0
end

function moments_C03(n::Int)
    return quadgk(lnm -> exp(lnm)^(n+1) * stellar_mass_function_C03(exp(lnm)), log(1e-7), log(10.0^1.8), rtol=1e-10)[1] 
end 

function b_max(r_host::Real, host::HostModel)
    return moments_C03(1)^(1/3)/σ_stars(r_host, host) * quadgk(lnz -> exp(lnz) * ρ_stars(r, exp(lnz), host)^(2/3), log(1e-10), log(1e+0), rtol=1e-10)[1] 
end


####################################
